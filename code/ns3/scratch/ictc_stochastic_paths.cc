// SPDX-License-Identifier: Proprietary
// N06 stochastic heterogeneous-path paired-potential-outcome simulator.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

struct PathSpec
{
    std::string id;
    std::string rate;
    std::string delay;
};

struct WorkloadSpec
{
    std::vector<std::string> rates;
    double onMeanSeconds;
    double offMeanSeconds;
};

std::vector<PathSpec>
Topology(const std::string& family)
{
    if (family == "two-balanced")
    {
        return {{"current", "5Mbps", "5ms"}, {"candidate", "5Mbps", "1ms"}};
    }
    if (family == "two-tradeoff")
    {
        return {{"current", "8Mbps", "4ms"}, {"candidate", "3Mbps", "1ms"}};
    }
    if (family == "three-heterogeneous")
    {
        return {{"current", "6Mbps", "5ms"},
                {"candidate", "4Mbps", "1ms"},
                {"alternate", "8Mbps", "3ms"}};
    }
    NS_FATAL_ERROR("unknown topology family " << family);
    return {};
}

WorkloadSpec
Workload(const std::string& workload, std::size_t pathCount)
{
    WorkloadSpec result;
    if (workload == "light")
    {
        result = {{"600Kbps", "800Kbps", "500Kbps"}, 0.20, 0.80};
    }
    else if (workload == "bursty")
    {
        result = {{"3Mbps", "3Mbps", "2Mbps"}, 0.45, 0.45};
    }
    else if (workload == "saturated")
    {
        result = {{"7Mbps", "5Mbps", "9Mbps"}, 0.90, 0.15};
    }
    else
    {
        NS_FATAL_ERROR("unknown workload " << workload);
    }
    result.rates.resize(pathCount);
    return result;
}

void
WriteTelemetry(std::ofstream& stream,
               const std::vector<Ptr<PointToPointNetDevice>>& egressDevices,
               const std::vector<PathSpec>& paths,
               Time decisionTime,
               Time interval,
               Time stop)
{
    const auto now = Simulator::Now();
    const auto phase = now < decisionTime ? "PRE_DECISION" : "POST_DECISION";
    for (std::size_t index = 0; index < paths.size(); ++index)
    {
        stream << now.GetNanoSeconds() << ',' << phase << ',' << paths[index].id << ','
               << egressDevices[index]->GetQueue()->GetNPackets() << "\n";
    }
    if (now + interval < stop)
    {
        Simulator::Schedule(interval,
                            &WriteTelemetry,
                            std::ref(stream),
                            egressDevices,
                            paths,
                            decisionTime,
                            interval,
                            stop);
    }
}

void
WriteState(std::ofstream& stream,
           const std::vector<Ptr<PointToPointNetDevice>>& egressDevices,
           const std::vector<bool>* observedLinkUp,
           const std::vector<PathSpec>& paths,
           Time decisionTime,
           Time interval,
           Time stop)
{
    const auto now = Simulator::Now();
    const auto phase = now < decisionTime ? "PRE_DECISION" : "POST_DECISION";
    for (std::size_t index = 0; index < paths.size(); ++index)
    {
        DataRateValue rate;
        egressDevices[index]->GetAttribute("DataRate", rate);
        stream << now.GetNanoSeconds() << ',' << phase << ',' << paths[index].id << ','
               << (observedLinkUp->at(index) ? 1 : 0) << ','
               << rate.Get().GetBitRate() << "\n";
    }
    if (now + interval < stop)
    {
        Simulator::Schedule(interval,
                            &WriteState,
                            std::ref(stream),
                            egressDevices,
                            observedLinkUp,
                            paths,
                            decisionTime,
                            interval,
                            stop);
    }
}

void
SetPathFailure(Ptr<NetDevice> receiver, std::vector<bool>* observedLinkUp)
{
    auto errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(1.0));
    receiver->SetAttribute("ReceiveErrorModel", PointerValue(errorModel));
    observedLinkUp->at(0) = false;
}

void
SetRate(Ptr<PointToPointNetDevice> device, uint64_t bitsPerSecond)
{
    device->SetDataRate(DataRate(bitsPerSecond));
}

std::string
Subnet(uint32_t ordinal)
{
    return "10.1." + std::to_string(ordinal) + ".0";
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string scenarioId = "n06-development";
    std::string topologyFamily = "two-balanced";
    std::string workload = "bursty";
    std::string event = "stable";
    std::string action = "keep";
    std::string outputDirectory = "n06-output";
    uint32_t seed = 81000;
    uint64_t run = 1;
    double durationSeconds = 6.0;
    double decisionSeconds = 2.0;
    double eventSeconds = 3.0;
    double foregroundIntervalMilliseconds = 2.0;
    uint32_t maximumPackets = 1500;
    uint32_t packetSizeBytes = 768;
    bool probeOnly = false;

    CommandLine commandLine(__FILE__);
    commandLine.AddValue("scenarioId", "Immutable scenario identifier", scenarioId);
    commandLine.AddValue("topologyFamily", "Declared N06 topology family", topologyFamily);
    commandLine.AddValue("workload", "Declared seeded workload regime", workload);
    commandLine.AddValue("event", "Declared action-independent event", event);
    commandLine.AddValue("action", "Foreground path: keep, reroute, or alternate", action);
    commandLine.AddValue("output", "New directory for raw CSV output", outputDirectory);
    commandLine.AddValue("seed", "RNG seed shared by paired actions", seed);
    commandLine.AddValue("run", "RNG run shared by paired actions", run);
    commandLine.AddValue("duration", "Simulation duration in seconds", durationSeconds);
    commandLine.AddValue("decisionAt", "Foreground decision/start time in seconds", decisionSeconds);
    commandLine.AddValue("eventAt", "Action-independent event time in seconds", eventSeconds);
    commandLine.AddValue("intervalMs", "Foreground inter-packet interval", foregroundIntervalMilliseconds);
    commandLine.AddValue("maxPackets", "Foreground packet count cap", maximumPackets);
    commandLine.AddValue("packetSize", "Foreground packet size", packetSizeBytes);
    commandLine.AddValue("probeOnly", "Stop before foreground action traffic", probeOnly);
    commandLine.Parse(argc, argv);

    const auto paths = Topology(topologyFamily);
    const auto workloadSpec = Workload(workload, paths.size());
    if (scenarioId.empty() || (action != "keep" && action != "reroute" && action != "alternate"))
    {
        NS_FATAL_ERROR("scenario and action must be valid");
    }
    if (action == "alternate" && paths.size() != 3)
    {
        NS_FATAL_ERROR("alternate requires the three-path topology");
    }
    if (event != "stable" && event != "candidate-capacity-drop" &&
        event != "current-failure" && event != "candidate-cross-shift")
    {
        NS_FATAL_ERROR("unknown event " << event);
    }
    if (durationSeconds <= eventSeconds || durationSeconds <= decisionSeconds ||
        eventSeconds <= 0.5 || decisionSeconds <= 0.5 || eventSeconds == decisionSeconds ||
        foregroundIntervalMilliseconds <= 0.0 || maximumPackets == 0 || packetSizeBytes == 0)
    {
        NS_FATAL_ERROR("invalid timing or foreground controls");
    }

    const std::filesystem::path outputPath(outputDirectory);
    if (std::filesystem::exists(outputPath))
    {
        NS_FATAL_ERROR("refusing to overwrite " << outputDirectory);
    }
    std::filesystem::create_directories(outputPath);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    const uint32_t pathCount = static_cast<uint32_t>(paths.size());
    const uint32_t destinationIndex = pathCount + 1;
    NodeContainer nodes;
    nodes.Create(pathCount + 2);

    std::vector<NetDeviceContainer> sourceLinks;
    std::vector<NetDeviceContainer> destinationLinks;
    for (uint32_t index = 0; index < pathCount; ++index)
    {
        PointToPointHelper link;
        link.SetDeviceAttribute("DataRate", StringValue(paths[index].rate));
        link.SetChannelAttribute("Delay", StringValue(paths[index].delay));
        sourceLinks.push_back(link.Install(nodes.Get(0), nodes.Get(index + 1)));
        destinationLinks.push_back(link.Install(nodes.Get(index + 1), nodes.Get(destinationIndex)));
    }

    InternetStackHelper stack;
    stack.Install(nodes);

    std::vector<Ipv4InterfaceContainer> sourceInterfaces;
    std::vector<Ipv4InterfaceContainer> destinationInterfaces;
    Ipv4AddressHelper address;
    uint32_t subnet = 1;
    for (uint32_t index = 0; index < pathCount; ++index)
    {
        address.SetBase(Subnet(subnet++).c_str(), "255.255.255.0");
        sourceInterfaces.push_back(address.Assign(sourceLinks[index]));
        address.SetBase(Subnet(subnet++).c_str(), "255.255.255.0");
        destinationInterfaces.push_back(address.Assign(destinationLinks[index]));
    }

    uint32_t selectedPath = 0;
    if (action == "reroute")
    {
        selectedPath = 1;
    }
    else if (action == "alternate")
    {
        selectedPath = 2;
    }

    Ipv4StaticRoutingHelper staticRouting;
    const auto sourceIpv4 = nodes.Get(0)->GetObject<Ipv4>();
    staticRouting.GetStaticRouting(sourceIpv4)
        ->SetDefaultRoute(sourceInterfaces[selectedPath].GetAddress(1),
                          sourceIpv4->GetInterfaceForDevice(sourceLinks[selectedPath].Get(0)));
    for (uint32_t index = 0; index < pathCount; ++index)
    {
        const auto relayIpv4 = nodes.Get(index + 1)->GetObject<Ipv4>();
        staticRouting.GetStaticRouting(relayIpv4)
            ->SetDefaultRoute(destinationInterfaces[index].GetAddress(1),
                              relayIpv4->GetInterfaceForDevice(destinationLinks[index].Get(0)));
    }

    const uint16_t foregroundPort = 9000;
    UdpServerHelper foregroundServer(foregroundPort);
    auto foregroundServerApps = foregroundServer.Install(nodes.Get(destinationIndex));
    foregroundServerApps.Start(Seconds(0.25));
    foregroundServerApps.Stop(Seconds(durationSeconds));

    int64_t nextStream = 100;
    for (uint32_t index = 0; index < pathCount; ++index)
    {
        const uint16_t port = static_cast<uint16_t>(9100 + index);
        UdpServerHelper backgroundServer(port);
        auto serverApps = backgroundServer.Install(nodes.Get(destinationIndex));
        serverApps.Start(Seconds(0.20));
        serverApps.Stop(Seconds(durationSeconds));

        OnOffHelper background("ns3::UdpSocketFactory",
                               InetSocketAddress(destinationInterfaces[index].GetAddress(1), port));
        background.SetAttribute("DataRate", DataRateValue(DataRate(workloadSpec.rates[index])));
        background.SetAttribute("PacketSize", UintegerValue(1000));
        background.SetAttribute(
            "OnTime",
            StringValue("ns3::ExponentialRandomVariable[Mean=" +
                        std::to_string(workloadSpec.onMeanSeconds) + "]"));
        background.SetAttribute(
            "OffTime",
            StringValue("ns3::ExponentialRandomVariable[Mean=" +
                        std::to_string(workloadSpec.offMeanSeconds) + "]"));
        NodeContainer relay(nodes.Get(index + 1));
        auto apps = background.Install(relay);
        nextStream += background.AssignStreams(relay, nextStream);
        apps.Start(Seconds(0.25));
        apps.Stop(Seconds(durationSeconds - 0.10));
    }

    if (event == "candidate-cross-shift")
    {
        const uint16_t port = 9201;
        UdpServerHelper shiftServer(port);
        auto serverApps = shiftServer.Install(nodes.Get(destinationIndex));
        serverApps.Start(Seconds(eventSeconds - 0.10));
        serverApps.Stop(Seconds(durationSeconds));
        OnOffHelper shift("ns3::UdpSocketFactory",
                          InetSocketAddress(destinationInterfaces[1].GetAddress(1), port));
        shift.SetAttribute("DataRate", DataRateValue(DataRate("8Mbps")));
        shift.SetAttribute("PacketSize", UintegerValue(1000));
        shift.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.8]"));
        shift.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.1]"));
        NodeContainer candidateRelay(nodes.Get(2));
        auto apps = shift.Install(candidateRelay);
        shift.AssignStreams(candidateRelay, 900);
        apps.Start(Seconds(eventSeconds));
        apps.Stop(Seconds(durationSeconds - 0.10));
    }

    if (!probeOnly)
    {
        UdpClientHelper foreground(destinationInterfaces[0].GetAddress(1), foregroundPort);
        foreground.SetAttribute("MaxPackets", UintegerValue(maximumPackets));
        foreground.SetAttribute("Interval", TimeValue(MilliSeconds(foregroundIntervalMilliseconds)));
        foreground.SetAttribute("PacketSize", UintegerValue(packetSizeBytes));
        auto foregroundApps = foreground.Install(nodes.Get(0));
        foregroundApps.Start(Seconds(decisionSeconds));
        foregroundApps.Stop(Seconds(durationSeconds - 0.10));
    }

    std::vector<Ptr<PointToPointNetDevice>> egressDevices;
    for (const auto& link : destinationLinks)
    {
        egressDevices.push_back(DynamicCast<PointToPointNetDevice>(link.Get(0)));
    }
    std::vector<bool> observedLinkUp(paths.size(), true);
    std::ofstream telemetry(outputPath / "telemetry.csv");
    telemetry << "timestamp_ns,phase,path_id,queue_packets\n";
    std::ofstream state(outputPath / "state.csv");
    state << "timestamp_ns,phase,path_id,link_up,rate_bps\n";
    Simulator::Schedule(Seconds(0.50),
                        &WriteTelemetry,
                        std::ref(telemetry),
                        egressDevices,
                        paths,
                        Seconds(decisionSeconds),
                        MilliSeconds(100),
                        Seconds(probeOnly ? decisionSeconds : durationSeconds));
    Simulator::Schedule(Seconds(0.50),
                        &WriteState,
                        std::ref(state),
                        egressDevices,
                        &observedLinkUp,
                        paths,
                        Seconds(decisionSeconds),
                        MilliSeconds(100),
                        Seconds(probeOnly ? decisionSeconds : durationSeconds));

    if (event == "candidate-capacity-drop")
    {
        Simulator::Schedule(Seconds(eventSeconds), &SetRate, egressDevices[1], 1000000);
    }
    else if (event == "current-failure")
    {
        Simulator::Schedule(Seconds(eventSeconds),
                            &SetPathFailure,
                            destinationLinks[0].Get(1),
                            &observedLinkUp);
    }

    FlowMonitorHelper flowMonitorHelper;
    const auto monitor = flowMonitorHelper.InstallAll();
    Simulator::Stop(Seconds(probeOnly ? decisionSeconds : durationSeconds));
    Simulator::Run();
    monitor->CheckForLostPackets();

    const auto classifier = DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());
    std::ofstream summary(outputPath / "summary.csv");
    summary << "scenario_id,topology_family,workload,event,action,seed,run,tx_packets,rx_packets,"
               "lost_packets,mean_delay_ns,jitter_ns,throughput_bps,first_tx_ns,last_rx_ns\n";
    std::ofstream background(outputPath / "background.csv");
    background << "path_id,flow_id,tx_packets,rx_packets,lost_packets,mean_delay_ns\n";
    uint32_t foregroundRows = 0;
    for (const auto& [flowId, statistics] : monitor->GetFlowStats())
    {
        const auto tuple = classifier->FindFlow(flowId);
        const auto meanDelay = statistics.rxPackets == 0
                                   ? 0
                                   : (statistics.delaySum / statistics.rxPackets).GetNanoSeconds();
        const auto jitter = statistics.rxPackets <= 1
                                ? 0
                                : (statistics.jitterSum / (statistics.rxPackets - 1)).GetNanoSeconds();
        const auto interval = statistics.timeLastRxPacket - statistics.timeFirstTxPacket;
        const auto throughput = interval.GetNanoSeconds() > 0
                                    ? statistics.rxBytes * 8.0 / interval.GetSeconds()
                                    : 0.0;
        if (tuple.destinationPort == foregroundPort)
        {
            ++foregroundRows;
            summary << scenarioId << ',' << topologyFamily << ',' << workload << ',' << event << ','
                    << action << ',' << seed << ',' << run << ',' << statistics.txPackets << ','
                    << statistics.rxPackets << ',' << statistics.lostPackets << ',' << meanDelay << ','
                    << jitter << ',' << std::fixed << std::setprecision(3) << throughput << ','
                    << statistics.timeFirstTxPacket.GetNanoSeconds() << ','
                    << statistics.timeLastRxPacket.GetNanoSeconds() << "\n";
        }
        else if (tuple.destinationPort >= 9100 && tuple.destinationPort <= 9201)
        {
            const auto pathIndex = tuple.destinationPort == 9201 ? 1 : tuple.destinationPort - 9100;
            background << paths[pathIndex].id << ',' << flowId << ',' << statistics.txPackets << ','
                       << statistics.rxPackets << ',' << statistics.lostPackets << ',' << meanDelay
                       << "\n";
        }
    }
    const uint32_t expectedForegroundRows = probeOnly ? 0 : 1;
    if (foregroundRows != expectedForegroundRows)
    {
        NS_FATAL_ERROR("unexpected foreground flow count " << foregroundRows);
    }

    std::ofstream parameters(outputPath / "parameters.json");
    parameters << "{\n"
               << "  \"action\": \"" << action << "\",\n"
               << "  \"decision_at_seconds\": " << decisionSeconds << ",\n"
               << "  \"duration_seconds\": " << durationSeconds << ",\n"
               << "  \"event\": \"" << event << "\",\n"
               << "  \"event_at_seconds\": " << eventSeconds << ",\n"
               << "  \"probe_only\": " << (probeOnly ? "true" : "false") << ",\n"
               << "  \"run\": " << run << ",\n"
               << "  \"scenario_id\": \"" << scenarioId << "\",\n"
               << "  \"seed\": " << seed << ",\n"
               << "  \"topology_family\": \"" << topologyFamily << "\",\n"
               << "  \"workload\": \"" << workload << "\"\n"
               << "}\n";

    Simulator::Destroy();
    return 0;
}
