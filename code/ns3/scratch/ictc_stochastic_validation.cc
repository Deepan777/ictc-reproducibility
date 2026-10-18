// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Deepan777
// N11 genuinely stochastic heterogeneous-path paired-potential-outcome simulator.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cmath>
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
    double rateMbps;
    double delayMs;
};

struct WorkloadSpec
{
    std::vector<double> ratesMbps;
    double onMeanSeconds;
    double offMeanSeconds;
};

struct RealizedPath
{
    std::string id;
    uint64_t initialCapacityBps;
    int64_t propagationDelayNs;
    uint32_t queueLimitPackets;
    uint64_t crossTrafficRateBps;
};

std::vector<PathSpec>
Topology(const std::string& family)
{
    if (family == "two-balanced")
    {
        return {{"current", 5.0, 5.0}, {"candidate", 5.0, 1.0}};
    }
    if (family == "two-tradeoff")
    {
        return {{"current", 8.0, 4.0}, {"candidate", 3.0, 1.0}};
    }
    if (family == "three-heterogeneous")
    {
        return {{"current", 6.0, 5.0}, {"candidate", 4.0, 1.0}, {"alternate", 8.0, 3.0}};
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
        result = {{0.6, 0.8, 0.5}, 0.20, 0.80};
    }
    else if (workload == "bursty")
    {
        result = {{3.0, 3.0, 2.0}, 0.45, 0.45};
    }
    else if (workload == "saturated")
    {
        result = {{7.0, 5.0, 9.0}, 0.90, 0.15};
    }
    else
    {
        NS_FATAL_ERROR("unknown workload " << workload);
    }
    result.ratesMbps.resize(pathCount);
    return result;
}

double
Uniform(int64_t stream, double minimum, double maximum)
{
    auto variable = CreateObject<UniformRandomVariable>();
    variable->SetStream(stream);
    return variable->GetValue(minimum, maximum);
}

class PoissonUdpApplication : public Application
{
  public:
    PoissonUdpApplication()
        : m_socket(nullptr),
          m_running(false),
          m_sent(0),
          m_maxPackets(0),
          m_packetSize(0),
          m_interval(CreateObject<ExponentialRandomVariable>())
    {
    }

    void Configure(Address peer, uint32_t maxPackets, uint32_t packetSize, double meanSeconds)
    {
        m_peer = peer;
        m_maxPackets = maxPackets;
        m_packetSize = packetSize;
        m_interval->SetAttribute("Mean", DoubleValue(meanSeconds));
    }

    int64_t AssignStreams(int64_t stream)
    {
        m_interval->SetStream(stream);
        return 1;
    }

  private:
    void StartApplication() override
    {
        m_running = true;
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        ScheduleNext();
    }

    void StopApplication() override
    {
        m_running = false;
        if (m_sendEvent.IsPending())
        {
            Simulator::Cancel(m_sendEvent);
        }
        if (m_socket)
        {
            m_socket->Close();
        }
    }

    void ScheduleNext()
    {
        if (!m_running || m_sent >= m_maxPackets)
        {
            return;
        }
        const auto seconds = std::max(0.00005, m_interval->GetValue());
        m_sendEvent = Simulator::Schedule(Seconds(seconds), &PoissonUdpApplication::Send, this);
    }

    void Send()
    {
        m_socket->Send(Create<Packet>(m_packetSize));
        ++m_sent;
        ScheduleNext();
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    bool m_running;
    uint32_t m_sent;
    uint32_t m_maxPackets;
    uint32_t m_packetSize;
    EventId m_sendEvent;
    Ptr<ExponentialRandomVariable> m_interval;
};

void
DeliverTelemetry(std::ofstream& stream,
                 std::ofstream& state,
                 const std::vector<PathSpec>& paths,
                 std::vector<uint32_t> queues,
                 std::vector<bool> linkUp,
                 std::vector<uint64_t> rates,
                 Time decisionTime)
{
    const auto now = Simulator::Now();
    const auto phase = now < decisionTime ? "PRE_DECISION" : "POST_DECISION";
    for (std::size_t index = 0; index < paths.size(); ++index)
    {
        stream << now.GetNanoSeconds() << ',' << phase << ',' << paths[index].id << ','
               << queues[index] << "\n";
        state << now.GetNanoSeconds() << ',' << phase << ',' << paths[index].id << ','
              << (linkUp[index] ? 1 : 0) << ',' << rates[index] << "\n";
    }
}

void
SampleTelemetry(std::ofstream& stream,
                 std::ofstream& state,
                 std::ofstream& audit,
                 const std::vector<Ptr<PointToPointNetDevice>>& egressDevices,
                 const std::vector<bool>* observedLinkUp,
                const std::vector<PathSpec>& paths,
                Ptr<UniformRandomVariable> lossVariable,
                Ptr<ExponentialRandomVariable> delayVariable,
                double lossProbability,
                double maximumDelaySeconds,
                Time decisionTime,
                Time interval,
                Time stop)
{
    const auto sample = Simulator::Now();
    const bool dropped = lossVariable->GetValue() < lossProbability;
    const double delaySeconds = std::min(maximumDelaySeconds, delayVariable->GetValue());
    const auto delay = Seconds(delaySeconds);
    const auto available = sample + delay;
    const auto phase = available < decisionTime ? "PRE_DECISION" : "POST_DECISION";
    audit << sample.GetNanoSeconds() << ',' << available.GetNanoSeconds() << ',' << phase << ','
          << (dropped ? 1 : 0) << ',' << delay.GetNanoSeconds() << ',' << paths.size() << "\n";
    if (!dropped && available < stop)
    {
        std::vector<uint32_t> queues;
        std::vector<uint64_t> rates;
        queues.reserve(paths.size());
        rates.reserve(paths.size());
        for (const auto& device : egressDevices)
        {
            queues.push_back(device->GetQueue()->GetNPackets());
            DataRateValue rate;
            device->GetAttribute("DataRate", rate);
            rates.push_back(rate.Get().GetBitRate());
        }
        Simulator::Schedule(delay,
                            &DeliverTelemetry,
                            std::ref(stream),
                            std::ref(state),
                            std::cref(paths),
                            queues,
                            *observedLinkUp,
                            rates,
                            decisionTime);
    }
    if (sample + interval < stop)
    {
        Simulator::Schedule(interval,
                            &SampleTelemetry,
                            std::ref(stream),
                            std::ref(state),
                            std::ref(audit),
                            std::cref(egressDevices),
                            observedLinkUp,
                            std::cref(paths),
                            lossVariable,
                            delayVariable,
                            lossProbability,
                            maximumDelaySeconds,
                            decisionTime,
                            interval,
                            stop);
    }
}

void
SetFailure(Ptr<RateErrorModel> model, std::vector<bool>* observedLinkUp)
{
    model->SetRate(1.0);
    observedLinkUp->at(0) = false;
}

void
ClearFailure(Ptr<RateErrorModel> model, std::vector<bool>* observedLinkUp)
{
    model->SetRate(0.0);
    observedLinkUp->at(0) = true;
}

void
SetRate(Ptr<PointToPointNetDevice> device, uint64_t bitsPerSecond)
{
    device->SetDataRate(DataRate(bitsPerSecond));
}

std::string
Subnet(uint32_t ordinal)
{
    return "10.2." + std::to_string(ordinal) + ".0";
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string scenarioId = "n11-validation";
    std::string topologyFamily = "two-balanced";
    std::string workload = "bursty";
    std::string event = "stable";
    std::string action = "keep";
    std::string outputDirectory = "n11-output";
    uint32_t seed = 83000;
    uint64_t run = 1;
    double durationSeconds = 6.0;
    double decisionSeconds = 3.8;
    double eventSeconds = 3.0;
    double foregroundIntervalMilliseconds = 2.0;
    uint32_t maximumPackets = 1500;
    uint32_t packetSizeBytes = 768;
    bool probeOnly = false;

    CommandLine commandLine(__FILE__);
    commandLine.AddValue("scenarioId", "Immutable scenario identifier", scenarioId);
    commandLine.AddValue("topologyFamily", "Declared topology family", topologyFamily);
    commandLine.AddValue("workload", "Declared workload regime", workload);
    commandLine.AddValue("event", "Declared action-independent event", event);
    commandLine.AddValue("action", "Foreground path: keep, reroute, or alternate", action);
    commandLine.AddValue("output", "New directory for raw CSV output", outputDirectory);
    commandLine.AddValue("seed", "RNG seed shared by paired actions", seed);
    commandLine.AddValue("run", "RNG run shared by paired actions", run);
    commandLine.AddValue("duration", "Simulation duration in seconds", durationSeconds);
    commandLine.AddValue("decisionAt", "Foreground decision/start time in seconds", decisionSeconds);
    commandLine.AddValue("eventAt", "Nominal action-independent event time", eventSeconds);
    commandLine.AddValue("intervalMs", "Nominal foreground mean inter-arrival time", foregroundIntervalMilliseconds);
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
    if (durationSeconds <= decisionSeconds || decisionSeconds <= eventSeconds + 0.3 ||
        eventSeconds <= 0.5 || foregroundIntervalMilliseconds <= 0.0 || maximumPackets == 0 ||
        packetSizeBytes == 0)
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

    std::vector<RealizedPath> realized;
    for (std::size_t index = 0; index < paths.size(); ++index)
    {
        const auto capacity = static_cast<uint64_t>(std::llround(
            paths[index].rateMbps * 1e6 * Uniform(10 + index, 0.82, 1.18)));
        const auto delay = static_cast<int64_t>(std::llround(
            paths[index].delayMs * 1e6 * Uniform(20 + index, 0.78, 1.22)));
        const auto queue = static_cast<uint32_t>(std::floor(Uniform(30 + index, 48.0, 129.0)));
        const auto cross = static_cast<uint64_t>(std::llround(
            workloadSpec.ratesMbps[index] * 1e6 * Uniform(40 + index, 0.65, 1.35)));
        realized.push_back({paths[index].id, capacity, delay, queue, cross});
    }
    const double foregroundMeanMs =
        foregroundIntervalMilliseconds * Uniform(50, 0.65, 1.35);
    const double telemetryLossProbability = Uniform(51, 0.02, 0.18);
    const double telemetryDelayMeanMs = Uniform(52, 5.0, 80.0);
    const double telemetryDelayMaximumMs = 200.0;
    const double eventJitterSeconds = Uniform(53, -0.20, 0.20);
    const double failureStartSeconds = eventSeconds + eventJitterSeconds;
    const double failureDurationSeconds = Uniform(54, 1.0, 1.6);
    const double capacityStartSeconds = eventSeconds + eventJitterSeconds;
    const uint64_t capacityDropBps = static_cast<uint64_t>(std::llround(
        realized[1].initialCapacityBps * Uniform(55, 0.12, 0.38)));
    const double crossShiftStartSeconds = eventSeconds + eventJitterSeconds;
    const double crossShiftDurationSeconds = Uniform(56, 1.2, 2.4);
    const uint64_t crossShiftRateBps = static_cast<uint64_t>(std::llround(
        8e6 * Uniform(57, 0.65, 1.35)));

    std::ofstream plan(outputPath / "stochastic_plan.json");
    plan << std::fixed << std::setprecision(9) << "{\n"
         << "  \"capacity_drop_rate_bps\": " << capacityDropBps << ",\n"
         << "  \"capacity_event_start_seconds\": " << capacityStartSeconds << ",\n"
         << "  \"cross_shift_duration_seconds\": " << crossShiftDurationSeconds << ",\n"
         << "  \"cross_shift_rate_bps\": " << crossShiftRateBps << ",\n"
         << "  \"cross_shift_start_seconds\": " << crossShiftStartSeconds << ",\n"
         << "  \"failure_duration_seconds\": " << failureDurationSeconds << ",\n"
         << "  \"failure_start_seconds\": " << failureStartSeconds << ",\n"
         << "  \"foreground_mean_interval_ms\": " << foregroundMeanMs << ",\n"
         << "  \"paths\": [\n";
    for (std::size_t index = 0; index < realized.size(); ++index)
    {
        const auto& item = realized[index];
        plan << "    {\"cross_traffic_rate_bps\": " << item.crossTrafficRateBps
             << ", \"initial_capacity_bps\": " << item.initialCapacityBps
             << ", \"path_id\": \"" << item.id << "\", \"propagation_delay_ns\": "
             << item.propagationDelayNs << ", \"queue_limit_packets\": "
             << item.queueLimitPackets << "}" << (index + 1 == realized.size() ? "\n" : ",\n");
    }
    plan << "  ],\n"
         << "  \"run\": " << run << ",\n"
         << "  \"scenario_id\": \"" << scenarioId << "\",\n"
         << "  \"schema_version\": \"n11-stochastic-plan-v1\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"telemetry_delay_maximum_ms\": " << telemetryDelayMaximumMs << ",\n"
         << "  \"telemetry_delay_mean_ms\": " << telemetryDelayMeanMs << ",\n"
         << "  \"telemetry_loss_probability\": " << telemetryLossProbability << "\n"
         << "}\n";

    const uint32_t pathCount = static_cast<uint32_t>(paths.size());
    const uint32_t destinationIndex = pathCount + 1;
    NodeContainer nodes;
    nodes.Create(pathCount + 2);

    std::vector<NetDeviceContainer> sourceLinks;
    std::vector<NetDeviceContainer> destinationLinks;
    for (uint32_t index = 0; index < pathCount; ++index)
    {
        PointToPointHelper link;
        link.SetDeviceAttribute("DataRate", DataRateValue(DataRate(realized[index].initialCapacityBps)));
        link.SetChannelAttribute("Delay", TimeValue(NanoSeconds(realized[index].propagationDelayNs)));
        link.SetQueue("ns3::DropTailQueue<Packet>",
                      "MaxSize",
                      StringValue(std::to_string(realized[index].queueLimitPackets) + "p"));
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

    uint32_t selectedPath = action == "reroute" ? 1 : 0;
    if (action == "alternate")
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
        background.SetAttribute("DataRate", DataRateValue(DataRate(realized[index].crossTrafficRateBps)));
        background.SetAttribute("PacketSize", UintegerValue(1000));
        background.SetAttribute("OnTime",
                                StringValue("ns3::ExponentialRandomVariable[Mean=" +
                                            std::to_string(workloadSpec.onMeanSeconds) + "]"));
        background.SetAttribute("OffTime",
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
        serverApps.Start(Seconds(crossShiftStartSeconds - 0.10));
        serverApps.Stop(Seconds(durationSeconds));
        OnOffHelper shift("ns3::UdpSocketFactory",
                          InetSocketAddress(destinationInterfaces[1].GetAddress(1), port));
        shift.SetAttribute("DataRate", DataRateValue(DataRate(crossShiftRateBps)));
        shift.SetAttribute("PacketSize", UintegerValue(1000));
        shift.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.8]"));
        shift.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.1]"));
        NodeContainer candidateRelay(nodes.Get(2));
        auto apps = shift.Install(candidateRelay);
        shift.AssignStreams(candidateRelay, 900);
        apps.Start(Seconds(crossShiftStartSeconds));
        apps.Stop(Seconds(std::min(durationSeconds - 0.10,
                                   crossShiftStartSeconds + crossShiftDurationSeconds)));
    }

    if (!probeOnly)
    {
        auto foreground = CreateObject<PoissonUdpApplication>();
        foreground->Configure(InetSocketAddress(destinationInterfaces[0].GetAddress(1), foregroundPort),
                              maximumPackets,
                              packetSizeBytes,
                              foregroundMeanMs / 1000.0);
        foreground->AssignStreams(700);
        nodes.Get(0)->AddApplication(foreground);
        foreground->SetStartTime(Seconds(decisionSeconds));
        foreground->SetStopTime(Seconds(durationSeconds - 0.10));
    }

    std::vector<Ptr<PointToPointNetDevice>> egressDevices;
    for (const auto& link : destinationLinks)
    {
        egressDevices.push_back(DynamicCast<PointToPointNetDevice>(link.Get(0)));
    }
    std::vector<bool> observedLinkUp(paths.size(), true);
    auto failureModel = CreateObject<RateErrorModel>();
    failureModel->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    failureModel->SetRate(0.0);
    destinationLinks[0].Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(failureModel));

    std::ofstream telemetry(outputPath / "telemetry.csv");
    telemetry << "timestamp_ns,phase,path_id,queue_packets\n";
    std::ofstream telemetryAudit(outputPath / "telemetry_delivery.csv");
    telemetryAudit << "sample_timestamp_ns,available_timestamp_ns,phase,dropped,delay_ns,path_count\n";
    std::ofstream state(outputPath / "state.csv");
    state << "timestamp_ns,phase,path_id,link_up,rate_bps\n";
    auto lossVariable = CreateObject<UniformRandomVariable>();
    lossVariable->SetStream(800);
    auto delayVariable = CreateObject<ExponentialRandomVariable>();
    delayVariable->SetAttribute("Mean", DoubleValue(telemetryDelayMeanMs / 1000.0));
    delayVariable->SetStream(801);
    const auto stopTime = Seconds(probeOnly ? decisionSeconds : durationSeconds);
    Simulator::Schedule(Seconds(0.50),
                        &SampleTelemetry,
                        std::ref(telemetry),
                        std::ref(state),
                        std::ref(telemetryAudit),
                        std::cref(egressDevices),
                        &observedLinkUp,
                        std::cref(paths),
                        lossVariable,
                        delayVariable,
                        telemetryLossProbability,
                        telemetryDelayMaximumMs / 1000.0,
                        Seconds(decisionSeconds),
                        MilliSeconds(100),
                        stopTime);
    if (event == "candidate-capacity-drop")
    {
        Simulator::Schedule(Seconds(capacityStartSeconds), &SetRate, egressDevices[1], capacityDropBps);
    }
    else if (event == "current-failure")
    {
        Simulator::Schedule(Seconds(failureStartSeconds), &SetFailure, failureModel, &observedLinkUp);
        Simulator::Schedule(Seconds(failureStartSeconds + failureDurationSeconds),
                            &ClearFailure,
                            failureModel,
                            &observedLinkUp);
    }

    FlowMonitorHelper flowMonitorHelper;
    const auto monitor = flowMonitorHelper.InstallAll();
    Simulator::Stop(stopTime);
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
               << "  \"stochastic_validation\": true,\n"
               << "  \"topology_family\": \"" << topologyFamily << "\",\n"
               << "  \"workload\": \"" << workload << "\"\n"
               << "}\n";

    Simulator::Destroy();
    return 0;
}
