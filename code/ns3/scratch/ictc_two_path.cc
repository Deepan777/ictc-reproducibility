// SPDX-License-Identifier: Proprietary
// M11 wired two-path ns-3 scenario.  KEEP and REROUTE are separate paired runs.

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

using namespace ns3;

namespace
{

void
WriteTelemetry(std::ofstream& stream,
               Ptr<PointToPointNetDevice> current,
               Ptr<PointToPointNetDevice> candidate,
               Ptr<PointToPointNetDevice> alternate,
               Time interval,
               Time stop)
{
    const auto timestampNs = Simulator::Now().GetNanoSeconds();
    stream << timestampNs << ",current,"
           << current->GetQueue()->GetNPackets() << "\n";
    stream << timestampNs << ",candidate,"
           << candidate->GetQueue()->GetNPackets() << "\n";
    if (alternate)
    {
        stream << timestampNs << ",alternate,"
               << alternate->GetQueue()->GetNPackets() << "\n";
    }
    if (Simulator::Now() + interval < stop)
    {
        Simulator::Schedule(interval,
                            &WriteTelemetry,
                            std::ref(stream),
                            current,
                            candidate,
                            alternate,
                            interval,
                            stop);
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string action = "keep";
    std::string topology = "three-path";
    std::string outputDirectory = "ictc-output";
    uint32_t seed = 1;
    uint64_t run = 1;
    double durationSeconds = 6.0;
    double intervalMilliseconds = 10.0;
    uint32_t maximumPackets = 400;
    uint32_t packetSizeBytes = 512;
    double failureAtSeconds = -1.0;

    CommandLine commandLine(__FILE__);
    commandLine.AddValue("action", "Path action: keep, reroute, or alternate", action);
    commandLine.AddValue("topology", "Available path set: two-path or three-path", topology);
    commandLine.AddValue("output", "New directory for scenario CSV output", outputDirectory);
    commandLine.AddValue("seed", "RNG seed shared by paired runs", seed);
    commandLine.AddValue("run", "RNG run number shared by paired runs", run);
    commandLine.AddValue("duration", "Simulation duration in seconds", durationSeconds);
    commandLine.AddValue("intervalMs", "Foreground inter-packet interval in milliseconds", intervalMilliseconds);
    commandLine.AddValue("maxPackets", "Foreground packet count cap", maximumPackets);
    commandLine.AddValue("packetSize", "Foreground packet size in bytes", packetSizeBytes);
    commandLine.AddValue("failureAt", "Optional selected-path interface failure time in seconds", failureAtSeconds);
    commandLine.Parse(argc, argv);

    if (action != "keep" && action != "reroute" && action != "alternate")
    {
        NS_FATAL_ERROR("action must be keep, reroute, or alternate");
    }
    if (topology != "two-path" && topology != "three-path")
    {
        NS_FATAL_ERROR("topology must be two-path or three-path");
    }
    if (topology == "two-path" && action == "alternate")
    {
        NS_FATAL_ERROR("alternate action requires the three-path topology");
    }
    if (durationSeconds <= 1.0)
    {
        NS_FATAL_ERROR("duration must exceed one second");
    }
    if (intervalMilliseconds <= 0.0 || maximumPackets == 0 || packetSizeBytes == 0)
    {
        NS_FATAL_ERROR("traffic controls must be positive");
    }
    if (failureAtSeconds >= durationSeconds)
    {
        NS_FATAL_ERROR("failureAt must be negative or earlier than duration");
    }

    const std::filesystem::path outputPath(outputDirectory);
    if (std::filesystem::exists(outputPath))
    {
        NS_FATAL_ERROR("refusing to overwrite output directory " << outputDirectory);
    }
    std::filesystem::create_directories(outputPath);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    NodeContainer nodes;
    nodes.Create(topology == "three-path" ? 5 : 4);
    // source, current relay, candidate relay, [alternate relay], destination
    const uint32_t destinationNode = topology == "three-path" ? 4 : 3;

    PointToPointHelper currentLink;
    currentLink.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    currentLink.SetChannelAttribute("Delay", StringValue("5ms"));
    PointToPointHelper candidateLink;
    candidateLink.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    candidateLink.SetChannelAttribute("Delay", StringValue("1ms"));
    PointToPointHelper alternateLink;
    alternateLink.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    alternateLink.SetChannelAttribute("Delay", StringValue("3ms"));

    NetDeviceContainer sourceCurrent = currentLink.Install(nodes.Get(0), nodes.Get(1));
    NetDeviceContainer currentDestination = currentLink.Install(nodes.Get(1), nodes.Get(destinationNode));
    NetDeviceContainer sourceCandidate = candidateLink.Install(nodes.Get(0), nodes.Get(2));
    NetDeviceContainer candidateDestination = candidateLink.Install(nodes.Get(2), nodes.Get(destinationNode));
    NetDeviceContainer sourceAlternate;
    NetDeviceContainer alternateDestination;
    if (topology == "three-path")
    {
        sourceAlternate = alternateLink.Install(nodes.Get(0), nodes.Get(3));
        alternateDestination = alternateLink.Install(nodes.Get(3), nodes.Get(destinationNode));
    }

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer currentInterfaces = address.Assign(sourceCurrent);
    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer currentDestinationInterfaces = address.Assign(currentDestination);
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer candidateInterfaces = address.Assign(sourceCandidate);
    address.SetBase("10.1.4.0", "255.255.255.0");
    Ipv4InterfaceContainer candidateDestinationInterfaces = address.Assign(candidateDestination);
    address.SetBase("10.1.5.0", "255.255.255.0");
    Ipv4InterfaceContainer alternateInterfaces;
    Ipv4InterfaceContainer alternateDestinationInterfaces;
    if (topology == "three-path")
    {
        alternateInterfaces = address.Assign(sourceAlternate);
        address.SetBase("10.1.6.0", "255.255.255.0");
        alternateDestinationInterfaces = address.Assign(alternateDestination);
    }

    Ipv4StaticRoutingHelper staticRouting;
    const auto sourceIpv4 = nodes.Get(0)->GetObject<Ipv4>();
    const auto currentRelayIpv4 = nodes.Get(1)->GetObject<Ipv4>();
    const auto candidateRelayIpv4 = nodes.Get(2)->GetObject<Ipv4>();
    const auto alternateRelayIpv4 =
        topology == "three-path" ? nodes.Get(3)->GetObject<Ipv4>() : nullptr;
    const auto destinationIpv4 = nodes.Get(destinationNode)->GetObject<Ipv4>();

    const auto sourceRouting = staticRouting.GetStaticRouting(sourceIpv4);
    const bool useCurrent = action == "keep";
    const bool useCandidate = action == "reroute";
    const auto sourceDevice = useCurrent ? sourceCurrent.Get(0)
                              : (useCandidate ? sourceCandidate.Get(0) : sourceAlternate.Get(0));
    const auto nextHop = useCurrent ? currentInterfaces.GetAddress(1)
                        : (useCandidate ? candidateInterfaces.GetAddress(1)
                                        : alternateInterfaces.GetAddress(1));
    sourceRouting->SetDefaultRoute(nextHop, sourceIpv4->GetInterfaceForDevice(sourceDevice));
    staticRouting.GetStaticRouting(currentRelayIpv4)
        ->SetDefaultRoute(currentDestinationInterfaces.GetAddress(1),
                          currentRelayIpv4->GetInterfaceForDevice(currentDestination.Get(0)));
    staticRouting.GetStaticRouting(candidateRelayIpv4)
        ->SetDefaultRoute(candidateDestinationInterfaces.GetAddress(1),
                          candidateRelayIpv4->GetInterfaceForDevice(candidateDestination.Get(0)));
    if (topology == "three-path")
    {
        staticRouting.GetStaticRouting(alternateRelayIpv4)
            ->SetDefaultRoute(alternateDestinationInterfaces.GetAddress(1),
                              alternateRelayIpv4->GetInterfaceForDevice(alternateDestination.Get(0)));
    }
    staticRouting.GetStaticRouting(destinationIpv4)
        ->SetDefaultRoute(currentDestinationInterfaces.GetAddress(0),
                          destinationIpv4->GetInterfaceForDevice(currentDestination.Get(1)));

    const uint16_t port = 9000;
    UdpServerHelper server(port);
    const auto serverApplications = server.Install(nodes.Get(destinationNode));
    serverApplications.Start(Seconds(0.5));
    serverApplications.Stop(Seconds(durationSeconds));

    UdpClientHelper client(currentDestinationInterfaces.GetAddress(1), port);
    client.SetAttribute("MaxPackets", UintegerValue(maximumPackets));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(intervalMilliseconds)));
    client.SetAttribute("PacketSize", UintegerValue(packetSizeBytes));
    const auto clientApplications = client.Install(nodes.Get(0));
    clientApplications.Start(Seconds(1.0));
    clientApplications.Stop(Seconds(durationSeconds - 0.25));

    FlowMonitorHelper flowMonitorHelper;
    const auto monitor = flowMonitorHelper.InstallAll();

    std::ofstream telemetry(outputPath / "telemetry.csv");
    telemetry << "timestamp_ns,path_id,queue_packets\n";
    const auto currentDevice = DynamicCast<PointToPointNetDevice>(sourceCurrent.Get(0));
    const auto candidateDevice = DynamicCast<PointToPointNetDevice>(sourceCandidate.Get(0));
    const auto alternateDevice = topology == "three-path"
                                     ? DynamicCast<PointToPointNetDevice>(sourceAlternate.Get(0))
                                     : nullptr;
    WriteTelemetry(telemetry,
                   currentDevice,
                   candidateDevice,
                   alternateDevice,
                   MilliSeconds(100),
                   Seconds(durationSeconds));
    if (failureAtSeconds >= 0.0)
    {
        Simulator::Schedule(Seconds(failureAtSeconds),
                            &Ipv4::SetDown,
                            sourceIpv4,
                            sourceIpv4->GetInterfaceForDevice(sourceDevice));
    }

    Simulator::Stop(Seconds(durationSeconds));
    Simulator::Run();

    monitor->CheckForLostPackets();
    const auto classifier = DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());
    std::ofstream summary(outputPath / "summary.csv");
    summary << "topology,action,seed,run,flow_id,source,destination,tx_packets,rx_packets,lost_packets,"
               "mean_delay_ns,first_tx_ns,last_rx_ns\n";
    for (const auto& [flowId, statistics] : monitor->GetFlowStats())
    {
        const auto tuple = classifier->FindFlow(flowId);
        const auto meanDelay = statistics.rxPackets == 0
                                   ? 0
                                   : (statistics.delaySum / statistics.rxPackets).GetNanoSeconds();
        summary << topology << ',' << action << ',' << seed << ',' << run << ',' << flowId << ','
                << tuple.sourceAddress << ',' << tuple.destinationAddress << ','
                << statistics.txPackets << ',' << statistics.rxPackets << ','
                << statistics.lostPackets << ',' << meanDelay << ','
                << statistics.timeFirstTxPacket.GetNanoSeconds() << ','
                << statistics.timeLastRxPacket.GetNanoSeconds() << "\n";
    }

    Simulator::Destroy();
    return 0;
}
