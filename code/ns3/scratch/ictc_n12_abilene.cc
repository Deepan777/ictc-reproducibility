// SPDX-License-Identifier: Proprietary
// N12 realistic multi-hop Abilene validation with mixed TCP/UDP traffic and FQ-CoDel.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

struct EdgeSpec
{
    uint32_t left;
    uint32_t right;
    double delayMs;
};

struct RouteSpec
{
    uint32_t source;
    uint32_t target;
    std::vector<uint32_t> nodes;
};

struct EdgeRuntime
{
    EdgeSpec spec;
    NetDeviceContainer devices;
    Ipv4InterfaceContainer interfaces;
    QueueDiscContainer queueDiscs;
    uint64_t initialRateBps;
    bool up;
};

constexpr std::array<EdgeSpec, 14> EDGES = {{{0, 1, 7.161492},
                                             {0, 2, 2.053070},
                                             {1, 10, 1.645782},
                                             {2, 9, 5.449548},
                                             {3, 4, 7.116263},
                                             {3, 6, 10.256989},
                                             {4, 5, 3.144732},
                                             {4, 6, 9.397473},
                                             {5, 8, 13.792266},
                                             {6, 7, 5.573788},
                                             {7, 8, 6.512178},
                                             {7, 10, 4.566549},
                                             {8, 9, 7.047250},
                                             {9, 10, 4.297527}}};

const std::array<std::array<RouteSpec, 3>, 3> ROUTES = {
    std::array<RouteSpec, 3>{RouteSpec{3, 0, {3, 6, 7, 10, 1, 0}},
                             RouteSpec{3, 0, {3, 4, 5, 8, 9, 2, 0}},
                             RouteSpec{3, 0, {3, 4, 6, 7, 10, 1, 0}}},
    std::array<RouteSpec, 3>{RouteSpec{5, 1, {5, 8, 7, 10, 1}},
                             RouteSpec{5, 1, {5, 8, 9, 2, 0, 1}},
                             RouteSpec{5, 1, {5, 4, 6, 7, 10, 1}}},
    std::array<RouteSpec, 3>{RouteSpec{8, 2, {8, 9, 2}},
                             RouteSpec{8, 2, {8, 7, 10, 1, 0, 2}},
                             RouteSpec{8, 2, {8, 5, 4, 6, 7, 10, 9, 2}}}};

double
Uniform(int64_t stream, double minimum, double maximum)
{
    auto variable = CreateObject<UniformRandomVariable>();
    variable->SetStream(stream);
    return variable->GetValue(minimum, maximum);
}

std::string
Subnet(uint32_t ordinal)
{
    const uint32_t second = ordinal / 250;
    const uint32_t third = ordinal % 250;
    return "10." + std::to_string(10 + second) + "." + std::to_string(third) + ".0";
}

uint32_t
FindEdge(uint32_t left, uint32_t right)
{
    for (uint32_t index = 0; index < EDGES.size(); ++index)
    {
        if ((EDGES[index].left == left && EDGES[index].right == right) ||
            (EDGES[index].left == right && EDGES[index].right == left))
        {
            return index;
        }
    }
    NS_FATAL_ERROR("route references absent edge " << left << "-" << right);
    return 0;
}

double
RouteDelayMs(const RouteSpec& route, const std::vector<EdgeRuntime>& edges)
{
    double result = 0.0;
    for (std::size_t index = 0; index + 1 < route.nodes.size(); ++index)
    {
        result += edges[FindEdge(route.nodes[index], route.nodes[index + 1])].spec.delayMs;
    }
    return result;
}

uint64_t
RouteCapacityBps(const RouteSpec& route, const std::vector<EdgeRuntime>& edges)
{
    uint64_t result = std::numeric_limits<uint64_t>::max();
    for (std::size_t index = 0; index + 1 < route.nodes.size(); ++index)
    {
        const auto& edge = edges[FindEdge(route.nodes[index], route.nodes[index + 1])];
        const uint32_t side = edge.spec.left == route.nodes[index] ? 0 : 1;
        DataRateValue current;
        edge.devices.Get(side)->GetAttribute("DataRate", current);
        result = std::min(result, current.Get().GetBitRate());
    }
    return result;
}

uint64_t
RouteQueuePackets(const RouteSpec& route, const std::vector<EdgeRuntime>& edges)
{
    uint64_t result = 0;
    for (std::size_t index = 0; index + 1 < route.nodes.size(); ++index)
    {
        const auto& edge = edges[FindEdge(route.nodes[index], route.nodes[index + 1])];
        const uint32_t side = edge.spec.left == route.nodes[index] ? 0 : 1;
        result += edge.queueDiscs.Get(side)->GetNPackets();
    }
    return result;
}

bool
RouteUp(const RouteSpec& route, const std::vector<EdgeRuntime>& edges)
{
    for (std::size_t index = 0; index + 1 < route.nodes.size(); ++index)
    {
        if (!edges[FindEdge(route.nodes[index], route.nodes[index + 1])].up)
        {
            return false;
        }
    }
    return true;
}

void
DeliverTelemetry(std::ofstream& telemetry,
                 uint64_t sampleNs,
                 uint64_t availableNs,
                 const std::array<RouteSpec, 3>& routes,
                 const std::array<uint64_t, 3>& capacities,
                 const std::array<double, 3>& delays,
                 const std::array<uint64_t, 3>& queues,
                 const std::array<bool, 3>& up,
                 Time decision)
{
    const auto phase = availableNs < static_cast<uint64_t>(decision.GetNanoSeconds())
                           ? "PRE_DECISION"
                           : "POST_DECISION";
    for (uint32_t route = 0; route < routes.size(); ++route)
    {
        telemetry << sampleNs << ',' << availableNs << ',' << phase << ',' << route << ','
                  << capacities[route] << ',' << std::fixed << std::setprecision(6)
                  << delays[route] << ',' << queues[route] << ',' << (up[route] ? 1 : 0) << "\n";
    }
}

void
SampleTelemetry(std::ofstream& telemetry,
                std::ofstream& audit,
                const std::array<RouteSpec, 3>& routes,
                const std::vector<EdgeRuntime>* edges,
                Ptr<UniformRandomVariable> loss,
                Ptr<ExponentialRandomVariable> delay,
                double lossProbability,
                double maximumDelaySeconds,
                Time decision,
                Time interval,
                Time stop)
{
    const auto sample = Simulator::Now();
    const bool dropped = loss->GetValue() < lossProbability;
    const auto deliveryDelay = Seconds(std::min(maximumDelaySeconds, delay->GetValue()));
    const auto available = sample + deliveryDelay;
    audit << sample.GetNanoSeconds() << ',' << available.GetNanoSeconds() << ','
          << (available < decision ? "PRE_DECISION" : "POST_DECISION") << ','
          << (dropped ? 1 : 0) << ',' << deliveryDelay.GetNanoSeconds() << "\n";
    if (!dropped && available < stop)
    {
        std::array<uint64_t, 3> capacities{};
        std::array<double, 3> delays{};
        std::array<uint64_t, 3> queues{};
        std::array<bool, 3> up{};
        for (uint32_t route = 0; route < routes.size(); ++route)
        {
            capacities[route] = RouteCapacityBps(routes[route], *edges);
            delays[route] = RouteDelayMs(routes[route], *edges);
            queues[route] = RouteQueuePackets(routes[route], *edges);
            up[route] = RouteUp(routes[route], *edges);
        }
        Simulator::Schedule(deliveryDelay,
                            &DeliverTelemetry,
                            std::ref(telemetry),
                            static_cast<uint64_t>(sample.GetNanoSeconds()),
                            static_cast<uint64_t>(available.GetNanoSeconds()),
                            std::cref(routes),
                            capacities,
                            delays,
                            queues,
                            up,
                            decision);
    }
    if (sample + interval < stop)
    {
        Simulator::Schedule(interval,
                            &SampleTelemetry,
                            std::ref(telemetry),
                            std::ref(audit),
                            std::cref(routes),
                            edges,
                            loss,
                            delay,
                            lossProbability,
                            maximumDelaySeconds,
                            decision,
                            interval,
                            stop);
    }
}

void
SetEdgeRate(std::vector<EdgeRuntime>* edges, uint32_t edge, uint64_t rate)
{
    auto& selected = edges->at(edge);
    DynamicCast<PointToPointNetDevice>(selected.devices.Get(0))->SetDataRate(DataRate(rate));
    DynamicCast<PointToPointNetDevice>(selected.devices.Get(1))->SetDataRate(DataRate(rate));
}

void
SetEdgeState(std::vector<EdgeRuntime>* edges,
             uint32_t edge,
             Ptr<RateErrorModel> left,
             Ptr<RateErrorModel> right,
             bool up)
{
    left->SetRate(up ? 0.0 : 1.0);
    right->SetRate(up ? 0.0 : 1.0);
    edges->at(edge).up = up;
}

double
HistogramQuantile(const Histogram& histogram, uint32_t count, double quantile)
{
    if (count == 0)
    {
        return 0.0;
    }
    const uint32_t threshold = static_cast<uint32_t>(std::ceil(count * quantile));
    uint32_t cumulative = 0;
    for (uint32_t index = 0; index < histogram.GetNBins(); ++index)
    {
        cumulative += histogram.GetBinCount(index);
        if (cumulative >= threshold)
        {
            return histogram.GetBinEnd(index);
        }
    }
    return histogram.GetNBins() == 0 ? 0.0 : histogram.GetBinEnd(histogram.GetNBins() - 1);
}

std::string
NodesJson(const std::vector<uint32_t>& nodes)
{
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < nodes.size(); ++index)
    {
        stream << nodes[index] << (index + 1 == nodes.size() ? "" : ",");
    }
    stream << ']';
    return stream.str();
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string scenarioId = "n12-probe";
    std::string traffic = "web";
    std::string event = "stable";
    std::string action = "route0";
    std::string outputDirectory = "n12-output";
    uint32_t matrix = 0;
    uint32_t seed = 84000;
    uint64_t run = 1;
    double durationSeconds = 8.0;
    double decisionSeconds = 3.5;
    double eventSeconds = 2.2;
    bool probeOnly = false;

    CommandLine commandLine(__FILE__);
    commandLine.AddValue("scenarioId", "Immutable scenario identifier", scenarioId);
    commandLine.AddValue("traffic", "web, bulk, or realtime", traffic);
    commandLine.AddValue("event", "stable, failure, capacity, or cross-shift", event);
    commandLine.AddValue("matrix", "Frozen Abilene OD matrix index", matrix);
    commandLine.AddValue("action", "route0, route1, or route2", action);
    commandLine.AddValue("output", "New raw output directory", outputDirectory);
    commandLine.AddValue("seed", "RNG seed shared by paired actions", seed);
    commandLine.AddValue("run", "RNG run shared by paired actions", run);
    commandLine.AddValue("duration", "Simulation duration", durationSeconds);
    commandLine.AddValue("decisionAt", "Decision and foreground start time", decisionSeconds);
    commandLine.AddValue("eventAt", "Nominal event start time", eventSeconds);
    commandLine.AddValue("probeOnly", "End at the decision without foreground traffic", probeOnly);
    commandLine.Parse(argc, argv);

    if (scenarioId.empty() || matrix >= ROUTES.size() ||
        (traffic != "web" && traffic != "bulk" && traffic != "realtime") ||
        (event != "stable" && event != "failure" && event != "capacity" &&
         event != "cross-shift") ||
        (action != "route0" && action != "route1" && action != "route2") ||
        eventSeconds <= 0.5 || decisionSeconds <= eventSeconds + 0.5 ||
        durationSeconds <= decisionSeconds + 1.0)
    {
        NS_FATAL_ERROR("invalid N12 scenario arguments");
    }
    const uint32_t selectedRoute = static_cast<uint32_t>(action.back() - '0');
    const auto& routes = ROUTES[matrix];
    const std::filesystem::path outputPath(outputDirectory);
    if (std::filesystem::exists(outputPath))
    {
        NS_FATAL_ERROR("refusing to overwrite " << outputDirectory);
    }
    std::filesystem::create_directories(outputPath);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    Config::SetDefault("ns3::FlowMonitor::DelayBinWidth", DoubleValue(0.001));

    const double capacityMultiplier = Uniform(10, 0.88, 1.12);
    const double delayMultiplier = Uniform(11, 0.92, 1.08);
    const double loadMultiplier = Uniform(12, 0.70, 1.30);
    const double eventJitter = Uniform(13, -0.20, 0.20);
    const double eventDuration = Uniform(14, 1.6, 2.6);
    const double telemetryLoss = Uniform(15, 0.02, 0.20);
    const double telemetryMeanDelayMs = Uniform(16, 8.0, 100.0);
    const double eventStart = eventSeconds + eventJitter;

    NodeContainer routers;
    NodeContainer hosts;
    routers.Create(11);
    hosts.Create(11);
    InternetStackHelper internet;
    internet.Install(routers);
    internet.Install(hosts);

    std::vector<EdgeRuntime> edgeRuntime;
    edgeRuntime.reserve(EDGES.size());
    Ipv4AddressHelper address;
    uint32_t subnet = 1;
    for (uint32_t index = 0; index < EDGES.size(); ++index)
    {
        const auto& spec = EDGES[index];
        const uint64_t rate = static_cast<uint64_t>(std::llround(
            25e6 * capacityMultiplier * Uniform(100 + index, 0.94, 1.06)));
        const double delay = spec.delayMs * delayMultiplier * Uniform(200 + index, 0.97, 1.03);
        PointToPointHelper link;
        link.SetDeviceAttribute("DataRate", DataRateValue(DataRate(rate)));
        link.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delay)));
        link.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("4p"));
        auto devices = link.Install(routers.Get(spec.left), routers.Get(spec.right));
        TrafficControlHelper trafficControl;
        trafficControl.SetRootQueueDisc("ns3::FqCoDelQueueDisc",
                                        "MaxSize",
                                        QueueSizeValue(QueueSize("256p")),
                                        "Target",
                                        StringValue("5ms"),
                                        "Interval",
                                        StringValue("100ms"));
        auto queues = trafficControl.Install(devices);
        address.SetBase(Subnet(subnet++).c_str(), "255.255.255.252");
        auto interfaces = address.Assign(devices);
        edgeRuntime.push_back({{spec.left, spec.right, delay}, devices, interfaces, queues, rate, true});
    }

    std::array<NetDeviceContainer, 11> accessDevices;
    std::array<Ipv4InterfaceContainer, 11> accessInterfaces;
    for (uint32_t index = 0; index < hosts.GetN(); ++index)
    {
        PointToPointHelper access;
        access.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
        access.SetChannelAttribute("Delay", StringValue("500us"));
        access.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("256p"));
        accessDevices[index] = access.Install(hosts.Get(index), routers.Get(index));
        address.SetBase(Subnet(subnet++).c_str(), "255.255.255.252");
        accessInterfaces[index] = address.Assign(accessDevices[index]);
        Ipv4StaticRoutingHelper staticHelper;
        auto hostIpv4 = hosts.Get(index)->GetObject<Ipv4>();
        staticHelper.GetStaticRouting(hostIpv4)
            ->SetDefaultRoute(accessInterfaces[index].GetAddress(1),
                              hostIpv4->GetInterfaceForDevice(accessDevices[index].Get(0)));
    }
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // A host-specific route forces only traffic to the foreground destination onto the action path.
    Ipv4StaticRoutingHelper staticHelper;
    const auto& selected = routes[selectedRoute];
    const auto destination = accessInterfaces[selected.target].GetAddress(0);
    for (std::size_t index = 0; index + 1 < selected.nodes.size(); ++index)
    {
        const uint32_t current = selected.nodes[index];
        const uint32_t next = selected.nodes[index + 1];
        const auto& edge = edgeRuntime[FindEdge(current, next)];
        const uint32_t side = edge.spec.left == current ? 0 : 1;
        const uint32_t neighborSide = 1 - side;
        auto ipv4 = routers.Get(current)->GetObject<Ipv4>();
        staticHelper.GetStaticRouting(ipv4)
            ->AddHostRouteTo(destination,
                             edge.interfaces.GetAddress(neighborSide),
                             ipv4->GetInterfaceForDevice(edge.devices.Get(side)),
                             1);
    }

    const std::array<std::pair<uint32_t, uint32_t>, 6> backgroundPairs =
        {{{4, 9}, {6, 2}, {10, 3}, {0, 8}, {7, 5}, {9, 1}}};
    const double offeredMbps = (traffic == "web" ? 10.0 : traffic == "bulk" ? 20.0 : 15.0) *
                               loadMultiplier;
    const double tcpRate = offeredMbps * 0.8590475067 / 4.0;
    const double udpRate = offeredMbps * 0.1409524933 / 2.0;
    int64_t nextStream = 500;
    for (uint32_t index = 0; index < backgroundPairs.size(); ++index)
    {
        auto [source, target] = backgroundPairs[index];
        if (target == routes[0].target)
        {
            target = (target + 4) % 11;
        }
        const bool tcp = index < 4;
        const uint16_t port = static_cast<uint16_t>(10000 + index);
        PacketSinkHelper sink(tcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        auto sinkApp = sink.Install(hosts.Get(target));
        sinkApp.Start(Seconds(0.2));
        sinkApp.Stop(Seconds(durationSeconds));
        OnOffHelper onoff(tcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory",
                          InetSocketAddress(accessInterfaces[target].GetAddress(0), port));
        onoff.SetAttribute("DataRate",
                           DataRateValue(DataRate(static_cast<uint64_t>(
                               std::llround((tcp ? tcpRate : udpRate) * 1e6)))));
        onoff.SetAttribute("PacketSize", UintegerValue(tcp ? 1219 : 647));
        const double onMean = traffic == "web" ? 0.25 : traffic == "bulk" ? 0.90 : 0.40;
        const double offMean = traffic == "web" ? 0.35 : traffic == "bulk" ? 0.12 : 0.20;
        onoff.SetAttribute("OnTime",
                           StringValue("ns3::ExponentialRandomVariable[Mean=" +
                                       std::to_string(onMean) + "]"));
        onoff.SetAttribute("OffTime",
                           StringValue("ns3::ExponentialRandomVariable[Mean=" +
                                       std::to_string(offMean) + "]"));
        NodeContainer sourceNode(hosts.Get(source));
        auto app = onoff.Install(sourceNode);
        nextStream += onoff.AssignStreams(sourceNode, nextStream);
        app.Start(Seconds(0.4 + 0.03 * index));
        app.Stop(Seconds(durationSeconds - 0.1));
    }

    const uint32_t affectedEdge = FindEdge(routes[0].nodes[0], routes[0].nodes[1]);
    auto failureLeft = CreateObject<RateErrorModel>();
    auto failureRight = CreateObject<RateErrorModel>();
    failureLeft->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    failureRight->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    failureLeft->SetRate(0.0);
    failureRight->SetRate(0.0);
    edgeRuntime[affectedEdge].devices.Get(0)->SetAttribute("ReceiveErrorModel",
                                                           PointerValue(failureLeft));
    edgeRuntime[affectedEdge].devices.Get(1)->SetAttribute("ReceiveErrorModel",
                                                           PointerValue(failureRight));
    uint64_t degradedRate = edgeRuntime[affectedEdge].initialRateBps;
    double crossRateMbps = 0.0;
    if (event == "failure")
    {
        Simulator::Schedule(Seconds(eventStart),
                            &SetEdgeState,
                            &edgeRuntime,
                            affectedEdge,
                            failureLeft,
                            failureRight,
                            false);
        Simulator::Schedule(Seconds(eventStart + eventDuration),
                            &SetEdgeState,
                            &edgeRuntime,
                            affectedEdge,
                            failureLeft,
                            failureRight,
                            true);
    }
    else if (event == "capacity")
    {
        degradedRate = static_cast<uint64_t>(std::llround(
            edgeRuntime[affectedEdge].initialRateBps * Uniform(17, 0.22, 0.42)));
        Simulator::Schedule(Seconds(eventStart),
                            &SetEdgeRate,
                            &edgeRuntime,
                            affectedEdge,
                            degradedRate);
        Simulator::Schedule(Seconds(eventStart + eventDuration),
                            &SetEdgeRate,
                            &edgeRuntime,
                            affectedEdge,
                            edgeRuntime[affectedEdge].initialRateBps);
    }
    else if (event == "cross-shift")
    {
        const auto& edge = edgeRuntime[affectedEdge];
        crossRateMbps = 18.0 * Uniform(18, 0.75, 1.25);
        const uint16_t port = 11000;
        const uint32_t downstream = edge.spec.right;
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        auto sinkApp = sink.Install(routers.Get(downstream));
        sinkApp.Start(Seconds(eventStart - 0.1));
        sinkApp.Stop(Seconds(eventStart + eventDuration + 0.1));
        OnOffHelper shift("ns3::UdpSocketFactory",
                          InetSocketAddress(edge.interfaces.GetAddress(1), port));
        shift.SetAttribute("DataRate",
                           DataRateValue(DataRate(static_cast<uint64_t>(crossRateMbps * 1e6))));
        shift.SetAttribute("PacketSize", UintegerValue(647));
        shift.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        shift.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        auto shiftApp = shift.Install(routers.Get(edge.spec.left));
        shiftApp.Start(Seconds(eventStart));
        shiftApp.Stop(Seconds(eventStart + eventDuration));
    }

    const uint16_t foregroundPort = 9000;
    const bool foregroundTcp = traffic != "realtime";
    PacketSinkHelper foregroundSink(foregroundTcp ? "ns3::TcpSocketFactory"
                                                   : "ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), foregroundPort));
    auto foregroundSinkApp = foregroundSink.Install(hosts.Get(routes[0].target));
    foregroundSinkApp.Start(Seconds(0.2));
    foregroundSinkApp.Stop(Seconds(durationSeconds));
    uint64_t foregroundBytes = traffic == "web" ? 2'000'000 : traffic == "bulk" ? 8'000'000 : 0;
    if (!probeOnly)
    {
        if (foregroundTcp)
        {
            BulkSendHelper foreground("ns3::TcpSocketFactory",
                                      InetSocketAddress(destination, foregroundPort));
            foreground.SetAttribute("MaxBytes", UintegerValue(foregroundBytes));
            foreground.SetAttribute("SendSize", UintegerValue(1219));
            auto app = foreground.Install(hosts.Get(routes[0].source));
            app.Start(Seconds(decisionSeconds));
            app.Stop(Seconds(durationSeconds - 0.05));
        }
        else
        {
            foregroundBytes = static_cast<uint64_t>((durationSeconds - decisionSeconds - 0.1) *
                                                    5e6 / 8.0);
            OnOffHelper foreground("ns3::UdpSocketFactory",
                                   InetSocketAddress(destination, foregroundPort));
            foreground.SetAttribute("DataRate", StringValue("5Mbps"));
            foreground.SetAttribute("PacketSize", UintegerValue(647));
            foreground.SetAttribute("MaxBytes", UintegerValue(foregroundBytes));
            foreground.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            foreground.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
            auto app = foreground.Install(hosts.Get(routes[0].source));
            app.Start(Seconds(decisionSeconds));
            app.Stop(Seconds(durationSeconds - 0.05));
        }
    }

    std::ofstream telemetry(outputPath / "telemetry.csv");
    telemetry << "sample_timestamp_ns,available_timestamp_ns,phase,route_id,bottleneck_capacity_bps,"
                 "propagation_delay_ms,queue_packets,link_up\n";
    std::ofstream telemetryAudit(outputPath / "telemetry_delivery.csv");
    telemetryAudit << "sample_timestamp_ns,available_timestamp_ns,phase,dropped,delay_ns\n";
    auto lossVariable = CreateObject<UniformRandomVariable>();
    lossVariable->SetStream(800);
    auto delayVariable = CreateObject<ExponentialRandomVariable>();
    delayVariable->SetAttribute("Mean", DoubleValue(telemetryMeanDelayMs / 1000.0));
    delayVariable->SetStream(801);
    const auto stop = Seconds(probeOnly ? decisionSeconds : durationSeconds);
    Simulator::Schedule(Seconds(0.4),
                        &SampleTelemetry,
                        std::ref(telemetry),
                        std::ref(telemetryAudit),
                        std::cref(routes),
                        &edgeRuntime,
                        lossVariable,
                        delayVariable,
                        telemetryLoss,
                        0.250,
                        Seconds(decisionSeconds),
                        MilliSeconds(100),
                        stop);

    FlowMonitorHelper flowHelper;
    auto monitor = flowHelper.InstallAll();
    Simulator::Stop(stop);
    Simulator::Run();
    monitor->CheckForLostPackets();

    const auto classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::ofstream summary(outputPath / "summary.csv");
    summary << "scenario_id,traffic,event,matrix,action,seed,run,protocol,tx_packets,rx_packets,"
               "lost_packets,tx_bytes,rx_bytes,delivery_ratio,mean_delay_ms,p95_delay_ms,"
               "p99_delay_ms,jitter_ms,throughput_mbps,first_tx_seconds,last_rx_seconds,"
               "observed_transfer_seconds,completion_ratio,queue_drops\n";
    uint32_t foregroundRows = 0;
    uint64_t totalQueueDrops = 0;
    for (const auto& edge : edgeRuntime)
    {
        totalQueueDrops += edge.queueDiscs.Get(0)->GetStats().nTotalDroppedPackets;
        totalQueueDrops += edge.queueDiscs.Get(1)->GetStats().nTotalDroppedPackets;
    }
    for (const auto& [flowId, statistics] : monitor->GetFlowStats())
    {
        const auto tuple = classifier->FindFlow(flowId);
        if (tuple.destinationPort != foregroundPort)
        {
            continue;
        }
        ++foregroundRows;
        const double delivery = statistics.txPackets == 0
                                    ? 0.0
                                    : static_cast<double>(statistics.rxPackets) / statistics.txPackets;
        const double meanDelay = statistics.rxPackets == 0
                                     ? 0.0
                                     : statistics.delaySum.GetSeconds() * 1000.0 / statistics.rxPackets;
        const double jitter = statistics.rxPackets <= 1
                                  ? 0.0
                                  : statistics.jitterSum.GetSeconds() * 1000.0 /
                                        (statistics.rxPackets - 1);
        const double span = statistics.rxPackets == 0
                                ? 0.0
                                : (statistics.timeLastRxPacket - statistics.timeFirstTxPacket).GetSeconds();
        const double throughput = span <= 0.0 ? 0.0 : statistics.rxBytes * 8.0 / span / 1e6;
        const double completion = foregroundBytes == 0
                                      ? 0.0
                                      : std::min(1.0,
                                                 static_cast<double>(statistics.rxBytes) /
                                                     foregroundBytes);
        summary << scenarioId << ',' << traffic << ',' << event << ',' << matrix << ',' << action
                << ',' << seed << ',' << run << ',' << (foregroundTcp ? "tcp-cubic" : "udp")
                << ',' << statistics.txPackets << ',' << statistics.rxPackets << ','
                << statistics.lostPackets << ',' << statistics.txBytes << ',' << statistics.rxBytes
                << ',' << std::fixed << std::setprecision(9) << delivery << ',' << meanDelay << ','
                << HistogramQuantile(statistics.delayHistogram, statistics.rxPackets, 0.95) * 1000.0
                << ',' << HistogramQuantile(statistics.delayHistogram, statistics.rxPackets, 0.99) *
                              1000.0
                << ',' << jitter << ',' << throughput << ','
                << statistics.timeFirstTxPacket.GetSeconds() << ','
                << statistics.timeLastRxPacket.GetSeconds() << ',' << span << ',' << completion << ','
                << totalQueueDrops << "\n";
    }
    if (foregroundRows != (probeOnly ? 0u : 1u))
    {
        NS_FATAL_ERROR("unexpected foreground flow count " << foregroundRows);
    }

    std::ofstream plan(outputPath / "plan.json");
    plan << std::fixed << std::setprecision(9) << "{\n"
         << "  \"action\": \"" << action << "\",\n"
         << "  \"affected_edge\": " << affectedEdge << ",\n"
         << "  \"capacity_multiplier\": " << capacityMultiplier << ",\n"
         << "  \"cross_shift_rate_mbps\": " << crossRateMbps << ",\n"
         << "  \"degraded_rate_bps\": " << degradedRate << ",\n"
         << "  \"event\": \"" << event << "\",\n"
         << "  \"event_duration_seconds\": " << eventDuration << ",\n"
         << "  \"event_start_seconds\": " << eventStart << ",\n"
         << "  \"load_multiplier\": " << loadMultiplier << ",\n"
         << "  \"matrix\": " << matrix << ",\n"
         << "  \"routes\": [" << NodesJson(routes[0].nodes) << ',' << NodesJson(routes[1].nodes)
         << ',' << NodesJson(routes[2].nodes) << "],\n"
         << "  \"run\": " << run << ",\n"
         << "  \"scenario_id\": \"" << scenarioId << "\",\n"
         << "  \"schema_version\": \"n12-native-plan-v1\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"telemetry_loss_probability\": " << telemetryLoss << ",\n"
         << "  \"telemetry_mean_delay_ms\": " << telemetryMeanDelayMs << ",\n"
         << "  \"traffic\": \"" << traffic << "\"\n"
         << "}\n";

    Simulator::Destroy();
    return 0;
}
