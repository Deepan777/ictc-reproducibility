"""Narrow conversions between validated domain actions and generated Protobuf messages."""

from __future__ import annotations

from ictc.domain.enums import ActionClass, ActionKind
from ictc.domain.models import NetworkAction
from ictc.proto import ictc_pb2


def network_action_to_proto(action: NetworkAction) -> ictc_pb2.NetworkAction:
    """Convert a validated domain action into its Protobuf representation."""
    action_class = {
        ActionClass.ROUTING: ictc_pb2.ACTION_CLASS_ROUTING,
    }[action.action_class]
    action_kind = {
        ActionKind.KEEP_CURRENT_PATH: ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH,
        ActionKind.REROUTE_TO_PATH: ictc_pb2.ACTION_KIND_REROUTE_TO_PATH,
    }[action.kind]
    return ictc_pb2.NetworkAction(
        action_id=action.action_id,
        action_class=action_class,
        action_kind=action_kind,
        path_id=action.path_id or "",
    )


def network_action_from_proto(message: ictc_pb2.NetworkAction) -> NetworkAction:
    """Rebuild and revalidate a domain action received over the protocol."""
    action_class = {
        ictc_pb2.ACTION_CLASS_ROUTING: ActionClass.ROUTING,
    }.get(message.action_class)
    if action_class is None:
        raise ValueError("unsupported Protobuf action class")
    action_kind = {
        ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH: ActionKind.KEEP_CURRENT_PATH,
        ictc_pb2.ACTION_KIND_REROUTE_TO_PATH: ActionKind.REROUTE_TO_PATH,
    }.get(message.action_kind)
    if action_kind is None:
        raise ValueError("unsupported Protobuf action kind")
    return NetworkAction(
        action_id=message.action_id,
        action_class=action_class,
        kind=action_kind,
        path_id=message.path_id or None,
    )
