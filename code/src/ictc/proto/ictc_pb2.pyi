from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_CLASS_UNSPECIFIED: _ClassVar[ActionClass]
    ACTION_CLASS_ROUTING: _ClassVar[ActionClass]

class ActionKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_KIND_UNSPECIFIED: _ClassVar[ActionKind]
    ACTION_KIND_KEEP_CURRENT_PATH: _ClassVar[ActionKind]
    ACTION_KIND_REROUTE_TO_PATH: _ClassVar[ActionKind]

class ContractState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTRACT_STATE_UNSPECIFIED: _ClassVar[ContractState]
    CONTRACT_STATE_PROPOSED: _ClassVar[ContractState]
    CONTRACT_STATE_REQUIREMENTS_GENERATED: _ClassVar[ContractState]
    CONTRACT_STATE_OFFERS_RECEIVED: _ClassVar[ContractState]
    CONTRACT_STATE_COMMITTED: _ClassVar[ContractState]
    CONTRACT_STATE_ACTIVE: _ClassVar[ContractState]
    CONTRACT_STATE_SATISFIED: _ClassVar[ContractState]
    CONTRACT_STATE_RELEASED: _ClassVar[ContractState]
    CONTRACT_STATE_INFEASIBLE: _ClassVar[ContractState]
    CONTRACT_STATE_EXPIRED: _ClassVar[ContractState]
    CONTRACT_STATE_BREACHED: _ClassVar[ContractState]
    CONTRACT_STATE_BREACHED_BEFORE_EXECUTION: _ClassVar[ContractState]
    CONTRACT_STATE_CANCELLED: _ClassVar[ContractState]

class AdmissionDecision(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ADMISSION_DECISION_UNSPECIFIED: _ClassVar[AdmissionDecision]
    ADMISSION_DECISION_ADMIT: _ClassVar[AdmissionDecision]
    ADMISSION_DECISION_MODIFY: _ClassVar[AdmissionDecision]
    ADMISSION_DECISION_DEFER: _ClassVar[AdmissionDecision]
    ADMISSION_DECISION_REJECT: _ClassVar[AdmissionDecision]
ACTION_CLASS_UNSPECIFIED: ActionClass
ACTION_CLASS_ROUTING: ActionClass
ACTION_KIND_UNSPECIFIED: ActionKind
ACTION_KIND_KEEP_CURRENT_PATH: ActionKind
ACTION_KIND_REROUTE_TO_PATH: ActionKind
CONTRACT_STATE_UNSPECIFIED: ContractState
CONTRACT_STATE_PROPOSED: ContractState
CONTRACT_STATE_REQUIREMENTS_GENERATED: ContractState
CONTRACT_STATE_OFFERS_RECEIVED: ContractState
CONTRACT_STATE_COMMITTED: ContractState
CONTRACT_STATE_ACTIVE: ContractState
CONTRACT_STATE_SATISFIED: ContractState
CONTRACT_STATE_RELEASED: ContractState
CONTRACT_STATE_INFEASIBLE: ContractState
CONTRACT_STATE_EXPIRED: ContractState
CONTRACT_STATE_BREACHED: ContractState
CONTRACT_STATE_BREACHED_BEFORE_EXECUTION: ContractState
CONTRACT_STATE_CANCELLED: ContractState
ADMISSION_DECISION_UNSPECIFIED: AdmissionDecision
ADMISSION_DECISION_ADMIT: AdmissionDecision
ADMISSION_DECISION_MODIFY: AdmissionDecision
ADMISSION_DECISION_DEFER: AdmissionDecision
ADMISSION_DECISION_REJECT: AdmissionDecision

class ProtocolMetadata(_message.Message):
    __slots__ = ("protocol_version", "message_id", "lifecycle_id", "sent_at_unix_ms", "delivery_attempt")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    LIFECYCLE_ID_FIELD_NUMBER: _ClassVar[int]
    SENT_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_ATTEMPT_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    message_id: str
    lifecycle_id: str
    sent_at_unix_ms: int
    delivery_attempt: int
    def __init__(self, protocol_version: _Optional[str] = ..., message_id: _Optional[str] = ..., lifecycle_id: _Optional[str] = ..., sent_at_unix_ms: _Optional[int] = ..., delivery_attempt: _Optional[int] = ...) -> None: ...

class NetworkAction(_message.Message):
    __slots__ = ("action_id", "action_class", "action_kind", "path_id")
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_CLASS_FIELD_NUMBER: _ClassVar[int]
    ACTION_KIND_FIELD_NUMBER: _ClassVar[int]
    PATH_ID_FIELD_NUMBER: _ClassVar[int]
    action_id: str
    action_class: ActionClass
    action_kind: ActionKind
    path_id: str
    def __init__(self, action_id: _Optional[str] = ..., action_class: _Optional[_Union[ActionClass, str]] = ..., action_kind: _Optional[_Union[ActionKind, str]] = ..., path_id: _Optional[str] = ...) -> None: ...

class TargetOutcome(_message.Message):
    __slots__ = ("metric", "unit", "weight")
    METRIC_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    metric: str
    unit: str
    weight: float
    def __init__(self, metric: _Optional[str] = ..., unit: _Optional[str] = ..., weight: _Optional[float] = ...) -> None: ...

class EvaluationHorizon(_message.Message):
    __slots__ = ("horizon_ms",)
    HORIZON_MS_FIELD_NUMBER: _ClassVar[int]
    horizon_ms: int
    def __init__(self, horizon_ms: _Optional[int] = ...) -> None: ...

class IctcPropose(_message.Message):
    __slots__ = ("proposal_id", "baseline_action", "proposed_action", "target_outcome", "evaluation_horizon", "action_deadline_unix_ms", "causal_graph_scope", "safety_policy_scope", "fallback_action", "metadata")
    PROPOSAL_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_ACTION_FIELD_NUMBER: _ClassVar[int]
    PROPOSED_ACTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_HORIZON_FIELD_NUMBER: _ClassVar[int]
    ACTION_DEADLINE_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    CAUSAL_GRAPH_SCOPE_FIELD_NUMBER: _ClassVar[int]
    SAFETY_POLICY_SCOPE_FIELD_NUMBER: _ClassVar[int]
    FALLBACK_ACTION_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    proposal_id: str
    baseline_action: NetworkAction
    proposed_action: NetworkAction
    target_outcome: TargetOutcome
    evaluation_horizon: EvaluationHorizon
    action_deadline_unix_ms: int
    causal_graph_scope: str
    safety_policy_scope: str
    fallback_action: NetworkAction
    metadata: ProtocolMetadata
    def __init__(self, proposal_id: _Optional[str] = ..., baseline_action: _Optional[_Union[NetworkAction, _Mapping]] = ..., proposed_action: _Optional[_Union[NetworkAction, _Mapping]] = ..., target_outcome: _Optional[_Union[TargetOutcome, _Mapping]] = ..., evaluation_horizon: _Optional[_Union[EvaluationHorizon, _Mapping]] = ..., action_deadline_unix_ms: _Optional[int] = ..., causal_graph_scope: _Optional[str] = ..., safety_policy_scope: _Optional[str] = ..., fallback_action: _Optional[_Union[NetworkAction, _Mapping]] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class VariableRequirement(_message.Message):
    __slots__ = ("variable_id", "role")
    VARIABLE_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    variable_id: str
    role: str
    def __init__(self, variable_id: _Optional[str] = ..., role: _Optional[str] = ...) -> None: ...

class IctcRequire(_message.Message):
    __slots__ = ("requirement_id", "target_estimand", "required_variables", "identification_strategies", "candidate_source_classes", "max_freshness_ms", "max_clock_skew_ms", "minimum_trust", "minimum_distinct_sources", "privacy_constraint", "maximum_uncertainty_width", "metadata")
    REQUIREMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ESTIMAND_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    IDENTIFICATION_STRATEGIES_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_SOURCE_CLASSES_FIELD_NUMBER: _ClassVar[int]
    MAX_FRESHNESS_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_CLOCK_SKEW_MS_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_TRUST_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_DISTINCT_SOURCES_FIELD_NUMBER: _ClassVar[int]
    PRIVACY_CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_UNCERTAINTY_WIDTH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    requirement_id: str
    target_estimand: str
    required_variables: _containers.RepeatedCompositeFieldContainer[VariableRequirement]
    identification_strategies: _containers.RepeatedScalarFieldContainer[str]
    candidate_source_classes: _containers.RepeatedScalarFieldContainer[str]
    max_freshness_ms: int
    max_clock_skew_ms: int
    minimum_trust: float
    minimum_distinct_sources: int
    privacy_constraint: str
    maximum_uncertainty_width: float
    metadata: ProtocolMetadata
    def __init__(self, requirement_id: _Optional[str] = ..., target_estimand: _Optional[str] = ..., required_variables: _Optional[_Iterable[_Union[VariableRequirement, _Mapping]]] = ..., identification_strategies: _Optional[_Iterable[str]] = ..., candidate_source_classes: _Optional[_Iterable[str]] = ..., max_freshness_ms: _Optional[int] = ..., max_clock_skew_ms: _Optional[int] = ..., minimum_trust: _Optional[float] = ..., minimum_distinct_sources: _Optional[int] = ..., privacy_constraint: _Optional[str] = ..., maximum_uncertainty_width: _Optional[float] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class IctcOffer(_message.Message):
    __slots__ = ("offer_id", "source_id", "measurable_variables", "sampling_rates_hz", "expected_delay_ms", "max_clock_skew_ms", "cost", "bandwidth_kbps", "trust_score", "provenance_mechanism", "administrative_domain", "available_from_unix_ms", "available_until_unix_ms", "metadata")
    class SamplingRatesHzEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    OFFER_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MEASURABLE_VARIABLES_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATES_HZ_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_CLOCK_SKEW_MS_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_KBPS_FIELD_NUMBER: _ClassVar[int]
    TRUST_SCORE_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_MECHANISM_FIELD_NUMBER: _ClassVar[int]
    ADMINISTRATIVE_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FROM_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_UNTIL_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    offer_id: str
    source_id: str
    measurable_variables: _containers.RepeatedScalarFieldContainer[str]
    sampling_rates_hz: _containers.ScalarMap[str, float]
    expected_delay_ms: int
    max_clock_skew_ms: int
    cost: float
    bandwidth_kbps: float
    trust_score: float
    provenance_mechanism: str
    administrative_domain: str
    available_from_unix_ms: int
    available_until_unix_ms: int
    metadata: ProtocolMetadata
    def __init__(self, offer_id: _Optional[str] = ..., source_id: _Optional[str] = ..., measurable_variables: _Optional[_Iterable[str]] = ..., sampling_rates_hz: _Optional[_Mapping[str, float]] = ..., expected_delay_ms: _Optional[int] = ..., max_clock_skew_ms: _Optional[int] = ..., cost: _Optional[float] = ..., bandwidth_kbps: _Optional[float] = ..., trust_score: _Optional[float] = ..., provenance_mechanism: _Optional[str] = ..., administrative_domain: _Optional[str] = ..., available_from_unix_ms: _Optional[int] = ..., available_until_unix_ms: _Optional[int] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class IctcCommit(_message.Message):
    __slots__ = ("contract_id", "selected_source_ids", "selected_fields", "sampling_rates_hz", "synchronization_epoch_unix_ms", "contract_start_unix_ms", "valid_until_unix_ms", "evidence_digests", "expected_total_cost", "fallback_action", "metadata")
    class SamplingRatesHzEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    SELECTED_SOURCE_IDS_FIELD_NUMBER: _ClassVar[int]
    SELECTED_FIELDS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATES_HZ_FIELD_NUMBER: _ClassVar[int]
    SYNCHRONIZATION_EPOCH_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_START_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    VALID_UNTIL_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_DIGESTS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TOTAL_COST_FIELD_NUMBER: _ClassVar[int]
    FALLBACK_ACTION_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    contract_id: str
    selected_source_ids: _containers.RepeatedScalarFieldContainer[str]
    selected_fields: _containers.RepeatedScalarFieldContainer[str]
    sampling_rates_hz: _containers.ScalarMap[str, float]
    synchronization_epoch_unix_ms: int
    contract_start_unix_ms: int
    valid_until_unix_ms: int
    evidence_digests: _containers.RepeatedScalarFieldContainer[str]
    expected_total_cost: float
    fallback_action: NetworkAction
    metadata: ProtocolMetadata
    def __init__(self, contract_id: _Optional[str] = ..., selected_source_ids: _Optional[_Iterable[str]] = ..., selected_fields: _Optional[_Iterable[str]] = ..., sampling_rates_hz: _Optional[_Mapping[str, float]] = ..., synchronization_epoch_unix_ms: _Optional[int] = ..., contract_start_unix_ms: _Optional[int] = ..., valid_until_unix_ms: _Optional[int] = ..., evidence_digests: _Optional[_Iterable[str]] = ..., expected_total_cost: _Optional[float] = ..., fallback_action: _Optional[_Union[NetworkAction, _Mapping]] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class ClauseStatus(_message.Message):
    __slots__ = ("clause_id", "satisfied", "reason_code")
    CLAUSE_ID_FIELD_NUMBER: _ClassVar[int]
    SATISFIED_FIELD_NUMBER: _ClassVar[int]
    REASON_CODE_FIELD_NUMBER: _ClassVar[int]
    clause_id: str
    satisfied: bool
    reason_code: str
    def __init__(self, clause_id: _Optional[str] = ..., satisfied: _Optional[bool] = ..., reason_code: _Optional[str] = ...) -> None: ...

class IctcStatus(_message.Message):
    __slots__ = ("contract_id", "state", "clauses", "freshness_ms", "missing_fraction", "measured_delay_ms", "clock_skew_ms", "source_trust", "source_available", "uncertainty_width", "metadata")
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    CLAUSES_FIELD_NUMBER: _ClassVar[int]
    FRESHNESS_MS_FIELD_NUMBER: _ClassVar[int]
    MISSING_FRACTION_FIELD_NUMBER: _ClassVar[int]
    MEASURED_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    CLOCK_SKEW_MS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TRUST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    UNCERTAINTY_WIDTH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    contract_id: str
    state: ContractState
    clauses: _containers.RepeatedCompositeFieldContainer[ClauseStatus]
    freshness_ms: int
    missing_fraction: float
    measured_delay_ms: int
    clock_skew_ms: int
    source_trust: float
    source_available: bool
    uncertainty_width: float
    metadata: ProtocolMetadata
    def __init__(self, contract_id: _Optional[str] = ..., state: _Optional[_Union[ContractState, str]] = ..., clauses: _Optional[_Iterable[_Union[ClauseStatus, _Mapping]]] = ..., freshness_ms: _Optional[int] = ..., missing_fraction: _Optional[float] = ..., measured_delay_ms: _Optional[int] = ..., clock_skew_ms: _Optional[int] = ..., source_trust: _Optional[float] = ..., source_available: _Optional[bool] = ..., uncertainty_width: _Optional[float] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class IctcBreach(_message.Message):
    __slots__ = ("contract_id", "violated_clause_ids", "affected_estimand", "severity", "occurred_at_unix_ms", "recommended_fallback", "audit_digest", "metadata")
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    VIOLATED_CLAUSE_IDS_FIELD_NUMBER: _ClassVar[int]
    AFFECTED_ESTIMAND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    OCCURRED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    RECOMMENDED_FALLBACK_FIELD_NUMBER: _ClassVar[int]
    AUDIT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    contract_id: str
    violated_clause_ids: _containers.RepeatedScalarFieldContainer[str]
    affected_estimand: str
    severity: str
    occurred_at_unix_ms: int
    recommended_fallback: NetworkAction
    audit_digest: str
    metadata: ProtocolMetadata
    def __init__(self, contract_id: _Optional[str] = ..., violated_clause_ids: _Optional[_Iterable[str]] = ..., affected_estimand: _Optional[str] = ..., severity: _Optional[str] = ..., occurred_at_unix_ms: _Optional[int] = ..., recommended_fallback: _Optional[_Union[NetworkAction, _Mapping]] = ..., audit_digest: _Optional[str] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class IctcRelease(_message.Message):
    __slots__ = ("contract_id", "release_reason", "final_state", "execution_outcome", "audit_digest", "retained_artifact_locations", "metadata")
    CONTRACT_ID_FIELD_NUMBER: _ClassVar[int]
    RELEASE_REASON_FIELD_NUMBER: _ClassVar[int]
    FINAL_STATE_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    AUDIT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    RETAINED_ARTIFACT_LOCATIONS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    contract_id: str
    release_reason: str
    final_state: ContractState
    execution_outcome: str
    audit_digest: str
    retained_artifact_locations: _containers.RepeatedScalarFieldContainer[str]
    metadata: ProtocolMetadata
    def __init__(self, contract_id: _Optional[str] = ..., release_reason: _Optional[str] = ..., final_state: _Optional[_Union[ContractState, str]] = ..., execution_outcome: _Optional[str] = ..., audit_digest: _Optional[str] = ..., retained_artifact_locations: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Union[ProtocolMetadata, _Mapping]] = ...) -> None: ...

class IctcEnvelope(_message.Message):
    __slots__ = ("propose", "require", "offer", "commit", "status", "breach", "release")
    PROPOSE_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_FIELD_NUMBER: _ClassVar[int]
    OFFER_FIELD_NUMBER: _ClassVar[int]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    BREACH_FIELD_NUMBER: _ClassVar[int]
    RELEASE_FIELD_NUMBER: _ClassVar[int]
    propose: IctcPropose
    require: IctcRequire
    offer: IctcOffer
    commit: IctcCommit
    status: IctcStatus
    breach: IctcBreach
    release: IctcRelease
    def __init__(self, propose: _Optional[_Union[IctcPropose, _Mapping]] = ..., require: _Optional[_Union[IctcRequire, _Mapping]] = ..., offer: _Optional[_Union[IctcOffer, _Mapping]] = ..., commit: _Optional[_Union[IctcCommit, _Mapping]] = ..., status: _Optional[_Union[IctcStatus, _Mapping]] = ..., breach: _Optional[_Union[IctcBreach, _Mapping]] = ..., release: _Optional[_Union[IctcRelease, _Mapping]] = ...) -> None: ...
