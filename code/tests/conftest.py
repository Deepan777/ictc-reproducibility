"""Keep code tests distinct from checks of separately held research records."""
import pytest

ARCHIVE_TESTS = {
    'test_tracked_evidence_matches_fresh_rendering',
    'test_dynamic_ledger_is_deterministic_and_evidence_is_tracked',
    'test_tracked_n02_evidence_matches_fresh_rendering',
    'test_n04_evidence_is_deterministic_and_tracked',
    'test_tracked_analytic_fixture_is_valid',
    'test_tracked_n03_evidence_matches_fresh_rendering',
    'test_runner_persists_fresh_redecision_before_outcomes',
    'test_runner_rejects_freeze_and_redecision_tampering',
    'test_validator_rejects_incomplete_fresh_matrix',
    'test_n10_runner_seals_freeze_and_rejects_manifest_tampering',
    'test_n10_runner_rejects_source_manifest_mismatch',
    'test_n10_runner_rejects_support_and_n08_mismatches',
    'test_n12_confirmation_is_twenty_new_seeds_and_policy_bound',
    'test_n13_transport_matrix_is_complete_and_policy_locked',
    'test_n13_transport_derivations_preserve_connected_backbone_and_routes',
    'test_tracked_n01_evidence_matches_fresh_rendering',
    'test_revision_evaluation_authorization_binds_generation_outputs',
    'test_revision_evaluation_authorization_rejects_changed_review',
    'test_revision_evaluation_protocol_freezes_selected_gate_without_granting_access',
}


def pytest_addoption(parser):
    parser.addoption('--include-archive-tests', action='store_true', default=False,
                     help='Include tests requiring separately supplied research records and selected models')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--include-archive-tests'):
        return
    reason = ('Requires research records, selected models or historical review bindings excluded '
              'from the code-only export; supply those records and use --include-archive-tests')
    for item in items:
        if item.originalname in ARCHIVE_TESTS:
            item.add_marker(pytest.mark.skip(reason=reason))
