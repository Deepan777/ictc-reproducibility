# Changelog

## comnet-revision-2026-10-v5

- Make a truncated simple-path diagnostic non-fatal to back-door identification.
  The diagnostic can also be disabled explicitly.
- Add an optional single-set mode that finds one inclusion-minimal available
  adjustment set, including mandatory context. Exhaustive identification remains
  the default. Single-set planning supplies only that alternative to acquisition
  synthesis; it does not optimize across all possible adjustment alternatives.
- Add margin, persistence and learned-ensemble admission filters, with separate
  validation selection, persisted decisions and paired outcome analysis.

The filter utilities are in `analysis_tools/admission_filters/`. The synthetic
graph-equivalence checker is `analysis_tools/check_single_equivalence.py`.
Compatibility tests use synthetic fixtures and do not require research archives
or repository history. Existing licence and dependency terms are unchanged.
