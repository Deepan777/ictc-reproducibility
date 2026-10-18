# ICTC research implementation

Implementation and reproduction utilities for Intervention-Conditioned Telemetry
Contracts for Risk-Constrained Network Control, maintained by
[Deepan777](https://github.com/Deepan777).

## Contents

- `code/src/ictc`: causal-evidence contracts, admission, telemetry handling and controllers.
- `code/ns3/scratch`: ns-3 simulation scenarios and native packet experiments.
- `reproduction`: controlled-study runners, packet-workspace preparation, model
  reconstruction, endpoint analysis and memory measurement.
- `configs` and `code/configs`: experiment and method configurations.
- `CMakeLists.txt`: native ns-3 integration build.

## Installation and reproduction scope

Use CPython 3.12. Install the Python package and its declared dependencies with
`python -m pip install ./code`. Run `ictc --help` for the command interface.
The native experiments require a separately obtained ns-3.46.1 build and the
compiler environment specified in the manuscript. Configure the native build with
`cmake -S . -B build -DNS3_ROOT=PATH_TO_NS3`, then `cmake --build build`.

Each reproduction utility exposes its required inputs through `--help`, for example
`python reproduction/run_controlled.py --help` and
`python reproduction/prepare_packet_work.py --help`. Use new output directories;
do not overwrite retained experiment evidence. The controller-safe observations,
outcomes and coefficient CSVs required by those commands are separate inputs.
The manuscript's Supplementary Reproducibility Data attachment supplies the
declared numerical inputs and development projections. Topology Zoo and MAWI
inputs must be obtained from their respective providers.

This code-only repository does not contain manuscript text, prompts, reviewer
correspondence, figures, generated experiment results, fitted binary bundles,
complete raw packet archives or external datasets. Code availability alone does
not constitute a complete native simulation-campaign rerun.

## Licences

The Python controller, analysis and reproduction utilities, configurations, build
scripts and original repository documentation are released under the MIT License;
see the root `LICENSE` and `code/LICENSE`.

The ns-3 integration sources under `code/ns3/` are released under GNU GPL version 2
only (`GPL-2.0-only`); see `code/ns3/LICENSE` and their SPDX notices. This scoped
licence accommodates ns-3's GPLv2-only licensing framework. The root MIT licence
does not override that exception. Third-party packages and externally obtained
software and datasets retain their own licences. No licence to a manuscript,
dataset, figure or other material absent from this repository is granted here.

Both software licences permit research and commercial use under their respective
terms and disclaim warranties. Redistributed ns-3-linked works must comply with
the applicable GPLv2 requirements.

## Version and citation

The open-licensed release is `comnet-revision-2026-10-18-v4`.
Earlier versioned snapshots are preserved; the v4 changes concern licensing,
package metadata and this repository overview, not controller behaviour.

```bibtex
@misc{ictcsoftware,
  author = {{Deepan777}},
  title = {{ICTC} reproducibility implementation},
  year = {2026},
  howpublished = {GitHub software repository},
  url = {https://github.com/Deepan777/ictc-reproducibility/tree/comnet-revision-2026-10-18-v4},
  note = {Version comnet-revision-2026-10-18-v4; MIT, with GPL-2.0-only ns-3 integration}
}
```
