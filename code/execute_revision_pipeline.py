"""Run all frozen revision stages, continuing only after integrity-valid inputs."""
from __future__ import annotations
import time
from pathlib import Path
from fit_revision_models import fit, select
from run_revision_native import ROOT, run, read
from analyze_revision import revision


def main() -> None:
    for topology in ("abilene", "geant-2012-03", "garr-2011-09"):
        expected = len(read(ROOT / f"configs/{topology}-development-matrix.json")["scenarios"])
        groups = ROOT / f"experiments/{topology}/development/groups"
        while len(list(groups.glob("*/group_manifest.json"))) < expected:
            if list(groups.glob("*/**/failure.json")):
                raise RuntimeError(f"retained development failure requires diagnosis: {topology}")
            time.sleep(5)
        if not (ROOT / f"analysis/models/{topology}/bundle.joblib").exists():
            fit(topology)
        run(topology, "validation", workers=4)
        if not (ROOT / f"analysis/models/{topology}/selection.json").exists():
            select(topology)
        run(topology, "test", workers=4)
        if not (ROOT / f"analysis/test/{topology}/analysis.json").exists():
            revision(topology)
    run("geant-stress", "test", workers=4)
    revision("geant-stress")
    print("ALL_FROZEN_REVISION_NATIVE_EXPERIMENTS_AND_ANALYSES_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
