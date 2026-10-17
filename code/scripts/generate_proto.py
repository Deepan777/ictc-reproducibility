"""Generate and verify ICTC Protobuf bindings with the locked grpcio-tools package."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO_SOURCE = ROOT / "proto" / "ictc.proto"
OUTPUT_DIRECTORY = ROOT / "src" / "ictc" / "proto"
GENERATED_FILENAMES = ("ictc_pb2.py", "ictc_pb2.pyi", "ictc_pb2_grpc.py")


def generate(output_directory: Path) -> None:
    """Generate bindings into an output directory and normalize package-relative imports."""
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_SOURCE.parent}",
        f"--python_out={output_directory}",
        f"--pyi_out={output_directory}",
        f"--grpc_python_out={output_directory}",
        str(PROTO_SOURCE),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    grpc_module = output_directory / "ictc_pb2_grpc.py"
    grpc_text = grpc_module.read_text(encoding="utf-8")
    normalized = grpc_text.replace(
        "import ictc_pb2 as ictc__pb2",
        "from . import ictc_pb2 as ictc__pb2",
    )
    grpc_module.write_text(normalized, encoding="utf-8", newline="\n")


def check_generated() -> None:
    """Raise when generated bindings differ from the reproducible generator output."""
    with tempfile.TemporaryDirectory(prefix="ictc-proto-") as temporary:
        generated_directory = Path(temporary)
        generate(generated_directory)
        for filename in GENERATED_FILENAMES:
            expected = (generated_directory / filename).read_bytes()
            actual_path = OUTPUT_DIRECTORY / filename
            if not actual_path.is_file() or actual_path.read_bytes() != expected:
                raise SystemExit(
                    f"generated binding is stale: {actual_path.relative_to(ROOT)}; "
                    "run `uv run python scripts/generate_proto.py`"
                )


def main() -> int:
    """Generate bindings by default, or verify them with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify without modifying generated files"
    )
    arguments = parser.parse_args()
    if arguments.check:
        check_generated()
    else:
        generate(OUTPUT_DIRECTORY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
