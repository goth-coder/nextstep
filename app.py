from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker-compose.yml"


def _run(cmd: list[str], dry_run: bool = False) -> int:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _run_docker(args: argparse.Namespace) -> int:
    if not _has_command("docker"):
        raise RuntimeError("Docker not found in PATH")

    if not COMPOSE_FILE.exists():
        raise FileNotFoundError(f"docker-compose file not found: {COMPOSE_FILE}")

    if args.no_cache:
        code = _run(["docker", "compose", "build", "--no-cache"], dry_run=args.dry_run)
        if code != 0:
            return code

    up_cmd = ["docker", "compose", "up", "--build"]
    if args.detach:
        up_cmd.append("-d")

    return _run(up_cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NextStep runner (Docker-only): starts containers with Docker Compose."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build all compose images with --no-cache before up",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run docker compose up -d --build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        return _run_docker(args)
    except KeyboardInterrupt:
        print("\nCancelled by user (Ctrl+C).")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
