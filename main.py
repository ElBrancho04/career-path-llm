from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_to_syspath() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _run_module_file(path: Path, argv: Sequence[str]) -> int:
    """Run a python file as __main__ with a custom argv."""
    old_argv = sys.argv
    try:
        sys.argv = [str(path)] + list(argv)
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        code: Any = getattr(exc, "code", 0)
        return int(code) if isinstance(code, int) else 1
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="career-path-llm",
        description="Unified CLI entrypoint for the career-path-llm project.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # 1) Run a single instance/variant (wraps src/main.py)
    p_run = sub.add_parser(
        "run",
        help="Run a single instance with a chosen variant and algorithm.",
    )
    p_run.add_argument("--instance", required=True, help="Path to instance JSON file.")
    p_run.add_argument("--variant", required=True, choices=["A", "B", "C", "D"], help="Variant to execute.")
    p_run.add_argument(
        "--algorithm",
        default="greedy",
        choices=["greedy", "a_star"],
        help="Base algorithm (for variants A–D).",
    )
    p_run.add_argument(
        "--objective",
        required=True,
        help="Objective text for B, or comma-separated skills for A/C/D.",
    )
    p_run.add_argument("--use_ollama", action="store_true", help="Enable Ollama calls.")
    p_run.add_argument("--output", help="Optional path to save result JSON.")

    # 2) Run the experiment matrix (wraps experiments/run_experiments.py)
    p_exp = sub.add_parser(
        "experiments",
        help="Run the experiment matrix and write results under results/.",
    )
    p_exp.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to experiments/run_experiments.py (prefix with --).",
    )

    # 3) Check Ollama config (wraps scripts/check_ollama.py)
    p_oll = sub.add_parser(
        "check-ollama",
        help="Validate Ollama endpoint/model configuration.",
    )
    p_oll.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to scripts/check_ollama.py (prefix with --).",
    )

    # 4) Where are key paths
    sub.add_parser(
        "paths",
        help="Print key repository paths (results, report, config, instances).",
    )

    return parser


def cmd_paths() -> int:
    root = _repo_root()
    print(f"repo_root: {root}")
    print(f"config:    {root / 'config'}")
    print(f"instances: {root / 'data' / 'instances'}")
    print(f"results:   {root / 'results'}")
    print(f"report:    {root / 'report'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _add_repo_to_syspath()

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _repo_root()

    if args.command == "paths":
        return cmd_paths()

    if args.command == "run":
        target = root / "src" / "main.py"
        forward = []
        forward += ["--instance", args.instance]
        forward += ["--variant", args.variant]
        forward += ["--algorithm", args.algorithm]
        forward += ["--objective", args.objective]
        if args.use_ollama:
            forward += ["--use_ollama"]
        if args.output:
            forward += ["--output", args.output]
        return _run_module_file(target, forward)

    if args.command == "experiments":
        target = root / "experiments" / "run_experiments.py"
        forward = list(args.args or [])
        if forward and forward[0] == "--":
            forward = forward[1:]
        return _run_module_file(target, forward)

    if args.command == "check-ollama":
        target = root / "scripts" / "check_ollama.py"
        forward = list(args.args or [])
        if forward and forward[0] == "--":
            forward = forward[1:]
        return _run_module_file(target, forward)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
