"""
run_multiseed.py — MOLM Multi-Seed Orchestrator
=================================================
Runs Phase 1 once (deterministic features), then loops over seeds for the
stochastic phases (2 NN baselines, 3 MOLM CV, 4 holdout, 5 generalization).

Each phase runs as a fresh subprocess for clean RNG isolation. Per-seed
outputs go to outputs/seeds/seed_<N>/, while Phase 1 features stay in
outputs/phase1/ and are read via the MOLM_SHARED_OUTPUT_DIR fallback.

USAGE
-----
    # Full run: Phase 1 once + 5 seeds × phases 2/3/4/5
    python run_multiseed.py

    # Skip Phase 1 if features.pkl already exists in outputs/phase1/
    python run_multiseed.py --skip-features

    # Specific seeds only
    python run_multiseed.py --seeds 42 123

    # Specific phases only (per seed)
    python run_multiseed.py --phases 3 5

    # Resume after a failure (don't stop on first error)
    python run_multiseed.py --continue-on-error

    # Custom output root
    python run_multiseed.py --shared-output-dir D:/research/molm_outputs

EXPECTED LAYOUT AFTER RUN
-------------------------
    outputs/
    ├── phase1/                    # shared, deterministic
    │   ├── features.pkl
    │   └── esm2/{emi,iso,igg}_esm2.csv
    └── seeds/
        ├── seed_42/
        │   ├── phase2/baselines.pkl
        │   ├── phase3/molm_cv.pkl
        │   ├── phase4/holdout.pkl
        │   └── phase5/generalization.pkl
        ├── seed_123/...
        └── seed_2024/...

Then run aggregate_multiseed.py to produce paper-ready CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# DEFAULTS
# ============================================================================
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

PER_SEED_PHASES = {
    2: "phase2_baselines.py",
    3: "phase3_molm_cv.py",
    4: "phase4_holdout.py",
    5: "phase5_generalization.py",
}

PHASE1_SCRIPT = "phase1_features.py"


# ============================================================================
# HELPERS
# ============================================================================
def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def run_subprocess(script_path: Path, env: dict, cwd: Path, label: str,
                   log_path: Path | None = None) -> tuple[bool, float]:
    """Run a phase script as subprocess. Returns (success, elapsed_seconds)."""
    print(f"\n{'=' * 78}")
    print(f">>> {label}")
    print(f"    script : {script_path.name}")
    print(f"    seed   : {env.get('MOLM_SEED', '?')}")
    print(f"    out    : {env.get('MOLM_OUTPUT_DIR', '?')}")
    if log_path:
        print(f"    log    : {log_path}")
    print(f"{'=' * 78}", flush=True)

    t0 = time.time()
    try:
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f_log:
                f_log.write(f"# {label}\n# started: {datetime.now().isoformat()}\n\n")
                f_log.flush()
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(script_path)],
                    env=env, cwd=str(cwd),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, text=True, encoding="utf-8", errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    f_log.write(line)
                proc.wait()
                rc = proc.returncode
        else:
            rc = subprocess.run(
                [sys.executable, "-u", str(script_path)],
                env=env, cwd=str(cwd),
            ).returncode
    except KeyboardInterrupt:
        print(f"\n!!! Interrupted by user during: {label}", flush=True)
        raise

    elapsed = time.time() - t0
    if rc != 0:
        print(f"!!! FAIL: {label} (exit {rc}) after {_fmt_elapsed(elapsed)}", flush=True)
        return False, elapsed
    print(f"<<< OK: {label} in {_fmt_elapsed(elapsed)}", flush=True)
    return True, elapsed


def build_env(seed: int, seed_dir: Path, shared_dir: Path) -> dict:
    """Construct the environment for a per-seed phase subprocess."""
    env = os.environ.copy()
    env["MOLM_SEED"] = str(seed)
    env["MOLM_OUTPUT_DIR"] = str(seed_dir)
    env["MOLM_SHARED_OUTPUT_DIR"] = str(shared_dir)
    # ESM-2 cache stays shared (deterministic embeddings — no need to recompute).
    env["MOLM_ESM2_DIR"] = str(shared_dir / "phase1" / "esm2")
    # Force UTF-8 stdout/stderr in the child Python so Unicode glyphs (✓, 📂,
    # 🔧, etc.) in the phase scripts don't crash on Windows' default cp1252.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Some phase code reads MOLM_DATA_PATH; pass it through if set.
    return env


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-seed orchestrator for the MOLM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to run.")
    parser.add_argument("--phases", nargs="+", type=int, default=[2, 3, 4, 5],
                        choices=[2, 3, 4, 5],
                        help="Which per-seed phases to run.")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip Phase 1 (features.pkl must already exist).")
    parser.add_argument("--project-dir", type=str, default=None,
                        help="Dir containing phase scripts. Default: this script's dir.")
    parser.add_argument("--shared-output-dir", type=str, default=None,
                        help="Shared outputs dir (holds phase1 features). Default: <project>/outputs.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't abort if a phase fails — continue with next.")
    parser.add_argument("--no-log-files", action="store_true",
                        help="Don't write per-phase .log files (just stream to stdout).")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override MOLM_DATA_PATH for all phases.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir or Path(__file__).resolve().parent).resolve()
    shared_output = Path(args.shared_output_dir or (project_dir / "outputs")).resolve()
    shared_output.mkdir(parents=True, exist_ok=True)
    seeds_root = shared_output / "seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    log_root = shared_output / "multiseed_logs"

    # Sanity check
    missing = [s for s in [PHASE1_SCRIPT] + list(PER_SEED_PHASES.values())
               if not (project_dir / s).exists()]
    if missing and not args.skip_features:
        print(f"FATAL: missing phase scripts in {project_dir}:")
        for m in missing:
            print(f"    {m}")
        return 1
    elif missing:
        # If skipping features, only the per-seed phases need to exist
        ms = [s for s in PER_SEED_PHASES.values() if not (project_dir / s).exists()]
        if ms:
            print(f"FATAL: missing per-seed phase scripts in {project_dir}: {ms}")
            return 1

    print(f"\n{'#' * 78}")
    print(f"# MOLM Multi-Seed Run — started {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'#' * 78}")
    print(f"  Project dir   : {project_dir}")
    print(f"  Shared output : {shared_output}")
    print(f"  Seeds         : {args.seeds}")
    print(f"  Per-seed      : phases {args.phases}")
    print(f"  Skip features : {args.skip_features}")
    print(f"  Logs          : {'(stdout only)' if args.no_log_files else log_root}")
    if args.data_path:
        print(f"  Data path     : {args.data_path}")

    overall_t0 = time.time()
    timings = {"phase1": None, "per_seed": {}}
    failures: list[str] = []

    # ------------------------------------------------------------------
    # Phase 1: deterministic features (single run)
    # ------------------------------------------------------------------
    if not args.skip_features:
        env = os.environ.copy()
        env["MOLM_SEED"] = "42"  # phase 1 is deterministic; just pin it
        env["MOLM_OUTPUT_DIR"] = str(shared_output)
        env["MOLM_ESM2_DIR"] = str(shared_output / "phase1" / "esm2")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if args.data_path:
            env["MOLM_DATA_PATH"] = args.data_path

        log_path = None if args.no_log_files else (log_root / "phase1.log")
        ok, elapsed = run_subprocess(
            project_dir / PHASE1_SCRIPT, env, project_dir,
            label="PHASE 1: Features (single deterministic run)",
            log_path=log_path,
        )
        timings["phase1"] = elapsed
        if not ok:
            print("FATAL: Phase 1 failed. Cannot continue without features.pkl.")
            return 1
    else:
        feat_path = shared_output / "phase1" / "features.pkl"
        if not feat_path.exists():
            print(f"FATAL: --skip-features set but {feat_path} not found.")
            return 1
        print(f">>> Skipping Phase 1 ({feat_path} exists)")

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------
    for seed in args.seeds:
        seed_dir = seeds_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        timings["per_seed"][seed] = {}

        seed_env = build_env(seed, seed_dir, shared_output)
        if args.data_path:
            seed_env["MOLM_DATA_PATH"] = args.data_path

        for phase in args.phases:
            script = PER_SEED_PHASES[phase]
            label = f"SEED {seed} | PHASE {phase}: {script}"
            log_path = None if args.no_log_files else (
                log_root / f"seed_{seed}_phase{phase}.log"
            )
            ok, elapsed = run_subprocess(
                project_dir / script, seed_env, project_dir,
                label=label, log_path=log_path,
            )
            timings["per_seed"][seed][phase] = {"ok": ok, "seconds": elapsed}
            if not ok:
                failures.append(label)
                if not args.continue_on_error:
                    print("\nFATAL: aborting. Use --continue-on-error to keep going.")
                    _print_summary(timings, failures, overall_t0, args.seeds, args.phases)
                    return 2

    # ------------------------------------------------------------------
    # Summary + persist timings
    # ------------------------------------------------------------------
    _print_summary(timings, failures, overall_t0, args.seeds, args.phases)

    timings_path = shared_output / "multiseed_timings.json"
    with open(timings_path, "w") as f:
        json.dump({
            "started": datetime.fromtimestamp(overall_t0).isoformat(),
            "seeds": args.seeds,
            "phases": args.phases,
            "shared_output": str(shared_output),
            "timings": timings,
            "failures": failures,
        }, f, indent=2, default=str)
    print(f"\n  ✓ Timings saved: {timings_path}")
    print(f"\n  Next step: python aggregate_multiseed.py --output-dir {shared_output}")

    return 0 if not failures else 3


def _print_summary(timings, failures, overall_t0, seeds, phases):
    total = time.time() - overall_t0
    print(f"\n{'#' * 78}")
    print(f"# MULTI-SEED RUN SUMMARY — total wall time: {_fmt_elapsed(total)}")
    print(f"{'#' * 78}")
    if timings.get("phase1") is not None:
        print(f"  Phase 1 (features) : {_fmt_elapsed(timings['phase1'])}")
    print(f"\n  Per-seed grid (✓ OK / ✗ FAIL / · skipped):")
    header = "   seed   " + " ".join(f"P{p:>2}    " for p in phases)
    print(header)
    for seed in seeds:
        row = f"   {seed:<6} "
        for phase in phases:
            entry = timings["per_seed"].get(seed, {}).get(phase)
            if entry is None:
                row += "·       "
            elif entry["ok"]:
                row += f"✓ {_fmt_elapsed(entry['seconds']):<5} "
            else:
                row += f"✗ {_fmt_elapsed(entry['seconds']):<5} "
        print(row)
    if failures:
        print(f"\n  ⚠ {len(failures)} failure(s):")
        for f in failures:
            print(f"      - {f}")
    else:
        print("\n  ✓ All phases succeeded.")


if __name__ == "__main__":
    sys.exit(main())
