"""
run_pareto_on.py — Run MOLM with Pareto Loss ON across 5 seeds
================================================================
Re-trains Phase 3 (MOLM CV) and re-runs Phase 5 (generalization + Pareto)
with PARETO_LOSS=True. Saves to separate output directories so OFF results
are not overwritten.

PREREQUISITE:
    Add this line to phase0_config.py, right after the PARETO_LOSS line:
    
    Change:
        PARETO_LOSS = False
    To:
        PARETO_LOSS = os.environ.get("MOLM_PARETO_LOSS", "0") == "1"

USAGE:
    python run_pareto_on.py
"""

import subprocess, os, sys, time, json, shutil
from pathlib import Path

SEEDS = [42, 123, 456, 789, 2024]
PHASES = ["phase3_molm_cv.py", "phase5_generalization.py"]

def main():
    project_dir = Path(__file__).parent.resolve()
    base_output = project_dir / "outputs"
    log_dir = base_output / "pareto_on_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    results = {}
    total_start = time.time()

    print("=" * 70)
    print("  PARETO LOSS ON — Multi-seed run")
    print(f"  Seeds: {SEEDS}")
    print(f"  Phases: {PHASES}")
    print("=" * 70)

    for seed in SEEDS:
        seed_output = base_output / "seeds_pareto_on" / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)

        # Copy shared phase1 features (deterministic, same for all seeds)
        shared_phase1 = base_output / "phase1"
        dst_phase1 = seed_output / "phase1"
        if shared_phase1.exists() and not dst_phase1.exists():
            print(f"  Copying phase1 features to seed {seed}...")
            shutil.copytree(str(shared_phase1), str(dst_phase1))

        # Copy phase2 baselines from original seed run
        original_seed = base_output / "seeds" / f"seed_{seed}"
        dst_phase2 = seed_output / "phase2"
        if (original_seed / "phase2").exists() and not dst_phase2.exists():
            print(f"  Copying phase2 baselines to seed {seed}...")
            shutil.copytree(str(original_seed / "phase2"), str(dst_phase2))

        for phase_script in PHASES:
            phase_name = phase_script.replace(".py", "")
            log_file = log_dir / f"seed_{seed}_{phase_name}.log"

            print(f"\n{'='*70}")
            print(f"  SEED {seed} | {phase_name} | PARETO_LOSS=ON")
            print(f"{'='*70}")

            env = os.environ.copy()
            env["MOLM_SEED"] = str(seed)
            env["MOLM_OUTPUT_DIR"] = str(seed_output)
            env["MOLM_PARETO_LOSS"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            # Features already copied to seed_output; no shared dir needed

            start = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, str(project_dir / phase_script)],
                    cwd=str(project_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=7200,
                )
                elapsed = time.time() - start

                # Write log
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"SEED {seed} | {phase_name} | PARETO_LOSS=ON\n")
                    f.write(f"Return code: {proc.returncode}\n")
                    f.write(f"Elapsed: {elapsed:.1f}s\n")
                    f.write("=" * 70 + "\nSTDOUT:\n" + (proc.stdout or ""))
                    if proc.stderr:
                        f.write("\n" + "=" * 70 + "\nSTDERR:\n" + (proc.stderr or ""))

                status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
                print(f"  [{status}] {elapsed:.1f}s")

                # Print last few lines of output
                for line in (proc.stdout or "").strip().split("\n")[-5:]:
                    print(f"    {line}")

                timings[f"seed_{seed}_{phase_name}"] = elapsed
                results[f"seed_{seed}_{phase_name}"] = status

            except subprocess.TimeoutExpired:
                print(f"  [TIMEOUT] seed {seed} {phase_name}")
                results[f"seed_{seed}_{phase_name}"] = "TIMEOUT"
            except Exception as e:
                print(f"  [ERROR] {e}")
                results[f"seed_{seed}_{phase_name}"] = f"ERROR: {e}"

    total_elapsed = time.time() - total_start

    # Save timings
    timings_path = base_output / "pareto_on_timings.json"
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)

    # Print summary
    print(f"\n{'#'*70}")
    print(f"  PARETO ON RUN COMPLETE — {total_elapsed/60:.1f} min total")
    print(f"{'#'*70}")
    for key, status in results.items():
        t = timings.get(key, 0)
        print(f"  {key:<45} {status:<10} {t:.0f}s")
    print(f"\n  Outputs: {base_output / 'seeds_pareto_on'}")
    print(f"  Logs:    {log_dir}")
    print(f"\n  Next: python extract_pareto_on_results.py")


if __name__ == "__main__":
    main()
