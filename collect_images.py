"""
collect_images.py — Gather all plots from all seeds into one folder
====================================================================
Copies all PNGs/PDFs from seed output folders into a single organized
directory with proper naming: phase_seedN_originalname.png

USAGE:
    python collect_images.py

OUTPUT:
    outputs/multiseed/figures/
        phase3/
            seed42_diagnostics_OneHot.png
            seed42_diagnostics_ESM2.png
            seed42_diagnostics_Fusion-ESM2.png
            seed123_diagnostics_OneHot.png
            ...
        phase5/
            seed42_pareto_diagnostics_ISO.png
            seed42_pareto_diagnostics_IgG.png
            seed123_pareto_diagnostics_ISO.png
            ...
"""

import shutil, sys
from pathlib import Path


def main() -> int:
    output_dir = Path("outputs").resolve()
    seeds_root = output_dir / "seeds"
    fig_root = output_dir / "multiseed" / "figures"

    if not seeds_root.exists():
        print(f"Seeds directory not found: {seeds_root}")
        return 1

    seeds = sorted([int(p.name.split("_")[1]) for p in seeds_root.iterdir()
                    if p.is_dir() and p.name.startswith("seed_")])

    print(f"Collecting images from {len(seeds)} seeds: {seeds}")

    total = 0
    phase_counts = {}

    for seed in seeds:
        seed_dir = seeds_root / f"seed_{seed}"

        for phase_dir in sorted(seed_dir.iterdir()):
            if not phase_dir.is_dir():
                continue

            phase_name = phase_dir.name  # e.g. "phase3", "phase5"

            # Find all image files
            images = list(phase_dir.glob("*.png")) + \
                     list(phase_dir.glob("*.pdf")) + \
                     list(phase_dir.glob("*.jpg")) + \
                     list(phase_dir.glob("*.svg"))

            if not images:
                continue

            # Create output folder
            dest_dir = fig_root / phase_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                new_name = f"seed{seed}_{img.name}"
                dest = dest_dir / new_name
                shutil.copy2(img, dest)
                total += 1

            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + len(images)
            print(f"  [seed {seed}] {phase_name}: {len(images)} images")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images collected: {total}")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} images -> {fig_root / phase}")
    print(f"\n  All saved to: {fig_root}")

    # Also list what we got
    print(f"\n{'='*60}")
    print(f"  FILE LISTING")
    print(f"{'='*60}")
    for phase_dir in sorted(fig_root.iterdir()):
        if not phase_dir.is_dir():
            continue
        print(f"\n  {phase_dir.name}/")
        for f in sorted(phase_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<55} {size_kb:>8.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
