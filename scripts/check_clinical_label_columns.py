#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def scan_directory(directory: Path) -> dict:
    csv_files = sorted(path for path in directory.iterdir() if path.suffix == ".csv")
    summary = {
        "total_csv_files": len(csv_files),
        "with_DMS_bin_score": 0,
        "with_DMS_score_bin": 0,
        "with_both": 0,
        "with_neither": 0,
    }

    for csv_file in csv_files:
        with csv_file.open(newline="") as handle:
            header = next(csv.reader(handle), [])

        has_dms_bin_score = "DMS_bin_score" in header
        has_dms_score_bin = "DMS_score_bin" in header

        if has_dms_bin_score:
            summary["with_DMS_bin_score"] += 1
        if has_dms_score_bin:
            summary["with_DMS_score_bin"] += 1
        if has_dms_bin_score and has_dms_score_bin:
            summary["with_both"] += 1
        if not has_dms_bin_score and not has_dms_score_bin:
            summary["with_neither"] += 1

    return summary


def print_summary(label: str, directory: Path, summary: dict) -> None:
    total = summary["total_csv_files"]
    print(f"{label}: {directory}")
    print(f"  total_csv_files: {total}")
    print(f"  with_DMS_bin_score: {summary['with_DMS_bin_score']} / {total}")
    print(f"  with_DMS_score_bin: {summary['with_DMS_score_bin']} / {total}")
    print(f"  with_both: {summary['with_both']} / {total}")
    print(f"  with_neither: {summary['with_neither']} / {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count clinical CSV files containing DMS_bin_score and DMS_score_bin.",
    )
    parser.add_argument(
        "--substitutions-dir",
        type=Path,
        default=Path("/network/scratch/n/noah.elrimawi-fine/ProteinGym/clinical_ProteinGym_substitutions"),
        help="Directory containing clinical substitutions CSV files.",
    )
    parser.add_argument(
        "--indels-dir",
        type=Path,
        default=Path("/network/scratch/n/noah.elrimawi-fine/ProteinGym/clinical_ProteinGym_indels"),
        help="Directory containing clinical indels CSV files.",
    )
    args = parser.parse_args()

    print_summary("Substitutions", args.substitutions_dir, scan_directory(args.substitutions_dir))
    print()
    print_summary("Indels", args.indels_dir, scan_directory(args.indels_dir))


if __name__ == "__main__":
    main()
