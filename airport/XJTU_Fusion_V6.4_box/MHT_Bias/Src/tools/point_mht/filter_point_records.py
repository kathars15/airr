# -*- coding: utf-8 -*-
"""Filter point_records rows by azimuth range and write a new CSV."""

import argparse
import csv
import os


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def in_azimuth_range(value, az_min, az_max):
    if value is None:
        return False
    value = value % 360.0
    az_min = az_min % 360.0
    az_max = az_max % 360.0
    if az_min <= az_max:
        return az_min <= value <= az_max
    return value >= az_min or value <= az_max


def main():
    parser = argparse.ArgumentParser(description="Filter point_records by azimuth range")
    parser.add_argument("--input", required=True, help="point_records.csv path")
    parser.add_argument("--output", required=True, help="filtered CSV output path")
    parser.add_argument("--az-min", type=float, default=120.0)
    parser.add_argument("--az-max", type=float, default=135.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    kept = 0
    total = 0
    with open(args.input, newline="", encoding="utf-8", errors="ignore") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        with open(args.output, "w", newline="", encoding="utf-8-sig") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                az = to_float(row.get("azimuth"))
                if in_azimuth_range(az, args.az_min, args.az_max):
                    writer.writerow(row)
                    kept += 1

    print(f"[filter] input: {args.input}")
    print(f"[filter] output: {args.output}")
    print(f"[filter] azimuth range: {args.az_min:.1f} - {args.az_max:.1f} deg")
    print(f"[filter] kept: {kept}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
