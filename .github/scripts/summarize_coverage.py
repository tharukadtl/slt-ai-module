#!/usr/bin/env python3
"""Parses a Cobertura-format coverage.xml (pytest-cov's --cov-report=xml output) and appends a
labeled section to $GITHUB_STEP_SUMMARY, distinct from summarize_junit.py's pass/fail summary.

Usage: python3 summarize_coverage.py <path-to-coverage.xml>
"""
import os
import sys
import xml.etree.ElementTree as ET


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: summarize_coverage.py <path-to-coverage.xml>")

    path = sys.argv[1]
    lines = ["## AI Tests -- Coverage", ""]

    if not os.path.exists(path):
        lines.append(f"No coverage report found at {path}.")
        write(lines)
        return

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        lines.append(f"{path} could not be parsed as XML.")
        write(lines)
        return

    line_rate = float(root.get("line-rate", 0))
    branch_rate = float(root.get("branch-rate", 0))
    lines_covered = root.get("lines-covered")
    lines_valid = root.get("lines-valid")

    lines.append(
        f"**{line_rate * 100:.1f}% line coverage**, {branch_rate * 100:.1f}% branch coverage"
        + (f" ({lines_covered}/{lines_valid} lines)" if lines_covered and lines_valid else "")
    )
    lines.append("")
    lines.append("Full HTML report available in this run's `coverage-report` artifact.")
    lines.append("")

    write(lines)


def write(lines):
    text = "\n".join(lines) + "\n"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
