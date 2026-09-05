#!/usr/bin/env python3
"""Parses one or more JUnit XML reports into a markdown summary and appends it
to $GITHUB_STEP_SUMMARY. Works with any JUnit-XML producer (Maven Surefire,
jest-junit, pytest --junitxml, Cypress's bundled mocha-junit-reporter) since
they all share the same <testsuite tests= failures= errors= skipped=> shape,
with or without an outer <testsuites> wrapper.

Usage: python3 summarize_junit.py "<glob-pattern>" ["<glob-pattern>" ...]
"""
import glob
import os
import sys
import xml.etree.ElementTree as ET


def testsuites_in(root):
    if root.tag == "testsuites":
        return list(root.findall("testsuite"))
    if root.tag == "testsuite":
        return [root]
    return []


def main():
    patterns = sys.argv[1:]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p, recursive=True)))

    lines = ["## Test Summary", ""]

    if not files:
        lines.append(f"No JUnit XML reports found matching: {', '.join(patterns)}")
        write(lines)
        return

    total = failures = errors = skipped = 0
    failing_names = []

    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        for suite in testsuites_in(tree.getroot()):
            total += int(suite.get("tests", 0) or 0)
            failures += int(suite.get("failures", 0) or 0)
            errors += int(suite.get("errors", 0) or 0)
            skipped += int(suite.get("skipped", 0) or 0)
            for case in suite.findall("testcase"):
                if case.find("failure") is not None or case.find("error") is not None:
                    classname = case.get("classname", "")
                    name = case.get("name", "")
                    failing_names.append(f"{classname} › {name}" if classname else name)

    passed = max(0, total - failures - errors - skipped)
    failed = failures + errors

    lines.append(
        f"**{total} total** — ✅ {passed} passed, ❌ {failed} failed, ⏭️ {skipped} skipped"
    )
    lines.append("")

    if failing_names:
        lines.append("### Failing tests")
        for n in failing_names:
            lines.append(f"- `{n}`")
    else:
        lines.append("_All tests passed._")
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
