import os
import sys
from typing import Any

import pytest

# Monkey-patch os.uname for Windows to support lhotse/NeMo during tests
if sys.platform == "win32" and not hasattr(os, "uname"):
    from collections import namedtuple
    UnameResult = namedtuple("UnameResult", ["sysname", "nodename", "release", "version", "machine"])
    def uname():
        # Avoid platform.* calls that might recurse back to os.uname
        return UnameResult(
            sysname="Windows",
            nodename=os.environ.get("COMPUTERNAME", "unknown"),
            release=os.environ.get("OS", "unknown"),
            version="10.0.xxxx",
            machine=os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
        )
    os.uname = uname


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """
    Add a summary of test results to the GitHub Step Summary and output warnings for skips.
    This hook is executed after all tests have finished.
    """
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    
    # Collect stats from the terminal reporter
    passed = terminalreporter.stats.get("passed", [])
    skipped = terminalreporter.stats.get("skipped", [])
    failed = terminalreporter.stats.get("failed", [])
    
    # 1. Output GitHub Actions Warnings for skipped tests
    # This makes them visible in the run summary and annotations
    if skipped:
        for s in skipped:
            reason = "No reason given"
            # Robustly retrieve reason from longrepr
            if hasattr(s, "longrepr"):
                if isinstance(s.longrepr, tuple) and len(s.longrepr) >= 3:
                    reason = str(s.longrepr[2])
                elif isinstance(s.longrepr, str):
                    reason = s.longrepr
            
            # Extract location for annotation
            # s.location is typically (file_path, line_index, test_name)
            file_path, line_index, _ = s.location if hasattr(s, "location") else ("unknown", -1, "")
            
            # Print annotation command (::warning ...) to stdout
            # Line number in annotations is 1-based, location index is 0-based
            print(f"::warning file={file_path},line={line_index + 1}::Test skipped: {s.nodeid} -- {reason}")

    # 2. Write rich summary to GITHUB_STEP_SUMMARY if available
    if not github_summary:
        return

    try:
        with open(github_summary, "a", encoding="utf-8") as f:
            f.write("## 🧪 Test Execution Summary\n\n")
            
            status_icon = "✅" if exitstatus == 0 else "❌"
            status_text = "Success" if exitstatus == 0 else "Failure"
            f.write(f"**Status**: {status_icon} {status_text}\n\n")
            
            # Summary Table
            f.write("| Metric | Count |\n")
            f.write("| --- | ---: |\n")
            f.write(f"| Passed | {len(passed)} |\n")
            f.write(f"| Skipped | {len(skipped)} |\n")
            f.write(f"| Failed | {len(failed)} |\n\n")
            
            if failed:
                f.write(f"### ❌ Failed ({len(failed)})\n")
                for e in failed:
                    f.write(f"- **{e.nodeid}**\n")
                f.write("\n")
            
            if skipped:
                f.write(f"### ⚠️ Skipped ({len(skipped)})\n")
                for s in skipped:
                    # Retrieve reason again for markdown report
                    reason = "No reason given"
                    if hasattr(s, "longrepr"):
                        if isinstance(s.longrepr, tuple) and len(s.longrepr) >= 3:
                            reason = str(s.longrepr[2])
                        elif isinstance(s.longrepr, str):
                            reason = s.longrepr
                    
                    f.write(f"- **{s.nodeid}**\n  - 📝 `{reason}`\n")
                f.write("\n")

            if passed:
                f.write(f"### ✅ Passed ({len(passed)})\n")
                f.write("<details><summary>Show passed tests</summary>\n\n")
                for p in passed:
                    f.write(f"- `{p.nodeid}`\n")
                f.write("\n</details>\n\n")

    except (OSError, IOError) as e:
        # Do not fail the build if summary writing fails
        print(f"Warning: Could not write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)
