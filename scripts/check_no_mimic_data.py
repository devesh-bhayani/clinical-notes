"""Post-write hook: checks that a written file contains no MIMIC/PHI signatures."""

import re
import sys

MIMIC_PATTERNS = [
    re.compile(r"(?i)admission\s+date\s*:"),
    re.compile(r"(?i)discharge\s+date\s*:"),
    re.compile(r"(?i)date\s+of\s+birth\s*:"),
    re.compile(r"(?i)attending\s*:"),
    re.compile(r"(?i)service\s*:\s*(medicine|surgery|icu|cardiology|neurology)"),
    re.compile(r"(?i)chief\s+complaint\s*:"),
    re.compile(r"(?i)social\s+security\s*(number|#|no)"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
    re.compile(r"(?i)medical\s+record\s*(number|#|no)"),
]

BLOCKED_PATH_FRAGMENTS = ["mimic", "noteevents", "discharge_summaries"]


def check_file(filepath: str) -> list[str]:
    violations = []

    # Check filename
    lower_path = filepath.lower().replace("\\", "/")
    for fragment in BLOCKED_PATH_FRAGMENTS:
        if fragment in lower_path:
            violations.append(f"Blocked path fragment '{fragment}' in: {filepath}")

    # Check content
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for pattern in MIMIC_PATTERNS:
                    if pattern.search(line):
                        violations.append(
                            f"{filepath}:{line_num}: matches PHI pattern '{pattern.pattern}'"
                        )
    except (OSError, IsADirectoryError):
        pass  # Non-readable files are not a compliance issue

    return violations


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_no_mimic_data.py <filepath>")
        sys.exit(0)

    filepath = sys.argv[1]
    violations = check_file(filepath)

    if violations:
        print("HIPAA COMPLIANCE VIOLATION — blocking write:")
        for v in violations:
            print(f"  ✗ {v}")
        sys.exit(1)

    sys.exit(0)
