Audit the repo for potential HIPAA data leaks before any commit.

1. Search all staged files (`git diff --cached --name-only`) for patterns matching MIMIC note text signatures:
   - Strings matching `(?i)(admission date|discharge date|date of birth|attending:|service:|chief complaint:)`
   - Any filename containing `mimic`, `noteevents`, or `discharge` under a data directory
2. Verify `/data/mimic/` is present in `.gitignore` — fail if missing
3. Scan all staged `.py` files for hardcoded violations:
   - Absolute paths containing `/data/mimic/` or `C:\data\`
   - Any string that looks like an API key or token: `(?i)(api_key|secret|token|password)\s*=\s*["'][^"']{8,}["']`
   - Any hardcoded path that should come from `.env` (look for `data/mimic` not wrapped in `os.getenv` or `os.environ`)
4. If any violations are found:
   - Print each violation with filename and line number
   - Exit with a non-zero status to block the commit
   - Do NOT attempt to auto-fix — report only
5. If all checks pass, print "Compliance check passed — safe to commit" and exit 0
