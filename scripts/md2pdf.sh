#!/usr/bin/env bash
# Render a markdown file to PDF, the way the sprint report was produced.
#
#     scripts/md2pdf.sh docs/apart-ai-incident-response-sprint-2026-report.md
#     scripts/md2pdf.sh input.md output.pdf
#
# pandoc → HTML with inline CSS, then headless Chrome prints it. Not
# `pandoc -o x.pdf` directly: that route goes through LaTeX, which drops
# emoji (the ✅/❌ comparison table) and styles tables unlike GitHub.
set -euo pipefail

IN=${1:?usage: md2pdf.sh INPUT.md [OUTPUT.pdf]}
OUT=${2:-${IN%.md}.pdf}
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML=$(mktemp -t md2pdf).html

pandoc "$IN" -o "$HTML" --standalone --metadata title=" "

python3 - "$HTML" <<'EOF'
import sys
path = sys.argv[1]
s = open(path).read()
s = s.replace("</head>", """<style>
body{max-width:52rem;margin:2rem auto;font-family:Georgia,serif;line-height:1.5;color:#111}
h1{font-size:1.7rem} h2{font-size:1.3rem;margin-top:2rem} h3{font-size:1.1rem} h4{font-size:1rem}
table{border-collapse:collapse;font-size:0.85rem;margin:1rem 0} th,td{border:1px solid #999;padding:4px 8px}
pre{background:#f5f5f5;padding:10px;font-size:0.78rem;overflow-x:auto;border:1px solid #ddd}
code{font-size:0.85em} a{color:#0645ad;text-decoration:none}
@media print{ pre{white-space:pre-wrap} h2{page-break-after:avoid} table{page-break-inside:avoid} }
</style></head>""")
open(path, "w").write(s)
EOF

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$OUT" "$HTML" 2>/dev/null
rm -f "$HTML"
echo "wrote $OUT"
