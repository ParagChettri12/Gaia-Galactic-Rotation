#!/usr/bin/env python3
"""Point every placeholder link at the real repository.

    python tools/set_repo.py your-github-username your-repo-name
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "CITATION.cff", "docs/index.html", "tools/export_page.py"]

if len(sys.argv) != 3:
    sys.exit(__doc__)
owner, repo = sys.argv[1], sys.argv[2]

for name in FILES:
    path = ROOT / name
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    new = text.replace("OWNER/REPO", f"{owner}/{repo}").replace("OWNER.github.io/REPO", f"{owner}.github.io/{repo}")
    if new != text:
        path.write_text(new, encoding="utf-8")
        print("updated", name)
print(f"Done. Site URL will be https://{owner}.github.io/{repo}/")
