#!/usr/bin/env python3
"""Build the GitHub Pages site in docs/ from a completed run of the notebook.

    python tools/export_page.py runs/2026-09-17_Rotation.ipynb
    python tools/export_page.py runs/my_run.ipynb --summary results_summary.json

Writes docs/notebook.html (the full rendered run), docs/index.html (the landing page)
and docs/assets/rotation-curve.png (the fitted rotation curve, if it is in the run).
Requires: pip install nbformat nbconvert
"""
import argparse, base64, json, re, subprocess, sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OWNER_REPO = "OWNER/REPO"      # rewritten by tools/set_repo.py


def markdown_outputs(nb):
    for cell in nb.cells:
        for out in cell.get("outputs") or []:
            text = out.get("data", {}).get("text/markdown")
            if text:
                yield cell, text


def find_table(nb, needle):
    for _, text in markdown_outputs(nb):
        if needle in text:
            rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
            if rows:
                return "\n".join(rows)
    return None


GREEK = {"rho": "ρ", "sigma": "σ", "Sigma": "Σ", "mu": "μ", "phi": "φ", "chi": "χ",
         "Theta": "Θ", "theta": "θ", "varpi": "ϖ", "infty": "∞", "odot": "☉", "times": "×"}


def detex(s):
    """Turn the light LaTeX used in the notebook's tables into readable HTML."""
    def one(m):
        t = m.group(1)
        for name, ch in GREEK.items():
            t = t.replace("\\" + name, ch)
        t = re.sub(r"_\{(.+?)\}", r"<sub>\1</sub>", t)
        t = re.sub(r"_(\w)", r"<sub>\1</sub>", t)
        t = re.sub(r"\^\{(.+?)\}", r"<sup>\1</sup>", t)
        return t.replace("\\", "")
    s = re.sub(r"\$(.+?)\$", one, s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"(\d(?:\.\d+)?)e\+?(\d+)", r"\1 × 10<sup>\2</sup>", s)
    return s


def md_table_to_html(md):
    if not md:
        return "<p>Not available in this run.</p>"
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in md.splitlines()]
    header, body = cells[0], [r for r in cells[2:] if r]
    fmt = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", detex(s))
    html = ["<table><thead><tr>" + "".join(f"<th>{fmt(h)}</th>" for h in header) + "</tr></thead><tbody>"]
    for row in body:
        html.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in row) + "</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def save_hero(nb):
    for cell in nb.cells:
        if "Fitted Complex Model" in cell.source:
            for out in cell.get("outputs") or []:
                png = out.get("data", {}).get("image/png")
                if png:
                    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
                    (DOCS / "assets" / "rotation-curve.png").write_bytes(base64.b64decode(png))
                    return True
    return False


def headline(nb, summary):
    counts, speed = "", ""
    for _, text in markdown_outputs(nb):
        m = re.search(r"\*\*([\d,]+)\*\* training stars and \*\*([\d,]+)\*\* independent test stars", text)
        if m:
            counts = f"{int(m.group(1).replace(',', '')):,} stars"
        m = re.search(r"circular speed of \*\*([\d.]+) km/s\*\*", text)
        if m:
            speed = m.group(1)
    if summary and not counts:
        c = summary.get("counts", {})
        counts = f"{c.get('train_after_cuts', 0):,} stars"
    return counts, speed


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Milky Way's Rotation Curve from Gaia</title>
<meta name="description" content="A reproducible notebook that queries the Gaia Archive, measures the Milky Way's rotation curve, and tests the fitted models on an independent sample of stars.">
<style>
 :root {{ --bg:#ffffff; --fg:#1b1f24; --muted:#59636e; --line:#d8dee4; --accent:#1f6feb; --card:#f6f8fa; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#30363d; --accent:#4493f8; --card:#161b22; }} }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:860px; margin:0 auto; padding:48px 20px 80px; }}
 h1 {{ font-size:2.1rem; line-height:1.2; margin:0 0 .3em; }}
 h2 {{ font-size:1.3rem; margin:2.2em 0 .6em; padding-bottom:.3em; border-bottom:1px solid var(--line); }}
 .authors {{ color:var(--muted); margin:0 0 1.4em; }}
 .lede {{ font-size:1.1rem; }}
 .btns {{ display:flex; flex-wrap:wrap; gap:10px; margin:1.6em 0; }}
 .btn {{ display:inline-block; padding:10px 16px; border-radius:8px; border:1px solid var(--line); color:var(--fg); text-decoration:none; background:var(--card); font-weight:600; font-size:.95rem; }}
 .btn.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
 .btn:hover {{ filter:brightness(1.07); }}
 img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }}
 table {{ border-collapse:collapse; width:100%; font-size:.94rem; margin:.6em 0 1.2em; display:block; overflow-x:auto; }}
 th,td {{ border:1px solid var(--line); padding:7px 10px; text-align:right; white-space:nowrap; }}
 th:first-child, td:first-child {{ text-align:left; }}
 thead th {{ background:var(--card); }}
 ul {{ padding-left:1.2em; }}
 code {{ background:var(--card); padding:2px 5px; border-radius:5px; font-size:.9em; }}
 pre {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; overflow-x:auto; }}
 footer {{ margin-top:3em; padding-top:1.2em; border-top:1px solid var(--line); color:var(--muted); font-size:.87rem; }}
 a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">

<h1>The Milky Way's Rotation Curve, measured from Gaia</h1>
<p class="authors">Parag Chettri and Santiago Berumen &middot; run of {run_date}</p>

<p class="lede">A notebook that queries the ESA Gaia Archive on its own, turns {counts} into a Galactocentric rotation curve, fits four models to it, and then tests those models on stars it has never seen. Everything on this page was produced by the code in the repository, with no pre-packaged data files.</p>

<div class="btns">
  <a class="btn primary" href="notebook.html">Read the full notebook</a>
  <a class="btn" href="https://github.com/{owner_repo}/blob/main/Rotation.ipynb">Download the notebook</a>
  <a class="btn" href="https://colab.research.google.com/github/{owner_repo}/blob/main/Rotation.ipynb">Run it in Colab</a>
  <a class="btn" href="https://github.com/{owner_repo}">Source on GitHub</a>
</div>

<img src="assets/rotation-curve.png" alt="The fitted Milky Way rotation curve, broken into bulge, stellar disk, HI and H2 layers, and dark halo">

<h2>How the models compare</h2>
<p>Every model is scored against the same binned curve with the same covariance matrix, so the numbers are directly comparable, and lower is better throughout. The last column is the gap in Bayesian Information Criterion to the best model, where a gap above 10 counts as very strong evidence.</p>
{model_table}

<h2>Testing on stars the models never saw</h2>
<p>Gaia's data releases are not independent surveys, so a "DR2 test set" usually contains the same stars as a DR3 training set, re-measured. Here the split is done by star, and the audit has to come back empty for the test to mean anything.</p>
{overlap_table}
<p>The models below were fitted to the DR3 stars only, then scored on the DR2 stars without any refitting.</p>
{test_table}

<h2>What it says about the Galaxy</h2>
{derived_table}
<p>{conclusion}</p>

<h2>Reproducing it</h2>
<pre>git clone https://github.com/{owner_repo}.git
cd {repo_name}
pip install -r requirements.txt
jupyter lab Rotation.ipynb</pre>
<p>The first run submits two asynchronous jobs to the Gaia Archive and takes a few minutes to half an hour. Results are cached locally, so later runs are immediate.</p>

<footer>
<p>This work has made use of data from the European Space Agency mission <em>Gaia</em>, processed by the Gaia Data Processing and Analysis Consortium (DPAC). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.</p>
<p>Code and text released under the MIT License.</p>
</footer>

</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a completed .ipynb, saved with its outputs")
    ap.add_argument("--summary", help="results_summary.json from the same run (optional)")
    args = ap.parse_args()

    run = Path(args.run)
    nb = nbformat.read(run, 4)
    if not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
        sys.exit(f"{run} has no outputs. Run the notebook and save it before exporting the page.")

    summary = json.loads(Path(args.summary).read_text()) if args.summary else None
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").touch()

    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--output-dir", str(DOCS),
                    "--output", "notebook.html", str(run)], check=True)

    if not save_hero(nb):
        print("note: no 'Fitted Complex Model' figure found; keeping the existing hero image")

    m = re.search(r"(\d{4}-\d{2}-\d{2})", run.name)
    run_date = (summary or {}).get("run_at", "")[:10] or (m.group(1) if m else "")
    counts, _ = headline(nb, summary)
    conclusion = ""
    for _, text in markdown_outputs(nb):
        if "preferred by the Bayesian Information Criterion" in text:
            conclusion = re.sub(r"[*$\\]", "", text.split("\n\n")[0])

    owner_repo = OWNER_REPO
    (DOCS / "index.html").write_text(TEMPLATE.format(
        run_date=run_date or "the latest run",
        counts=counts or "tens of thousands of stars",
        owner_repo=owner_repo,
        repo_name=owner_repo.split("/")[-1],
        model_table=md_table_to_html(find_table(nb, "ΔBIC")),
        overlap_table=md_table_to_html(find_table(nb, "Independent test stars kept")),
        test_table=md_table_to_html(find_table(nb, "χ² per bin")),
        derived_table=md_table_to_html(find_table(nb, "Derived quantity")),
        conclusion=conclusion or "See the notebook for the full discussion.",
    ), encoding="utf-8")
    print(f"Wrote {DOCS/'index.html'} and {DOCS/'notebook.html'}")


if __name__ == "__main__":
    main()
