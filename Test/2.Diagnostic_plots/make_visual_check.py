#!/usr/bin/env python3
"""make_visual_check.py -- side-by-side visual gallery: bayesplot (R PNGs from
make_visual_check.R) vs BayesForge (interactive plotly), same fixture arrays.

Usage:
    cd /home/sosa/work/BF-ppc
    Rscript Test/2.Diagnostic_plots/make_visual_check.R .   # first: the R PNGs
    python3 Test/2.Diagnostic_plots/make_visual_check.py .  # then: the gallery

Writes visual_check/index.html: one row per plot, R left / BF right, with a
note on the known algorithmic differences (KDE bandwidth, draw sampling,
error_hist semantics). Self-contained: the R PNGs are base64-inlined and
plotly.js is embedded, so it opens offline.
"""
import base64
import json
import os
import sys
import warnings

import numpy as np
base = os.path.abspath(sys.argv[1] if len(sys.argv) >= 1 and os.path.isdir(sys.argv[1]) else os.getcwd())
# import BayesForge from the repo, not from site-packages (sys.path[0] is the
# script's directory when run as `python3 Test/.../make_visual_check.py`)
if base not in sys.path:
    sys.path.insert(0, base)

import plotly.offline as pyo

from BayesForge.Diagnostic import ppc

assert os.path.abspath(ppc.__file__) == os.path.join(base, "BayesForge", "Diagnostic", "ppc.py"), \
    "wrong BayesForge: " + ppc.__file__

here = os.path.join(base, "Test", "2.Diagnostic_plots")
fx = os.path.join(here, "fixtures")
out = os.path.join(here, "visual_check")
os.makedirs(out, exist_ok=True)

with open(os.path.join(fx, "input.json")) as f:
    INP = json.load(f)
Y = np.array(INP["continuous"]["y"], dtype=float)
YREP = np.array(INP["continuous"]["yrep"], dtype=float)
LL = np.array(INP["continuous"]["log_lik"], dtype=float)
X = np.array(INP["x"], dtype=float)
CY = np.array(INP["count"]["y"])
CYREP = np.array(INP["count"]["yrep"])

BY = (Y > np.median(Y)).astype(int)
BP = 1.0 / (1.0 + np.exp(-YREP))
ST = np.resize(np.array([1, 1, 0], dtype=float), len(Y))
TY = np.exp(Y / 2)
TYREP = np.exp(YREP[:3] / 2)

NOTE_KDE = ("Approximate pair: bayesplot KDE uses nrd0 on all 200 draws, BF uses "
            "Scott's rule on an n-draw sample (seed 0) -- shapes should agree, "
            "curves will not overlap exactly.")
NOTE_HIST = ("Approximate pair: both draw n=8 reps with their own RNG, so the "
             "sampled draws (hence the panels) differ.")
NOTE_ERR = ("KNOWN DIVERGENCE (xfail in test_ppc_vs_bayesplot.py): bayesplot "
            "ppc_error_hist facets per-draw errors y - yrep[s,i]; BF plots ONE "
            "histogram of the 30 mean residuals y - E[yrep].")
NOTE_SCAT = ("KNOWN DIVERGENCE: bayesplot ppc_scatter plots y vs the AVERAGE of "
             "yrep with inner/outer interval lines; BF plots 9 replicate scatter "
             "panels (y vs yrep[s]). Same intent, different picture.")

# (key, title, BF figure, note)
def _figs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # e.g. ppc_loo_pit reff warning
        return [
            ("intervals", "ppc_intervals (prob=0.5, prob_outer=0.9)",
             ppc.ppc_intervals(Y, YREP, x=X, prob=0.5, prob_outer=0.9), ""),
            ("ribbon", "ppc_ribbon (prob=0.5, prob_outer=0.9)",
             ppc.ppc_ribbon(Y, YREP, x=X, prob=0.5, prob_outer=0.9), ""),
            ("intervals_sorted", "ppc_intervals_sorted (prob=prob_outer=0.9)",
             ppc.ppc_intervals_sorted(Y, YREP, prob=0.9),
             "R side: ppc_intervals on rows sorted by y (no bayesplot counterpart)."),
            ("stat_mean", "ppc_stat (stat=mean)",
             ppc.ppc_stat(Y, YREP, stat="mean"), ""),
            ("stat_sd", "ppc_stat (stat=sd)",
             ppc.ppc_stat(Y, YREP, stat="sd"), ""),
            ("error_scatter", "ppc_error_scatter / ppc_error_scatter_avg",
             ppc.ppc_error_scatter(Y, YREP), ""),
            ("error_hist", "ppc_error_hist",
             ppc.ppc_error_hist(Y, YREP), NOTE_ERR),
            ("density", "ppc_density (n=50)",
             ppc.ppc_density(Y, YREP, n=50), NOTE_KDE),
            ("hist", "ppc_hist (n=8, bins=30)",
             ppc.ppc_hist(Y, YREP, bins=30), NOTE_HIST),
            ("boxplot", "ppc_boxplot (n=20)",
             ppc.ppc_boxplot(Y, YREP), NOTE_HIST),
            ("scatter", "ppc_scatter (9 reps)",
             ppc.ppc_scatter(Y, YREP), NOTE_SCAT),
            ("loo_pit", "ppc_loo_pit / ppc_loo_pit_overlay (PSIS weights)",
             ppc.ppc_loo_pit(Y, YREP, log_likelihood=LL),
             "R side: ppc_loo_pit_overlay (ppc_loo_pit is deprecated). PSIS "
             "implementations differ (loo::psis vs BF _psis_weights): PIT "
             "values agree to ~1.2e-8."),
            ("bars", "ppc_bars (count)",
             ppc.ppc_bars(CY, CYREP), ""),
            ("rootogram", "ppc_rootogram (hanging)",
             ppc.ppc_rootogram(CY, CYREP), ""),
            ("error_binned", "ppc_error_binned (binary, 3 reps)",
             ppc.ppc_error_binned(BY, BP[:3]), ""),
            ("km_overlay", "ppc_km_overlay (3 reps)",
             ppc.ppc_km_overlay(TY, TYREP, ST, n=3),
             "R side: survival::survfit steps drawn by hand (ppc_km_overlay "
             "needs ggfortify, not installed); cut = 1.2 x max observed time."),
        ]


def img_tag(key):
    png = os.path.join(out, f"r_{key}.png")
    if not os.path.exists(png):
        return ('<div class="missing">r_%s.png manquant &mdash; lancer '
                '<code>Rscript Test/2.Diagnostic_plots/make_visual_check.R .</code>'
                '</div>' % key)
    with open(png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return '<img src="data:image/png;base64,%s" alt="r_%s">' % (b64, key)


rows = []
for key, title, fig, note in _figs():
    fig_div = fig.to_html(full_html=False, include_plotlyjs=False,
                          default_width=700, default_height=480)
    rows.append(f"""
<div class="pair">
  <h2>{title}</h2>
  {f'<p class="note">{note}</p>' if note else ''}
  <div class="cols">
    <div class="col"><h3>bayesplot (R 1.16.0)</h3>{img_tag(key)}</div>
    <div class="col"><h3>BayesForge (plotly)</h3>{fig_div}</div>
  </div>
</div>""")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PPC visual check: bayesplot vs BayesForge</title>
<script>{pyo.get_plotlyjs()}</script>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-bottom: .2rem; }}
 h3 {{ font-size: .85rem; color: #555; margin: .3rem 0; font-weight: 600; }}
 .note {{ color: #8a6d00; background: #fff8dc; border-left: 3px solid #d4a72c;
          padding: .35rem .6rem; font-size: .85rem; margin: .2rem 0; }}
 .cols {{ display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }}
 .col {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: .5rem; }}
 .col img {{ max-width: 700px; width: 100%; height: auto; }}
 .missing {{ color: #b00; font-size: .85rem; padding: 1rem; }}
 .pair {{ margin-bottom: 2.2rem; }}
</style></head><body>
<h1>PPC visual check &mdash; bayesplot (left) vs BayesForge (right)</h1>
<p>Same fixture arrays (fixtures/input.json). Numeric parity is asserted in
test_ppc_vs_bayesplot.py; this page is for eyeballing the rendering.</p>
{''.join(rows)}
</body></html>"""

path = os.path.join(out, "index.html")
with open(path, "w") as f:
    f.write(html)
print("wrote", path)
