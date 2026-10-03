#!/usr/bin/env python3
"""Re-render Figure 2 (theory_validation.png) at 600 dpi from the committed CSVs in figures/,
without re-running any simulation, and copy it to ../manuscript/figures/Fig2_theory.png when that folder exists."""
import os, sys, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.figure
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments")); sys.path.insert(0, HERE)
_orig = matplotlib.figure.Figure.savefig
def _hires(self, fname, *a, **k):
    k["dpi"] = 600
    return _orig(self, fname, *a, **k)
matplotlib.figure.Figure.savefig = _hires
import theory_validation as tv
tv.make_figures()
src = os.path.join(HERE, "figures", "theory_validation.png")
dst = os.path.join(os.path.dirname(HERE), "manuscript", "figures", "Fig2_theory.png")
if os.path.isdir(os.path.dirname(dst)):
    shutil.copy(src, dst)
print("copied", src)
