# =============================================================================
# What Can Be Known When a Mining Injury Is Reported? Temporal Availability and
# Severity Prediction in 24 Years of MSHA Records (Applied Sciences, 2026)
# Author: Abdulgani Kahraman, Balikesir University
#
# Step 8. Draws Figures 2-5 of the paper from the results of the earlier steps:
#   results/step06_summary.json        JSON block printed at the end of step 6
#                                       (between ###JSON_BEGIN### and ###JSON_END###)
#   results/table9_field_attribution.csv  Table 9 values printed by step 4
# Figure 1 is a schematic of the reporting lifecycle and is not drawn from data.
# Usage: python code/08_make_figures.py   (writes PDF and PNG files to figures/)
# =============================================================================
import csv, json, os
from decimal import Decimal, ROUND_HALF_DOWN
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
J = json.load(open(os.path.join(RES, "step06_summary.json"), encoding="utf-8"))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})
C = {"N": "#1f4e79", "S100": "#5b9bd5", "S1000": "#2e8b57", "B": "#d97b29", "L": "#8b1a1a"}
LEVELS = ["L0", "L1", "L2", "L3"]

def r3(x):
    """Three decimals as reported in the paper (half-down for exact .xxx5 ties)."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_DOWN))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("written:", name)

# ── Figure 2: row-normalised confusion matrices, 2006-2023 ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
for ax, key, title in zip(axes, ["cm_N", "cm_S"], ["(a) Notification set", "(b) Completed-report set"]):
    cm = np.array(J[key], dtype=float)
    r = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for k in range(4):
            ax.text(k, i, f"{r[i, k]:.3f}", ha="center", va="center",
                    color="white" if r[i, k] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(4), LEVELS); ax.set_yticks(range(4), LEVELS)
    ax.set_xlabel("Predicted level"); ax.set_ylabel("True level"); ax.set_title(title, loc="left")
    ax.spines[:].set_visible(False)
fig.tight_layout(); save(fig, "Figure2_confusion_matrices")

# ── Figure 3: precision-recall curves and decision curve, 2018-2023 ─────────
grid = np.array(J["grid"])
fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), gridspec_kw={"height_ratios": [3, 1.35]})
(a, b), (la, lb) = axs
la.axis("off"); lb.axis("off")
pr_items = [("N", "Notification (structured only)", "-"), ("S100", "Completed report, 100 terms", "--"),
            ("S1000", "Completed report, 1000 terms", "-."), ("B", "Assessment", ":")]
for key, lab, ls in pr_items:
    d = J["pr"][key]
    a.plot(grid, d["p"], ls, color=C[key], lw=2, label=f"{lab} (AP {r3(J['ap'][key][0])})")
    a.fill_between(grid, d["lo"], d["hi"], color=C[key], alpha=0.18, lw=0)
prev = J["cal"]["N"]["raw"]["prev"]
a.axhline(prev, color="0.3", ls="--", lw=1.2, label=f"Alert on every record (prevalence {prev:.3f})")
a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.grid(alpha=0.3)
a.set_xlabel("Severe-class recall"); a.set_ylabel("Severe-class precision"); a.set_title("(a) Precision–recall", loc="left")
la.legend(*a.get_legend_handles_labels(), loc="upper center", frameon=False)
a.xaxis.labelpad = 2

pt = np.array(J["dca"]["pt"]); x = np.arange(len(pt))
for key, lab, ls in pr_items[:3]:
    d = J["dca"][key]
    b.plot(x, d["nb"], ls, marker="o", ms=4, color=C[key], lw=2, label=lab)
    b.fill_between(x, d["lo"], d["hi"], color=C[key], alpha=0.18, lw=0)
b.plot(x, J["dca"]["all"], "--", color="0.3", lw=1.2, label="Alert on every record")
b.axhline(0, color="0.2", lw=0.9, label="Alert on no record")
b.set_ylim(-0.003, 0.0175); b.grid(alpha=0.3)
show = [0, 2, 4, 6, 8, 10, 12]
b.set_xticks(x[show], [f"{pt[i]:g}\n({round(1/pt[i]-1):d}:1)" for i in show])
b.set_xlabel("Threshold probability\n(cost ratio, missed case : unnecessary alert)")
b.set_ylabel("Net benefit"); b.set_title("(b) Decision curve", loc="left")
lb.legend(*b.get_legend_handles_labels(), loc="upper center", frameon=False)
fig.tight_layout(); save(fig, "Figure3_pr_and_decision_curve")

# ── Figure 4: annual mean macro F1 by feature set, 2006-2023 ────────────────
years = list(range(2006, 2024))
fig, ax = plt.subplots(figsize=(7.5, 4))
for key, lab, mk in [("N", "Notification", "o"), ("S", "Completed report", "s"),
                     ("B", "Assessment", "^"), ("L", "Outcome-informed", "D")]:
    col = C["S100"] if key == "S" else C[key]
    ax.plot(years, J["annual"][key], marker=mk, ms=4, lw=1.8, color=col, label=lab)
ax.set_ylim(0.3, 1.02); ax.set_xticks(years[::2]); ax.grid(alpha=0.3)
ax.set_xlabel("Evaluation year"); ax.set_ylabel("Macro F1")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout(); save(fig, "Figure4_annual_macro_f1")

# ── Figure 5: field-by-field attribution, 2018-2023 ─────────────────────────
rows = list(csv.DictReader(open(os.path.join(RES, "table9_field_attribution.csv"), encoding="utf-8")))
base = next(r for r in rows if r["condition"] == "Completed-report baseline")
keep = [r for r in rows if r["condition"].startswith("+") and "narrative" not in r["condition"]]
keep.sort(key=lambda r: float(r["mean_macro_f1"]))
labels = [r["condition"].replace("+ all post-closure and severity-triggered fields",
                                 "+ all post-closure and\nseverity-triggered fields") for r in keep]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)
for ax, col, title, xl in [(axes[0], "mean_macro_f1", "(a) Macro F1", "Mean macro F1, 2018–2023"),
                           (axes[1], "mean_severe_recall", "(b) Severe-class recall", "Mean severe-class recall, 2018–2023")]:
    v = [float(r[col]) for r in keep]
    colors = [C["L"] if (col == "mean_severe_recall" and x >= 0.999) else C["S100"] for x in v]
    ax.barh(range(len(v)), v, color=colors, height=0.65)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i, f"{x:.3f}", va="center", fontsize=9)
    b0 = float(base[col]); ax.axvline(b0, color="0.3", ls="--", lw=1.2)
    ax.text(b0 + 0.005, len(v) - 0.45, f"baseline {b0:.3f}", fontsize=9, color="0.3")
    ax.set_xlim(0, 1.12); ax.grid(axis="x", alpha=0.3); ax.set_title(title, loc="left"); ax.set_xlabel(xl)
axes[0].set_yticks(range(len(labels)), labels)
fig.tight_layout(); save(fig, "Figure5_field_attribution")
