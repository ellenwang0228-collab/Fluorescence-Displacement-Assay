#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

HARTREE_TO_KCAL = 627.5095

# ─── matplotlib setup (non-interactive / no X display needed on cluster) ─────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  XYZ multi-frame parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_crest_conformers(path: Path):
    """
    Parse a CREST crest_conformers.xyz file.
    Returns list of energies in Hartree (from the comment line of each frame).
    Comment lines that don't start with a float are skipped gracefully.
    """
    energies = []
    with open(path, errors="ignore") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        try:
            e = float(comment.split()[0])
            energies.append(e)
        except (ValueError, IndexError):
            pass   # comment not an energy — skip but still advance
        i += 2 + n_atoms

    return energies


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(root: Path):
    """
    Scan root for crest_conformers.xyz files.
    Returns list of (label, energies_kcal_relative) sorted by label.
    """
    pairs = []
    for conf_xyz in sorted(root.rglob("crest_conformers.xyz")):
        d = conf_xyz.parent
        label = d.name          # e.g. "MyHost_Apixaban"
        energies_Ha = parse_crest_conformers(conf_xyz)
        if not energies_Ha:
            print(f"  [SKIP]  {label} — no parseable energies in {conf_xyz}")
            continue
        e_arr = np.array(energies_Ha)
        e_rel = (e_arr - e_arr.min()) * HARTREE_TO_KCAL   # kcal/mol above minimum
        pairs.append((label, e_rel))

    return sorted(pairs, key=lambda x: x[0])


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def pair_stats(label, e_rel):
    return {
        "pair":             label,
        "n_conformers":     len(e_rel),
        "e_span_kcal":      round(float(e_rel.max()), 3),
        "e_mean_kcal":      round(float(e_rel.mean()), 3),
        "e_median_kcal":    round(float(np.median(e_rel)), 3),
        "e_p25_kcal":       round(float(np.percentile(e_rel, 25)), 3),
        "e_p75_kcal":       round(float(np.percentile(e_rel, 75)), 3),
        "n_within_1kcal":   int(np.sum(e_rel <= 1.0)),
        "n_within_3kcal":   int(np.sum(e_rel <= 3.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Colour pairs by energy span — teal (narrow/rigid) → orange (wide/flexible)
def _span_colour(span_kcal, vmin=0, vmax=6):
    t = min(max((span_kcal - vmin) / max(vmax - vmin, 1e-9), 0), 1)
    # interpolate: teal (0.15, 0.56, 0.56) → orange (0.90, 0.55, 0.15)
    r = 0.15 + t * (0.90 - 0.15)
    g = 0.56 + t * (0.55 - 0.56)
    b = 0.56 + t * (0.15 - 0.56)
    return (r, g, b)


def make_figure(pairs, stats, out_png: Path, title: str):
    n = len(pairs)
    if n == 0:
        print("No pairs to plot.")
        return

    # Dynamic figure height: 0.55 in per pair, min 6 in
    fig_h = max(6, 0.55 * n + 3)
    fig = plt.figure(figsize=(16, fig_h), facecolor="white")

    # Three columns: violin | n_conformers bar | span bar
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1], wspace=0.05,
                           left=0.28, right=0.97, top=0.93, bottom=0.06)
    ax_v  = fig.add_subplot(gs[0])   # violin / strip
    ax_n  = fig.add_subplot(gs[1])   # n_conformers
    ax_sp = fig.add_subplot(gs[2])   # energy span

    labels      = [p[0] for p in pairs]
    energies    = [p[1] for p in pairs]
    spans       = [s["e_span_kcal"] for s in stats]
    n_conf      = [s["n_conformers"] for s in stats]
    colours     = [_span_colour(sp, 0, max(spans) if spans else 1) for sp in spans]

    max_span = max(spans) if spans else 6
    positions = list(range(n))

    # ── violin plot ───────────────────────────────────────────────────────────
    for i, (e_rel, col) in enumerate(zip(energies, colours)):
        if len(e_rel) >= 4:
            vp = ax_v.violinplot(e_rel, positions=[i], vert=False,
                                 showmedians=True, showextrema=False,
                                 widths=0.7)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.65)
                body.set_edgecolor("none")
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(1.5)
        # Jitter-strip all points
        jitter = np.random.default_rng(i).uniform(-0.25, 0.25, len(e_rel))
        ax_v.scatter(e_rel, np.full_like(e_rel, i) + jitter,
                     s=6, color=col, alpha=0.55, zorder=3, linewidths=0)
        # Mark the minimum (always 0) with a diamond
        ax_v.scatter([0], [i], marker="D", s=30, color=col,
                     edgecolors="white", linewidths=0.6, zorder=5)

    ax_v.set_yticks(positions)
    ax_v.set_yticklabels(labels, fontsize=8.5)
    ax_v.set_xlabel("ΔE above minimum conformer  (kcal mol⁻¹)", fontsize=9)
    ax_v.set_xlim(-0.3, max(max_span * 1.08, 0.5))
    ax_v.axvline(1.0, color="#BBBBBB", lw=0.7, ls="--", zorder=0)
    ax_v.axvline(3.0, color="#DDBBBB", lw=0.7, ls="--", zorder=0)
    ax_v.text(1.0, n - 0.2, "1 kcal", fontsize=6, color="#AAAAAA", ha="center")
    ax_v.text(3.0, n - 0.2, "3 kcal", fontsize=6, color="#CCAAAA", ha="center")
    ax_v.set_ylim(-0.7, n - 0.3)
    ax_v.invert_yaxis()
    ax_v.spines[["top", "right"]].set_visible(False)
    ax_v.grid(axis="x", color="#EEEEEE", lw=0.5)
    ax_v.set_title("Conformer energy distribution", fontsize=10, pad=6)

    # ── n_conformers bar ──────────────────────────────────────────────────────
    ax_n.barh(positions, n_conf, color=colours, alpha=0.75, height=0.55)
    ax_n.set_yticks(positions)
    ax_n.set_yticklabels([])
    ax_n.set_xlabel("Conformers", fontsize=9)
    ax_n.set_ylim(-0.7, n - 0.3)
    ax_n.invert_yaxis()
    ax_n.spines[["top", "right", "left"]].set_visible(False)
    ax_n.tick_params(left=False)
    ax_n.grid(axis="x", color="#EEEEEE", lw=0.5)
    ax_n.set_title("N", fontsize=10, pad=6)
    # Annotate counts
    for i, nc in enumerate(n_conf):
        ax_n.text(nc + max(n_conf) * 0.02, i, str(nc),
                  va="center", fontsize=7, color="#555555")
    ax_n.set_xlim(0, max(n_conf) * 1.25 if n_conf else 1)

    # ── energy span bar ───────────────────────────────────────────────────────
    ax_sp.barh(positions, spans, color=colours, alpha=0.75, height=0.55)
    ax_sp.set_yticks(positions)
    ax_sp.set_yticklabels([])
    ax_sp.set_xlabel("Span (kcal)", fontsize=9)
    ax_sp.set_ylim(-0.7, n - 0.3)
    ax_sp.invert_yaxis()
    ax_sp.spines[["top", "right", "left"]].set_visible(False)
    ax_sp.tick_params(left=False)
    ax_sp.grid(axis="x", color="#EEEEEE", lw=0.5)
    ax_sp.set_title("Span", fontsize=10, pad=6)
    for i, sp in enumerate(spans):
        ax_sp.text(sp + max(spans) * 0.02 if spans else 0.05, i,
                   f"{sp:.1f}", va="center", fontsize=7, color="#555555")
    ax_sp.set_xlim(0, max(spans) * 1.25 if spans else 1)

    # ── colour legend ─────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_span_colour(0,   0, max_span), label="Narrow span (rigid)"),
        mpatches.Patch(color=_span_colour(max_span * 0.5, 0, max_span), label="Medium span"),
        mpatches.Patch(color=_span_colour(max_span, 0, max_span), label="Wide span (flexible)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#888",
                   markersize=6, label="Best conformer (ΔE = 0)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.62, 0.0))

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)

    # Footnote about comparability
    fig.text(0.28, 0.002,
             "ΔE values are relative within each pair — absolute GFN-FF energies are NOT "
             "comparable across pairs (different compositions).",
             fontsize=6.5, color="#888888", style="italic")

    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(stats, out_csv: Path):
    cols = ["pair", "n_conformers", "e_span_kcal", "e_mean_kcal",
            "e_median_kcal", "e_p25_kcal", "e_p75_kcal",
            "n_within_1kcal", "n_within_3kcal"]
    with open(out_csv, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in stats:
            fh.write(",".join(str(s[c]) for c in cols) + "\n")
    print(f"  Saved: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise CREST conformer energy distributions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir", default=".",
                    help="crest_results/ directory to scan (default: CWD)")
    ap.add_argument("--out", default="crest_energy_summary.png",
                    help="Output PNG path (default: crest_energy_summary.png)")
    ap.add_argument("--title", default="CREST GFN-FF conformer energy distributions",
                    help="Figure title")
    ap.add_argument("--sort", choices=["name", "span", "nconf"], default="name",
                    help="Sort pairs by: name (default), span (energy span), "
                         "nconf (number of conformers)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    out_png = Path(args.out).resolve()
    out_csv = out_png.with_suffix(".csv")

    print(f"crest_energy_plot.py")
    print(f"  Scanning : {root}")

    pairs = discover_pairs(root)
    if not pairs:
        sys.exit(f"No crest_conformers.xyz files found under {root}")

    print(f"  Found    : {len(pairs)} pair(s)")

    stats = [pair_stats(label, e_rel) for label, e_rel in pairs]

    # Sort
    if args.sort == "span":
        order = sorted(range(len(stats)), key=lambda i: stats[i]["e_span_kcal"])
        pairs = [pairs[i] for i in order]
        stats = [stats[i] for i in order]
        args.title += " (sorted by span)"
    elif args.sort == "nconf":
        order = sorted(range(len(stats)), key=lambda i: stats[i]["n_conformers"],
                       reverse=True)
        pairs = [pairs[i] for i in order]
        stats = [stats[i] for i in order]
        args.title += " (sorted by n_conformers)"

    # Print summary table to terminal
    print(f"\n  {'Pair':<45} {'N':>5} {'Span':>7} {'Within 1kcal':>13} {'Within 3kcal':>13}")
    print(f"  {'-'*45} {'-'*5} {'-'*7} {'-'*13} {'-'*13}")
    for s in stats:
        print(f"  {s['pair']:<45} {s['n_conformers']:>5} "
              f"{s['e_span_kcal']:>6.2f} "
              f"{s['n_within_1kcal']:>8} ({100*s['n_within_1kcal']/max(s['n_conformers'],1):4.0f}%)"
              f"{s['n_within_3kcal']:>8} ({100*s['n_within_3kcal']/max(s['n_conformers'],1):4.0f}%)")
    print()

    make_figure(pairs, stats, out_png, args.title)
    write_csv(stats, out_csv)

    print(f"\nDone.")
    print(f"  Tip: sort by span to identify the most vs least flexible binders:")
    print(f"       python3 crest_energy_plot.py --sort span")


if __name__ == "__main__":
    np.random.seed(42)   # reproducible jitter
    main()
