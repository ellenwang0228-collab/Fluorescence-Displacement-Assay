#!/usr/bin/env python3
"""
Shared figure-layout math for grid/individual curve-fit plots.

Single source of truth for figure sizing, margins, and GridSpec spacing —
used by the on-screen canvas classes in fda_launcher.py and the file-export
functions in pipeline_fda.py / pipeline_ki.py. Stdlib-only so it can be
imported from the headless pipeline modules without pulling in Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULTS: dict = dict(
    cell_w=1.9, cell_h=3.0, resid_h=0.70, dots_h=0.80,
    pdf_cols=4,
    hspace=1.50, wspace=2.70, resid_gap=0.06,
    margin_l=1.2, margin_r=1.7, margin_t=0.9, margin_b=0.9,
)


def resolve_cfg(layout_cfg: dict | None) -> dict:
    """Merge a possibly-partial layout_cfg over DEFAULTS."""
    return {**DEFAULTS, **(layout_cfg or {})}


@dataclass
class CellLayout:
    sub_rows: int
    height_ratios: list
    row_h_in: float
    resid_gap: float


@dataclass
class FigureLayout:
    fig_w: float
    fig_h: float
    left: float
    right: float
    top: float
    bottom: float
    hspace: float
    wspace: float

    def outer_kwargs(self) -> dict:
        return dict(hspace=self.hspace, wspace=self.wspace,
                    left=self.left, right=self.right,
                    top=self.top, bottom=self.bottom)

    def single_kwargs(self) -> dict:
        return dict(left=self.left, right=self.right,
                    top=self.top, bottom=self.bottom)


def cell_layout(layout_cfg: dict | None, *, show_residuals: bool = False,
                show_dots: bool = False) -> CellLayout:
    """Height stack for one cell: curve panel, plus optional residual/dots
    panels added additively (inches)."""
    cfg = resolve_cfg(layout_cfg)
    ratios = [cfg["cell_h"]]
    if show_residuals:
        ratios.append(cfg["resid_h"])
    if show_dots:
        ratios.append(cfg["dots_h"])
    return CellLayout(sub_rows=len(ratios), height_ratios=ratios,
                       row_h_in=sum(ratios), resid_gap=cfg["resid_gap"])


def figure_layout(layout_cfg: dict | None, rows: int, cols: int,
                   row_h_in: float, cell_w_in: float | None = None) -> FigureLayout:
    """Figure size + outer-GridSpec kwargs for a rows x cols grid of cells,
    each row_h_in inches tall. rows=cols=1 is the single/individual-view case."""
    cfg = resolve_cfg(layout_cfg)
    cw = cell_w_in if cell_w_in is not None else cfg["cell_w"]
    hs_in, ws_in = cfg["hspace"], cfg["wspace"]
    ml, mr, mt, mb = cfg["margin_l"], cfg["margin_r"], cfg["margin_t"], cfg["margin_b"]

    grid_w = cols * cw + max(cols - 1, 0) * ws_in
    grid_h = rows * row_h_in + max(rows - 1, 0) * hs_in
    fig_w, fig_h = grid_w + ml + mr, grid_h + mt + mb

    return FigureLayout(
        fig_w=fig_w, fig_h=fig_h,
        hspace=(hs_in / row_h_in) if row_h_in > 0 else 0.0,
        wspace=(ws_in / cw) if cw > 0 else 0.0,
        left=ml / fig_w, right=1 - mr / fig_w,
        top=1 - mt / fig_h, bottom=mb / fig_h,
    )
