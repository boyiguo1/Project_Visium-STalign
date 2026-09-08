#!/usr/bin/env python3
# =============================================================================
#  stalign_pipeline.py -- Xenium to Visium registration, one donor per run
#
#  Batch form of the exploratory notebook. Given a donor name it resolves
#  the Xenium files and the matching Visium capture area, exports the
#  Visium data from the SPE object if that has not been done, fetches the
#  SpaceRanger spatial outputs if missing, registers the two sections,
#  evaluates the result and writes everything to <outdir>/<donor>_STalign_result.
#
#  EVERYTHING WORKS IN MICRONS. Xenium centroids are microns as stored;
#  Visium pixel coordinates are converted on load, so the initial affine
#  carries rotation and translation only and its determinant is one.
#
#  The pixel-to-micron factor is recovered per capture area from the
#  100 um spot pitch, a chip constant, not from spot_diameter_fullres,
#  which Space Ranger documents as a visualisation estimate spanning
#  60-70 um. That field is still read, for the in-spot radius.
#
#  Xenium input comes either from the raw files (parquet + h5, with the
#  QC below) or from a prepared AnnData, which skips loading and QC
#  entirely. The prepared path is what to use for an imputed object.
#
#  USAGE
#    python stalign_pipeline.py --sample Br8667
#    python stalign_pipeline.py --all
#    python stalign_pipeline.py --sample Br8667 --no-figures --niter 500
#    python stalign_pipeline.py --sample Br8667 \
#        --xenium-h5ad ~/imputed/{sample}_imputed.h5ad
#    python stalign_pipeline.py --list
#
#  OUTPUTS per donor, in <outdir>/<donor>_STalign_result/
#    <donor>_aligned_coordinates.csv     cell id, native and aligned x/y
#    <donor>_aligned_coordinates.README  units, frames and restrictions
#    <donor>_visium_coordinates.csv      spot barcode, microns, domain
#    <donor>_visium_coordinates.csv      spot barcode, microns, domain
#    <donor>_metrics.json                every number this run produced
#    <donor>_run.log                     parameters, log and figure captions
#    <donor>_report.html                 the same with the figures embedded
#    fig_*.png / .pdf                    diagnostics and figures
#  and, across donors, <outdir>/cohort_summary.csv
# =============================================================================
import os, sys, re, io, json, glob, gzip, time, argparse, subprocess, traceback
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.io import mmread
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import spearmanr, mannwhitneyu, rankdata, norm
import scanpy as sc, anndata as ad
import torch
from STalign import STalign

# ------------------------------------------------------------- constants ----
SPOT_PITCH   = 100.0    # um, Visium spot centre-to-centre (10x spec)
COL_DX_UM    = SPOT_PITCH / 2
ROW_DY_UM    = SPOT_PITCH * np.sqrt(3) / 2
DX_RASTER    = 50.0     # um, rasterization step, both platforms
BLUR         = 2.0      # in units of dx
GRID_UM      = 200.0
LDDMM_A_UM   = 300.0    # um, smoothing kernel ~ one cortical layer
MATCH_THRESH = 0.85     # Clifton 2023
MIRROR_MIN_GAP = 0.05   # below this the two mirror candidates are a tie

# Gaussian-mixture variances, on images normalised to the unit interval.
# These are the STalign package defaults. Earlier versions of this pipeline
# inherited sigmaM = sigmaB = 0.3 from an exploratory notebook, which was
# never revisited: at that width the matching term treats a difference of a
# third of the dynamic range as decisive, and the mixture collapsed to a
# hard decision. Measured across three donors, under 1% of cells scored
# between 0.2 and 0.8, and one donor returned 1.000 for every cell. The
# 0.85 threshold from Clifton et al. then selected nothing, since almost
# every value was already 0 or 1.
#
# The package defaults give a ratio sigmaA/sigmaM of 5, against 1 in the
# atlas alignment reported by Clifton et al. A higher ratio makes the model
# readier to assign unmatched source tissue to the artifact component
# rather than force it onto the target, which suits partially matched
# sections; these pairs overlap by 62 to 97% of area.
LDDMM_SIGMA_M = 1.0     # STalign default
LDDMM_SIGMA_B = 2.0     # STalign default
LDDMM_SIGMA_A = 5.0     # STalign default
KERNEL_SIGMA = 150.0    # um, scorer kernel width
KERNEL_MINSUP = 8
DTYPE        = torch.float64

SPD_ORDER  = ["SpD07-L1","SpD06-L2/3","SpD02-L3/4","SpD05-L5",
              "SpD03-L6","SpD04-WM","SpD01-WMtz"]
LAY        = ["L1/M","L2/3","L3/4","L5","L6","WM","WMtz"]
MARK_UPPER = ["CUX2","LAMP5","RORB"]
MARK_WM    = ["MBP","PLP1","MOBP","MOG","CLDN11"]
WM_MARKERS = ["MBP","MOBP","PLP1","MAG","CLDN11","CNP","MOG",
              "BCAS1","TF","CNDP1","TMEM144"]
DOM_COLORS = ["#E69F00","#56B4E9","#009E73","#F0E442",
              "#0072B2","#7A0177","#CC79A7"]
CMAP_DOM   = ListedColormap(DOM_COLORS)
NORM_DOM   = BoundaryNorm(np.arange(-0.5, 7.5, 1), CMAP_DOM.N)

U_EDGE = np.array([-1.0, -1.0]); U_EDGE /= np.linalg.norm(U_EDGE)
V_TAN  = np.array([U_EDGE[1], -U_EDGE[0]])

DOMAIN2LAYER = {"spd07":"L1/M","spd06":"L2/3","spd02":"L3/4","spd05":"L5",
                "spd03":"L6","spd04":"WM","spd01":"WMtz"}
CLUSTER2CT   = {2:"L2/3 Ex",11:"L4/5 Ex",18:"L5 Ex",14:"L6 Ex",9:"In: SST,PVALB",
                12:"In: VIP,LAMP5",6:"Oligo",8:"Oligo",1:"Oligo",10:"Ambig/Oligo",
                7:"Mic",17:"Ast",16:"Ast",4:"Ast",5:"Endo",13:"Endo",3:"Endo",
                15:"Ambig/In/Endo"}
LIEBER_BASE = ("https://raw.githubusercontent.com/LieberInstitute/"
               "spatialDLPFC_SCZ_XENIUM/devel/processed-data")
SR_RAW = ("https://raw.githubusercontent.com/LieberInstitute/"
          "spatialDLPFC_SCZ/main/processed-data/spaceranger")

# One style for every figure, so panels are comparable across the report.
# Titles carry the panel letter; captions live in the log and the HTML
# rather than inside the image, which keeps the figures usable as-is in a
# manuscript where the caption is typeset separately.
plt.rcParams.update({
    # Raster at 600 dpi and keep a vector copy: journals ask for 300 dpi at
    # final width for halftones and vector for line art, and a scatter of
    # 70,000 cells is neither cleanly.
    "figure.dpi": 110, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed as TrueType, not curves,
                                             # so text stays editable
    "axes.grid": False, "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlepad": 8,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.4,
    "figure.titlesize": 12, "figure.titleweight": "bold",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "#141413", "axes.linewidth": 0.8,
    "lines.linewidth": 1.4, "lines.markersize": 4,
})
PANEL_H = 5.4          # inches, height of a single-row figure
CAPTIONS = {}          # filled in below, consumed by the log and the report


# ============================================================================
#  figure captions
#
#  Held here rather than drawn into the images. A figure with the caption
#  baked in cannot be dropped into a manuscript, and a caption that lives
#  only in a chat message cannot be found again six months later.
# ============================================================================
CAPTIONS = {
 "fig_initial_placement":
   "Initial placement of the source section. (a) Rotation swept over the full "
   "circle at coarse resolution, each candidate scored by a Gaussian-kernel "
   "correlation between the laminar coordinate predicted from nearby source "
   "cells and the target spot annotation. (b) Refinement at fine resolution; "
   "the shaded band marks angles scoring within 0.005 of the maximum, which is "
   "the width the reported angle carries. (c) The resulting placement, with the "
   "matched pial boundaries of both sections in black.",
 "fig_registration":
   "Registration result. (a) Per-cell matching probability from the mixture "
   "model. (b) Aligned Xenium cells above the matching threshold, coloured by "
   "spatial domain, over Visium spots coloured by their own domain. "
   "(c) Jacobian determinant of the fitted transform over occupied source "
   "tissue; unity denotes preserved area and values at or below zero would "
   "indicate folding.",
 "fig_panelD_domains":
   "Spatial domain annotation of both sections. (D1) Aligned Xenium cells. "
   "(D2) Visium spots. Both panels share axis limits and one colour scale, so "
   "a domain appearing at the same figure position lies at the same tissue "
   "position.",
 "fig_panelD_overlap":
   "The overlapping region. (a) Cells classified by two criteria: proximity to "
   "a spot, and the matching probability from the mixture model. Their "
   "disagreement bounds how far the probabilistic estimate of overlap departs "
   "from the geometric one. (b) Xenium and (c) Visium restricted to cells and "
   "spots satisfying both.",
 "fig_layer_concordance":
   "Laminar concordance between aligned cells and the spots they land on. "
   "(a) Row-normalised confusion matrix; dashed lines mark the grey-white "
   "boundary. (b) Mean landed domain per source domain with standard "
   "deviation, against the identity. Departure from the identity gives the "
   "systematic offset. These annotations drive the registration, so the "
   "agreement bounds alignment quality from above rather than testing it.",
 "fig_layer_landing":
   "Where each source domain lands. Filled points are aligned Xenium cells of "
   "the domain named; open rings are Visium spots of the same domain; pale "
   "grey is all spots. Cells on the rings indicate correspondence, cells "
   "beside them a laminar offset, and cells scattered across the section that "
   "the domain label is not spatially coherent.",
 "fig_expression_concordance":
   "Shared-gene expression concordance, which enters the registration at no "
   "point. (a) Cosine similarity between Xenium pseudospot and Visium spot "
   "profiles, split by spatial patterning; dashed lines mark the values "
   "reported for MERFISH-to-Visium in mouse brain. (b) Spearman correlation "
   "against the nonzero fraction on the sparser platform; the statistic "
   "collapses toward zero where either side is mostly ties at zero.",
 "fig_crosstech":
   "Cross-technology expression correspondence. (a) Normalised expression in "
   "the Visium target, (b) in the aligned Xenium source, and (c) matched "
   "pseudospot against spot, for three spatially patterned genes spanning the "
   "range of cosine similarity and one non-patterned gene as a negative "
   "control. (d, e) Cosine distributions for the two strata; the solid line is "
   "the median of this section and the dashed line the published reference.",
}


# ============================================================================
#  run report
#
#  Everything printed during a run is also captured, so the log and the
#  HTML are records of what actually happened rather than a summary written
#  afterwards. The HTML embeds the figures and their captions in execution
#  order, giving a reviewable artifact without opening a notebook.
# ============================================================================
class Report:
    def __init__(self, sample, outdir):
        self.sample, self.outdir = sample, outdir
        self.sections, self.params, self.metrics = [], {}, {}
        self.started = time.time()
        self.step("Run")

    def step(self, title, code=None):
        """
        Open a cell. `code` is the call this stage makes, shown in the HTML
        so the report reads as an executed notebook. It is the call rather
        than the function body: the body lives in this file and would go
        stale in the report the moment either changed, whereas the call
        together with the parameter table is what a reader needs.
        """
        self.sections.append(dict(title=title, code=code, lines=[], figs=[]))

    def line(self, text):
        self.sections[-1]["lines"].append(text)

    def fig(self, name):
        self.sections[-1]["figs"].append(name)

    def param(self, **kw):  self.params.update(kw)
    def metric(self, **kw): self.metrics.update(kw)

    # ---- outputs ----
    def write(self):
        self._write_log()
        self._write_html()

    def _kv(self, d, width=28):
        return "\n".join(f"  {k:<{width}} {v}" for k, v in d.items())

    def _write_log(self):
        p = os.path.join(self.outdir, f"{self.sample}_run.log")
        w = [f"STalign registration -- {self.sample}",
             f"finished {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}, "
             f"{time.time()-self.started:.0f} s", "",
             "=" * 72, "PARAMETERS", "=" * 72, self._kv(self.params), "",
             "=" * 72, "METRICS", "=" * 72, self._kv(self.metrics), ""]
        for i, sec in enumerate(self.sections):
            if not sec["lines"] and not sec["figs"] and not sec.get("code"):
                continue
            w += ["=" * 72, f"[{i}] {sec['title'].upper()}", "=" * 72]
            if sec.get("code"):
                w += [""] + ["    " + l for l in sec["code"].split("\n")] + [""]
            w += sec["lines"]
            for f in sec["figs"]:
                w += ["", f"[{f}.png]", "  " + CAPTIONS.get(f, "").replace(
                    ". ", ".\n  ")]
            w += [""]
        open(p, "w").write("\n".join(w) + "\n")
        return p

    def _write_html(self):
        import html as _h
        p = os.path.join(self.outdir, f"{self.sample}_report.html")
        css = """
        :root{--fg:#141413;--mut:#6b6b68;--line:#e6e4dd;--bg:#faf9f5;
               --in:#0072B2}
        *{box-sizing:border-box}
        body{font:14px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
             sans-serif;max-width:1180px;margin:0 auto;padding:36px 28px 80px;
             color:var(--fg);background:#fff}
        h1{font-size:23px;margin:0 0 4px;font-weight:650}
        .sub{color:var(--mut);margin:0 0 30px;font-size:13px}
        h2{font-size:15px;margin:0 0 12px;font-weight:650}
        .cell{margin:0 0 30px;border-left:3px solid transparent;padding-left:14px}
        .cell:hover{border-left-color:var(--line)}
        .prompt{color:var(--in);font:12px/1 ui-monospace,SFMono-Regular,Menlo,
                monospace;margin:0 0 6px;letter-spacing:.02em}
        pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;
            padding:12px 14px;overflow-x:auto;margin:0 0 10px;
            font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
            white-space:pre-wrap;word-break:break-word}
        pre.out{background:#fff;border-color:#f0eee7;color:#333}
        .kw{color:#7A0177}.cm{color:#6b6b68;font-style:italic}
        table{border-collapse:collapse;font-size:13px;margin:6px 0 20px;width:100%}
        td{padding:3px 18px 3px 0;vertical-align:top;border-bottom:1px solid #f4f2ec}
        td:first-child{color:var(--mut);white-space:nowrap;width:46%}
        figure{margin:16px 0 8px}
        img{max-width:100%;border:1px solid var(--line);border-radius:4px;
            display:block}
        figcaption{font-size:12.5px;color:#4a4a47;margin-top:9px;line-height:1.55}
        figcaption b{color:var(--fg)}
        .cols{display:grid;grid-template-columns:1fr 1fr;gap:0 44px}
        @media(max-width:820px){.cols{grid-template-columns:1fr}}
        """

        def esc(x): return _h.escape(str(x))

        def hl(code):
            """Light syntax colouring: keywords and comments only. Anything
            heavier would need a real lexer for no gain at this size."""
            out = []
            for line in code.split("\n"):
                if line.lstrip().startswith("#"):
                    out.append(f"<span class='cm'>{esc(line)}</span>"); continue
                e = esc(line)
                for kw in ("import ", "for ", "if ", "else", "return ", "def "):
                    e = e.replace(kw, f"<span class='kw'>{kw}</span>")
                out.append(e)
            return "\n".join(out)

        def tbl(d):
            return "<table>" + "".join(
                f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
                for k, v in d.items()) + "</table>"

        def two_col(d):
            it = list(d.items()); half = (len(it) + 1) // 2
            return ("<div class='cols'>" + tbl(dict(it[:half]))
                    + tbl(dict(it[half:])) + "</div>")

        h = ["<!doctype html><meta charset='utf-8'>",
             f"<title>{esc(self.sample)} &middot; STalign</title>",
             f"<style>{css}</style>",
             f"<h1>{esc(self.sample)}</h1>",
             f"<p class='sub'>Xenium to Visium registration &middot; "
             f"{pd.Timestamp.now():%Y-%m-%d %H:%M} &middot; "
             f"{time.time()-self.started:.0f} s</p>",
             "<div class='cell'><h2>Parameters</h2>", two_col(self.params), "</div>",
             "<div class='cell'><h2>Metrics</h2>", two_col(self.metrics), "</div>"]

        n = 0
        for sec in self.sections:
            if not sec["lines"] and not sec["figs"] and not sec.get("code"):
                continue
            n += 1
            h.append("<div class='cell'>")
            h.append(f"<h2>{esc(sec['title'])}</h2>")
            if sec.get("code"):
                h.append(f"<div class='prompt'>In [{n}]:</div>")
                h.append(f"<pre>{hl(sec['code'])}</pre>")
            if sec["lines"]:
                if sec.get("code"):
                    h.append("<div class='prompt'>Out:</div>")
                h.append("<pre class='out'>" + esc("\n".join(sec["lines"])) + "</pre>")
            for f in sec["figs"]:
                h.append(f"<figure><img src='{f}.png' alt='{f}'>"
                         f"<figcaption><b>{f}</b>. "
                         f"{esc(CAPTIONS.get(f, ''))}</figcaption></figure>")
            h.append("</div>")
        open(p, "w").write("\n".join(h))
        return p


_REPORT = None      # set per sample; log() tees into it when present


# ============================================================================
#  small shared helpers
# ============================================================================
def log(msg, indent=0):
    line = ("  " * indent) + msg
    print(line, flush=True)
    if _REPORT is not None: _REPORT.line(line)


def add_scalebar(ax, length_um=1000.0, label="1 mm", color="#141413", lw=3.0,
                 fontsize=9, x_frac=0.06, y_frac=0.08, clear_axes=True,
                 frame=True, box=True, box_pad=0.025):
    """
    Scale bar for tissue panels. Axis units must be millimetres and the
    aspect must already be equal, or the bar is valid along one direction
    only. Position is in axes fractions, so an inverted y-axis cannot put
    it off-screen; only the LENGTH is converted to data units. Call last,
    after set_aspect, since that adjusts xlim.
    """
    if clear_axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(frame)
        if frame: sp.set_color("#141413"); sp.set_linewidth(0.8)
    x0, x1 = sorted(ax.get_xlim())
    L_frac = (length_um / 1000.0) / (x1 - x0)
    if box:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (x_frac - box_pad, y_frac - 0.055 - box_pad),
            L_frac + 2*box_pad, 0.067 + 2*box_pad,
            boxstyle="round,pad=0.004", transform=ax.transAxes,
            facecolor="white", edgecolor="none", alpha=0.85,
            zorder=9, clip_on=False))
    ax.plot([x_frac, x_frac + L_frac], [y_frac, y_frac], transform=ax.transAxes,
            color=color, lw=lw, solid_capstyle="butt", zorder=11, clip_on=False)
    ax.text(x_frac + L_frac/2, y_frac - 0.030, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=fontsize, color=color, zorder=11)


def domain_legend(fig, y=0.02, ncol=7, ms=7):
    """
    One domain legend per figure, placed below the panels. Repeating it in
    every panel wastes space, and a colour bar is wrong here because the
    domains are categorical.
    """
    handles = [Line2D([], [], marker="o", ls="", ms=ms, color=DOM_COLORS[i],
                      label=l) for i, l in enumerate(LAY)]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handletextpad=0.4,
               columnspacing=1.6, fontsize=8.5)


def cbar(im, ax, label, fraction=0.046, pad=0.02):
    """Colour bar with the units named, since an unlabelled one is a ramp."""
    cb = plt.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)
    return cb


def panel(ax, letter, text, fontsize=10):
    """Panel label and title in one place, so every figure reads the same."""
    ax.set_title(f"{letter}   {text}", loc="left", fontweight="bold",
                 fontsize=fontsize)


def savefig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    if _REPORT is not None: _REPORT.fig(name)


def rot2(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def affine3(M, t):
    """STalign acts on (y, x, 1), so the 2x2 is transposed into that order."""
    A = np.eye(3)
    A[0,0], A[0,1], A[0,2] = M[1,1], M[1,0], t[1]
    A[1,0], A[1,1], A[1,2] = M[0,1], M[0,0], t[0]
    return A


def edge_mask(pts, u=U_EDGE, q=0.03):
    return (pts @ u) >= np.quantile(pts @ u, 1 - q)


def read_parquet_any(path):
    """.parquet.gz is parquet inside gzip; pyarrow cannot read the outer gzip."""
    with open(path, "rb") as f:
        is_gz = f.read(2) == b"\x1f\x8b"
    if is_gz:
        with gzip.open(path, "rb") as f:
            return pd.read_parquet(io.BytesIO(f.read()))
    return pd.read_parquet(path)


def to_dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def cpm(M):
    lib = M.sum(1, keepdims=True)
    return M / np.where(lib == 0, 1, lib) * 1e6


# ============================================================================
#  inputs: manifest, Visium export, SpaceRanger spatial files
# ============================================================================
def build_manifest(args):
    """Donor -> capture area, read from the SPE object once and cached."""
    mf = os.path.join(args.workdir, "spe_manifest.csv")
    if os.path.exists(mf) and not args.refresh_manifest:
        return pd.read_csv(mf)
    log(f"building manifest from {args.rds}")
    r = subprocess.run(["Rscript", args.rscript, "--rds", args.rds,
                        "--manifest", mf], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"manifest failed:\n{r.stdout}\n{r.stderr}")
    log(r.stdout.strip(), 1)
    return pd.read_csv(mf)


def resolve_sample(sample, manifest, args):
    """Locate the Xenium files and the Visium capture area for one donor."""
    hits = (glob.glob(f"{args.xenium_dir}/GSM*_{sample}-cells.parquet") +
            glob.glob(f"{args.xenium_dir}/GSM*_{sample}-cells.parquet.gz"))
    if not hits:
        raise FileNotFoundError(f"no Xenium cells file for {sample} in {args.xenium_dir}")
    stem = hits[0].replace("-cells.parquet.gz", "").replace("-cells.parquet", "")
    h5 = f"{stem}-cell_feature_matrix.h5"
    if not os.path.exists(h5):
        raise FileNotFoundError(f"missing {h5}")
    # A truncated download reads as a valid path but fails deep inside the
    # loader, so both files are checked before use. The size floor is set low,
    # at 100 kB: cells.parquet holds only ids, centroids and areas, so a
    # complete file for a small section compresses to under a megabyte. This
    # check is meant to catch the empty file a 404 leaves behind and a
    # transfer cut off near its start; the magic-byte test below and the
    # row-count agreement inside load_xenium are what actually discriminate.
    for f_ in (hits[0], h5):
        if os.path.getsize(f_) < 1e5:
            raise ValueError(f"{os.path.basename(f_)} is only "
                             f"{os.path.getsize(f_)/1e3:.0f} kB, too small to "
                             f"be a complete file; re-download it")
    with open(h5, "rb") as fh:
        if fh.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"{os.path.basename(h5)} is not a valid HDF5 file; "
                             f"re-download it")

    rows = manifest[manifest.donor.astype(str).str.contains(sample, na=False)]
    if len(rows) == 0:
        raise ValueError(f"{sample} not found among the Visium donors; "
                         f"available: {sorted(manifest.donor.unique())[:8]}")
    if len(rows) > 1:
        log(f"WARNING {sample} maps to {len(rows)} capture areas "
            f"({list(rows.capture_area)}); taking the one with most spots", 1)
        rows = rows.sort_values("spots", ascending=False)
    row = rows.iloc[0]
    return dict(sample=sample, cells_path=hits[0], xstem=stem,
                capture=row.capture_area, donor=row.donor,
                dx=row.get("dx"), sex=row.get("sex"), age=row.get("age"),
                n_spots_expected=int(row.spots))


def ensure_visium_export(info, args):
    """
    Flat files from the SPE, written once per donor and collected under a
    single directory rather than scattered through the working directory,
    since a full cohort produces two dozen of them.
    """
    vdir = os.path.join(args.export_root, f"{info['sample']}_export")
    need = [f"{info['sample']}_{f}" for f in
            ("counts.mtx", "barcodes.tsv", "genes.tsv",
             "metadata.csv", "coordinates.csv", "gene_annotation.csv")]
    if all(os.path.exists(os.path.join(vdir, f)) for f in need) and not args.re_export:
        return vdir
    log(f"exporting Visium {info['capture']} from the SPE object")
    cmd = ["Rscript", args.rscript, "--rds", args.rds, "--export",
           "--capture", info["capture"], "--outdir", vdir, "--tag", info["sample"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"export failed:\n{r.stdout}\n{r.stderr}")
    log(r.stdout.strip(), 1)
    return vdir


def ensure_spaceranger(capture, args):
    """
    scalefactors_json.json for this capture area. Only spot_diameter_fullres
    is used downstream, for the in-spot radius; the scale itself comes from
    the spot pitch.
    """
    srdir = os.path.join(args.workdir, "spaceranger", capture, "outs", "spatial")
    want = ["scalefactors_json.json", "tissue_positions.csv"]
    os.makedirs(srdir, exist_ok=True)
    for f in want:
        dest = os.path.join(srdir, f)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{SR_RAW}/{capture}/outs/spatial/{f}"
        log(f"fetching {f} for {capture}", 1)
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=120) as r_:
                data = r_.read()
            open(dest, "wb").write(data)
        except Exception as e:
            if f == "scalefactors_json.json":
                raise RuntimeError(f"cannot fetch {url}: {e}")
            log(f"optional file unavailable ({e})", 2)
    return srdir

# ============================================================================
#  loading
# ============================================================================
def load_xenium(info, args):
    """
    Xenium cells, from either the raw files or a prepared AnnData.

    Raw path: parquet centroids joined to the feature-matrix h5, with two
    QC steps. Cells with zero counts are dropped because they carry no
    signal to rasterize; MAD outliers are FLAGGED but never removed,
    since deleting cells punches holes in the density field the
    registration relies on. Both are switchable.

    Prepared path (--xenium-h5ad): the object is taken as given, with no
    filtering at all. This is the route for an imputed matrix, or for any
    object that has already been processed elsewhere. It must carry
    obsm['spatial'] in microns; cell_id_xe and orig_row are recovered
    from obs if present and reconstructed otherwise, since orig_row is
    what links to the Lieber annotation tables.
    """
    if args.xenium_h5ad:
        path = args.xenium_h5ad.format(sample=info["sample"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"prepared Xenium object not found: {path}")
        A = ad.read_h5ad(path)
        log(f"Xenium from prepared object: {os.path.basename(path)}", 1)

        if "spatial" not in A.obsm:
            raise KeyError("prepared object has no obsm['spatial']; "
                           "centroids in microns are required")
        sp = np.asarray(A.obsm["spatial"], float)
        if np.ptp(sp[:, 0]) < 500:
            log(f"WARNING spatial extent is {np.ptp(sp[:,0]):.0f} units wide; "
                f"Xenium centroids should be microns, so this looks like "
                f"pixels or millimetres", 1)

        if "cell_id_xe" not in A.obs:
            src = next((c for c in ("cell_id", "cell_ID", "barcode")
                        if c in A.obs), None)
            A.obs["cell_id_xe"] = (A.obs[src].astype(str).values if src
                                   else A.obs_names.astype(str))
        if "orig_row" not in A.obs:
            # orig_row indexes cells.parquet and is what the Lieber keys use;
            # recover it by matching cell_id rather than assuming row order,
            # which an upstream filter may have changed
            cells = read_parquet_any(info["cells_path"]).reset_index(drop=True)
            pos = pd.Series(np.arange(len(cells)) + 1,
                            index=cells["cell_id"].astype(str))
            A.obs["orig_row"] = A.obs["cell_id_xe"].astype(str).map(pos).values
            n_lost = int(pd.isna(A.obs["orig_row"]).sum())
            if n_lost:
                log(f"WARNING {n_lost} cells have no match in cells.parquet and "
                    f"will lack Lieber labels", 1)
        for c in ("total_counts", "detected"):
            if c not in A.obs: A.obs[c] = np.nan
        if "is_outlier" not in A.obs: A.obs["is_outlier"] = False

        n_panel = int(A.n_vars)
        if n_panel > 1000:
            log(f"NOTE the object carries {n_panel} genes, well beyond a Xenium "
                f"panel. If these were imputed from Visium, the shared-gene "
                f"evaluation is no longer orthogonal to the registration and "
                f"should be read as a consistency check, not a validation", 1)
        log(f"Xenium {A.n_obs} cells x {A.n_vars} genes, no filtering applied", 1)
        return A, 0

    # ---------------- raw path ----------------
    cells = read_parquet_any(info["cells_path"]).reset_index(drop=True)
    cells["orig_row"] = np.arange(len(cells)) + 1
    A = sc.read_10x_h5(f"{info['xstem']}-cell_feature_matrix.h5")
    A.var_names_make_unique()
    A = A[cells["cell_id"].astype(str).values].copy()
    A.obsm["spatial"]   = cells[["x_centroid", "y_centroid"]].to_numpy(float)
    A.obs["orig_row"]   = cells["orig_row"].values
    A.obs["cell_id_xe"] = cells["cell_id"].astype(str).values
    for c in ("cell_area", "nucleus_area"):
        if c in cells: A.obs[c] = cells[c].to_numpy()

    is_ctrl = A.var_names.str.match(
        r"^(NegControlProbe|NegControlCodeword|Unassigned|BLANK|Deprecated|Intergenic)",
        case=False)
    X = to_dense(A.X)
    n_empty = int((X.sum(1) == 0).sum())
    if not args.keep_empty and n_empty:
        A = A[X.sum(1) > 0].copy(); X = X[X.sum(1) > 0]

    A.obs["total_counts"] = X.sum(1)
    A.obs["detected"]     = (X > 0).sum(1)
    if args.no_qc:
        A.obs["is_outlier"] = False
    else:
        def mad_out(v, mode="both", n=3):
            v = np.asarray(v, float); m = np.median(v)
            s_ = np.median(np.abs(v - m)) * 1.4826
            return {"lower": v < m-n*s_, "both": (v < m-n*s_) | (v > m+n*s_)}[mode]
        A.obs["is_outlier"] = mad_out(X.sum(1)) | mad_out((X > 0).sum(1), "lower")

    A = A[:, ~is_ctrl].copy()
    log(f"Xenium {A.n_obs} cells x {A.n_vars} genes"
        + (f", dropped {n_empty} empty" if not args.keep_empty and n_empty
           else (f", kept {n_empty} empty" if n_empty else ""))
        + (", outlier flagging off" if args.no_qc
           else f", flagged {A.obs.is_outlier.mean()*100:.1f}%"), 1)
    return A, n_empty


def load_visium(info, vdir, srdir):
    """
    Visium matrix, coordinates and physical scale.

    The pixel-to-micron factor is recovered from the 100 um spot pitch and
    checked against the array geometry, which is independent of any
    registration. spot_diameter_fullres supplies the in-spot radius only.
    """
    s = info["sample"]
    rl = lambda p: [l.strip().strip('"') for l in open(p) if l.strip()]
    bc, gn = rl(f"{vdir}/{s}_barcodes.tsv"), rl(f"{vdir}/{s}_genes.tsv")
    orient = lambda M: (M.T if M.shape == (len(gn), len(bc)) else M).tocsr()

    V = ad.AnnData(X=orient(mmread(f"{vdir}/{s}_counts.mtx").tocsr()))
    if os.path.exists(f"{vdir}/{s}_logcounts.mtx"):
        V.layers["logcounts"] = orient(mmread(f"{vdir}/{s}_logcounts.mtx").tocsr())
    V.obs_names, V.var_names = bc, gn
    V.obs = V.obs.join(pd.read_csv(f"{vdir}/{s}_metadata.csv").set_index("barcode"))
    coord = pd.read_csv(f"{vdir}/{s}_coordinates.csv").set_index("barcode")
    xp, yp = coord.loc[bc, ["x", "y"]].to_numpy(float).T

    gann = (pd.read_csv(f"{vdir}/{s}_gene_annotation.csv")
              .drop_duplicates("gene_id").set_index("gene_id"))
    mapped = V.var_names.map(gann["gene_name"])
    V.var["gene_id"] = V.var_names.values
    V.var_names = np.where(mapped.notna(), mapped, V.var_names)
    V.var_names_make_unique()

    nn = cKDTree(np.c_[xp, yp]).query(np.c_[xp, yp], k=2)[0][:, 1]
    pitch_px  = float(np.median(nn))
    um_per_px = SPOT_PITCH / pitch_px
    tight     = float(np.mean(np.abs(nn - pitch_px) < 0.05 * pitch_px))

    xJ, yJ = xp * um_per_px, yp * um_per_px
    V.obsm["spatial"]    = np.c_[xJ, yJ]
    V.obsm["spatial_px"] = np.c_[xp, yp]

    sf = json.load(open(f"{srdir}/scalefactors_json.json"))
    spot_d_px = float(sf["spot_diameter_fullres"])
    spot_r_um = spot_d_px / 2 * um_per_px
    disc_cov  = np.pi * spot_r_um**2 / (SPOT_PITCH**2 * np.sqrt(3)/2)

    # geometry check, independent of the registration
    ok_arr = {"array_row", "array_col"} <= set(V.obs.columns)
    if ok_arr:
        exp_x = np.ptp(V.obs["array_col"].to_numpy(float)) * COL_DX_UM
        exp_y = np.ptp(V.obs["array_row"].to_numpy(float)) * ROW_DY_UM
        dev_x = float(np.ptp(xJ) / exp_x - 1); dev_y = float(np.ptp(yJ) / exp_y - 1)
    else:
        dev_x = dev_y = np.nan

    scale = dict(pitch_px=pitch_px, um_per_px=um_per_px, px_per_um=1/um_per_px,
                 spot_diameter_fullres=spot_d_px, spot_r_um=spot_r_um,
                 disc_coverage=disc_cov, nn_tight_frac=tight,
                 array_dev_x=dev_x, array_dev_y=dev_y,
                 implied_spot_diam_um=float(spot_d_px * um_per_px))
    log(f"Visium {V.n_obs} spots x {V.n_vars} genes | pitch {pitch_px:.2f} px "
        f"-> {um_per_px:.4f} um/px | spot r {spot_r_um:.2f} um", 1)
    log(f"array-geometry check: {dev_x*100:+.2f}% x, {dev_y*100:+.2f}% y", 1)
    return V, scale


def attach_labels(A, sample, cache_dir):
    """
    Lieber spatial-domain and cell-type labels, derived on Xenium alone.
    Cached locally so a batch run hits the network once.
    """
    os.makedirs(cache_dir, exist_ok=True)
    f_spd = os.path.join(cache_dir, "label_transfer_smoothed.csv")
    f_ct  = os.path.join(cache_dir, "banksy_clustering.csv")
    if not os.path.exists(f_spd):
        pd.read_csv(f"{LIEBER_BASE}/04_label_transfer/"
                    "label_transfer_N24_k50_smoothed_labels.csv").to_csv(f_spd, index=False)
    if not os.path.exists(f_ct):
        pd.read_csv(f"{LIEBER_BASE}/06_cell_type_clustering/"
                    "banksy_clustering_lambda0.1_res0.7.csv").to_csv(f_ct, index=False)
    spd = pd.read_csv(f_spd); spd.columns = ["key", "pred"]
    spd = spd.set_index("key")["pred"]
    ct  = pd.read_csv(f_ct).set_index("V2")["V1"]

    keys = pd.Series([f"{sample}_{int(r)}" if pd.notna(r) else None
                      for r in A.obs["orig_row"]], index=A.obs_names)
    A.obs["lieber_key"]   = keys.values
    A.obs["spd"]          = keys.map(spd).values
    A.obs["layer"]        = A.obs["spd"].map(DOMAIN2LAYER)
    A.obs["banksy_clust"] = keys.map(ct).values
    A.obs["cell_type"]    = A.obs["banksy_clust"].map(CLUSTER2CT)

    def marker_score(Ad, genes_):
        g = [x for x in genes_ if x in Ad.var_names]
        M = to_dense(Ad[:, g].X)
        lib = np.asarray(Ad.X.sum(1)).ravel()[:, None].astype(float); lib[lib == 0] = 1
        M = np.log1p(M / lib * 1e4)
        return ((M - M.mean(0)) / (M.std(0) + 1e-9)).mean(1)

    marker_axis = marker_score(A, MARK_WM) - marker_score(A, MARK_UPPER)
    code = A.obs["layer"].map({l: i for i, l in enumerate(LAY)}).astype(float).to_numpy()
    ok = ~np.isnan(code)
    prof = pd.DataFrame({"l": code[ok], "a": marker_axis[ok]}).groupby("l")["a"].mean()
    rho = float(spearmanr(prof.index, prof.values).statistic)
    cov = float(A.obs["layer"].notna().mean())
    log(f"labels: coverage {cov*100:.1f}%, join check rho {rho:.3f}", 1)
    return marker_axis, cov, rho


def build_ordinals(A, V, marker_axis):
    """
    Ordinals 1..7 from pia to white matter on both platforms.

    layer_axis fills the cells that failed Lieber QC by marker rank, which
    is a legitimate rasterization signal but not a layer call; lay_x_true
    keeps only genuine annotations and is what every layer test uses.
    """
    lay_ord = A.obs["layer"].map({l: i+1 for i, l in enumerate(LAY)})
    has = lay_ord.notna().values
    la = np.full(A.n_obs, np.nan)
    la[has] = lay_ord.values[has].astype(float)
    if (~has).any():
        mk = marker_axis[~has]
        la[~has] = 1 + 6*(rankdata(mk)-1)/(len(mk)-1)
    lay_x_true = A.obs["layer"].map({l: i for i, l in enumerate(LAY)}).to_numpy(float)
    lab = V.obs["spd_label"].map({d: i+1 for i, d in enumerate(SPD_ORDER)}).to_numpy(float)
    bad = np.isnan(lab)
    if bad.any():
        log(f"WARNING {bad.sum()} spots have spd_label outside SPD_ORDER", 1)
    return la, lay_x_true, lab, bad


# ============================================================================
#  initialisation: mirror, then edge-butted angle search
# ============================================================================
def make_kernel_scorer(pts, s_vals, ref, lab, sig=KERNEL_SIGMA):
    """
    Continuous placement score: each target spot takes a Gaussian-weighted
    mean of the source ordinal nearby, correlated with its own domain.

    The tree is built once on untransformed source coordinates and the
    spots are mapped back by the inverse transform, exact for orthogonal
    M, so a sweep does not rebuild a 70k-point tree per candidate.
    """
    tree = cKDTree(pts)
    def score(M, t):
        p  = (ref - t) @ M
        nb = tree.query_ball_point(p, r=3*sig, workers=-1)
        sh = np.full(len(ref), np.nan)
        for j, ii in enumerate(nb):
            if len(ii) < KERNEL_MINSUP: continue
            ii = np.asarray(ii)
            w  = np.exp(-((pts[ii] - p[j])**2).sum(1) / (2*sig*sig))
            sh[j] = (w @ s_vals[ii]) / w.sum()
        ok = ~np.isnan(sh)
        if ok.sum() < 200: return np.nan, int(ok.sum())
        return float(spearmanr(sh[ok], lab[ok]).statistic), int(ok.sum())
    return score


def binned_corr(pts, s_src, ref, s_ref, bs=250.0):
    """Coarse binned score, kept for the mirror decision and for reporting."""
    lo = np.minimum(pts.min(0), ref.min(0))
    gid = lambda p: (np.floor((p[:,0]-lo[0])/bs).astype(int)*100000 +
                     np.floor((p[:,1]-lo[1])/bs).astype(int))
    gs, gr = gid(pts), gid(ref)
    common = np.intersect1d(np.unique(gs), np.unique(gr))
    if len(common) < 20: return np.nan, 0
    a = np.array([s_src[gs == c].mean() for c in common])
    b = np.array([s_ref[gr == c].mean() for c in common])
    return float(spearmanr(a, b).statistic), len(common)


def place_score(pts, s_src, ref, s_ref, bs=500.0, w_shape=0.35):
    """
    Placement score combining laminar agreement with outline agreement.

    The laminar term alone is degenerate along the layers. An equally
    spaced ordinal forms a monotonic ramp, and a ramp is unchanged by
    sliding parallel to its isolines, so translation in that direction and
    rotation about the depth axis are both weakly determined. Measured on
    a synthetic ramp, dozens of distinct translations score identically.

    The outline term supplies exactly the missing constraint, because the
    tissue boundary is not invariant to sliding. It enters as a weighted
    term rather than as a hard constraint: an earlier version butted the
    outermost band of each section together, asserting that band to be the
    pial surface, and where a section has a protruding corner that
    assertion is false. Being hard, it could not be overridden by the
    evidence, and the source was driven off the target with no way back.

    Returns the combined score, the outline agreement alone, and the
    number of shared bins.
    """
    lo  = np.minimum(pts.min(0), ref.min(0))
    gid = lambda p: (np.floor((p[:, 0]-lo[0])/bs).astype(np.int64)*1000000 +
                     np.floor((p[:, 1]-lo[1])/bs).astype(np.int64))
    gs, gr = gid(pts), gid(ref)
    us, ur = np.unique(gs), np.unique(gr)
    common = np.intersect1d(us, ur)
    if len(common) < 15:
        return -2.0, 0.0, len(common)
    order_s = np.argsort(gs); order_r = np.argsort(gr)
    gs_s, ss_s = gs[order_s], s_src[order_s]
    gr_s, sr_s = gr[order_r], s_ref[order_r]
    li, ri = np.searchsorted(gs_s, common, "left"), np.searchsorted(gs_s, common, "right")
    a = np.array([ss_s[i:j].mean() for i, j in zip(li, ri)])
    li, ri = np.searchsorted(gr_s, common, "left"), np.searchsorted(gr_s, common, "right")
    b = np.array([sr_s[i:j].mean() for i, j in zip(li, ri)])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return -2.0, 0.0, len(common)
    rho = float(spearmanr(a, b).statistic)
    iou = len(common) / len(np.union1d(us, ur))
    return (1 - w_shape)*rho + w_shape*iou, iou, len(common)


def initial_affine(xI, yI, layer_axis, lay_x_true, ref, lab, args, outdir):
    """
    Rotation and translation searched jointly, in three passes of
    decreasing step and increasing cost.

    Translation is searched over the plausible range rather than fixed by
    a boundary heuristic. Sections whose outline misleads such a heuristic
    -- a protruding corner rather than a pial edge -- were previously
    placed off the target with no way to recover, since the refinement
    window was narrower than the resulting error.

    Reflection is settled first, on the laminar signal alone, and is not
    revisited: the coarse pass would otherwise spend half its budget on a
    hypothesis already rejected.
    """
    src = np.c_[xI, yI]

    # ---- reflection, from the depth axis of each section ----
    def depth_dir(x, y, s):
        P = np.c_[x - x.mean(), y - y.mean()]
        g = np.linalg.lstsq(P, s - s.mean(), rcond=None)[0]
        return g / np.linalg.norm(g)
    dI = depth_dir(xI, yI, layer_axis)
    dJ = depth_dir(ref[:, 0], ref[:, 1], lab)
    th_deg = float(np.degrees(np.arctan2(dJ[1], dJ[0]) - np.arctan2(dI[1], dI[0])))
    nrm = np.array([-dI[1], dI[0]])
    REFL = np.eye(2) - 2*np.outer(nrm, nrm)

    mirror_rho = {}
    for m in (False, True):
        M = rot2(th_deg) @ REFL if m else rot2(th_deg)
        t = ref.mean(0) - (src @ M.T).mean(0)
        mirror_rho[m], _ = binned_corr(src @ M.T + t, layer_axis, ref, lab)
    mirror = max(mirror_rho, key=lambda m: mirror_rho[m])
    mirror_gap = abs(mirror_rho[True] - mirror_rho[False])
    apply_mirror = (lambda M: M @ REFL) if mirror else (lambda M: M)
    log(f"mirror={mirror} (binned rho {mirror_rho[False]:.3f} vs "
        f"{mirror_rho[True]:.3f}, gap {mirror_gap:.3f})", 1)
    # Reflecting about the depth axis leaves the pia-to-white-matter order
    # unchanged and reverses only tangential handedness, so the laminar
    # signal is nearly blind to it. A small gap means the choice rests on
    # almost nothing, and a wrong choice would not show in any laminar
    # metric: the coordinates come out mirrored while still displaying the
    # correct depth sequence. The choice stands, but it is flagged, and
    # shared-gene expression is the evidence that can settle it.
    if mirror_gap < MIRROR_MIN_GAP:
        log(f"CHECK the two mirror candidates differ by only "
            f"{mirror_gap:.3f}; this choice is not supported by the laminar "
            f"signal and should be checked against expression concordance", 1)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(src), size=min(25000, len(src)), replace=False)
    src_s, lay_s = src[sub], layer_axis[sub]

    # ---- pass 1: angle x translation, coarse ----
    t_start = time.time()
    SPAN, STEP = args.trans_span, args.trans_span/6.0
    offs = np.arange(-SPAN, SPAN + STEP/2, STEP)
    cand = []
    for deg in np.arange(0, 360, args.coarse_step):
        M = apply_mirror(rot2(deg))
        P = src_s @ M.T
        t0 = ref.mean(0) - P.mean(0)
        for oy in offs:
            for ox in offs:
                s_, iou_, n_ = place_score(P + t0 + np.array([ox, oy]),
                                           lay_s, ref, lab, bs=500.0)
                if s_ > -1.5:
                    cand.append((deg, t0[0]+ox, t0[1]+oy, s_, iou_))
    C = pd.DataFrame(cand, columns=["deg", "tx", "ty", "score", "iou"])
    if not len(C):
        raise RuntimeError("no candidate placement produced a usable score")
    C = C.sort_values("score", ascending=False)
    log(f"pass 1: {len(cand)} placements over {len(np.arange(0,360,args.coarse_step))} "
        f"angles and +/-{SPAN:.0f} um, best {C.score.iloc[0]:.3f} "
        f"(IoU {C.iou.iloc[0]:.2f}), {time.time()-t_start:.0f} s", 1)

    # ---- pass 2: refine the leading candidates ----
    # Several distinct angles can score within noise of each other, so the
    # top few are carried forward rather than the single maximum.
    t_start = time.time()
    seeds, seen = [], []
    for _, r_ in C.iterrows():
        if any(abs(r_.deg - d_) < 20 or abs(abs(r_.deg - d_) - 360) < 20 for d_ in seen):
            continue
        seeds.append(r_); seen.append(r_.deg)
        if len(seeds) >= 3: break
    ref2 = []
    for sd in seeds:
        for deg in np.arange(sd.deg - 6, sd.deg + 6.01, 1.0):
            M = apply_mirror(rot2(deg)); P = src_s @ M.T
            for oy in np.arange(-300, 301, 100.0):
                for ox in np.arange(-300, 301, 100.0):
                    s_, iou_, n_ = place_score(P + [sd.tx+ox, sd.ty+oy], lay_s,
                                               ref, lab, bs=350.0)
                    ref2.append((deg, sd.tx+ox, sd.ty+oy, s_, iou_))
    R2 = pd.DataFrame(ref2, columns=["deg", "tx", "ty", "score", "iou"]
                      ).sort_values("score", ascending=False)
    b2 = R2.iloc[0]
    log(f"pass 2: {len(ref2)} placements around {len(seeds)} seed angles "
        f"{[round(float(s.deg),1) for s in seeds]}, best {b2.deg:.1f} deg, "
        f"{time.time()-t_start:.0f} s", 1)

    # ---- pass 3: polish on the kernel scorer, full resolution ----
    t_start = time.time()
    score_full = make_kernel_scorer(src, layer_axis, ref, lab)
    fine = []
    for deg in np.arange(b2.deg - 2, b2.deg + 2.01, args.fine_step):
        M = apply_mirror(rot2(deg))
        t = np.array([b2.tx, b2.ty]) + (ref.mean(0) - (src @ M.T).mean(0)) * 0
        r_, n_ = score_full(M, t)
        fine.append((deg, r_))
    F = pd.DataFrame(fine, columns=["deg", "rho"]).dropna()
    k = int(F.rho.values.argmax())
    if 0 < k < len(F) - 1:
        y0, y1, y2 = F.rho.values[k-1:k+2]
        den = y0 - 2*y1 + y2
        deg_opt = float(F.deg.values[k] +
                        (0.5*(y0-y2)/den if den else 0.0)*args.fine_step)
    else:
        deg_opt = float(F.deg.values[k])

    M_opt = apply_mirror(rot2(deg_opt))
    t_fin = np.array([b2.tx, b2.ty])
    best = (-2, t_fin)
    for oy in np.arange(-100, 101, 50.0):
        for ox in np.arange(-100, 101, 50.0):
            r_, _ = score_full(M_opt, t_fin + [ox, oy])
            if np.isfinite(r_) and r_ > best[0]:
                best = (r_, t_fin + np.array([ox, oy]))
    rho_fin, t_fin = float(best[0]), best[1]
    srcH = src @ M_opt.T + t_fin
    rho_opt, n_opt = score_full(M_opt, t_fin)

    half = F[F.rho >= F.rho.max() - 0.005]
    ang_lo, ang_hi = float(half.deg.min()), float(half.deg.max())
    _, iou_fin, _ = place_score(srcH, layer_axis, ref, lab, bs=350.0)
    over = float((cKDTree(ref).query(srcH)[0] <= SPOT_PITCH).mean())
    log(f"pass 3: {deg_opt:.1f} deg (flat over {ang_lo:.1f}-{ang_hi:.1f}), "
        f"kernel rho {rho_fin:.4f}, outline IoU {iou_fin:.2f}, "
        f"{over*100:.0f}% of cells over the target, {time.time()-t_start:.0f} s", 1)
    if over < 0.35:
        log(f"CHECK only {over*100:.0f}% of source cells lie over the target; "
            f"the two sections may not overlap enough to register", 1)

    if args.figures:
        fig, ax = plt.subplots(1, 3, figsize=(17, PANEL_H))
        prof = C.groupby("deg").score.max()
        ax[0].plot(prof.index, prof.values, lw=1.3, c="#0072B2")
        ax[0].axvline(deg_opt % 360, c="#7A0177", lw=1.4,
                      label=f"adopted, {deg_opt % 360:.1f}\u00b0")
        ax[0].set_xlim(0, 360); ax[0].set_xticks(np.arange(0, 361, 90))
        ax[0].set_xlabel("Rotation (degrees)")
        ax[0].set_ylabel("Best score over translations")
        ax[0].legend(loc="lower right")
        panel(ax[0], "a", f"Joint sweep, {args.coarse_step:.0f}\u00b0 steps")

        ax[1].plot(F.deg, F.rho, lw=1.4, c="#0072B2")
        ax[1].axvspan(ang_lo, ang_hi, color="#7A0177", alpha=.12,
                      label=f"within 0.005 of maximum ({ang_hi-ang_lo:.1f}\u00b0)")
        ax[1].axvline(deg_opt, c="#7A0177", lw=1.4)
        ax[1].set_xlabel("Rotation (degrees)")
        ax[1].set_ylabel("Laminar correspondence (Spearman $\\rho$)")
        ax[1].legend(loc="lower center")
        panel(ax[1], "b", f"Refinement, {args.fine_step:.2f}\u00b0 steps")

        ax[2].scatter(ref[:, 0]/1000, ref[:, 1]/1000, s=20, c=lab-1,
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.32, linewidths=0)
        ax[2].scatter(srcH[:, 0]/1000, srcH[:, 1]/1000, s=.8, c=layer_axis-1,
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.55, linewidths=0)
        ax[2].set_aspect("equal"); ax[2].invert_yaxis()
        add_scalebar(ax[2], 1000.0, "1 mm")
        panel(ax[2], "c", f"Resulting placement (IoU {iou_fin:.2f})")
        domain_legend(fig, y=0.005)
        fig.suptitle("Rigid initialisation of the Xenium section", y=1.0)
        savefig(fig, outdir, "fig_initial_placement")

    return dict(A=affine3(M_opt, t_fin), M=M_opt, t=t_fin, out=srcH,
                mirror=bool(mirror), deg=deg_opt, deg_lo=ang_lo, deg_hi=ang_hi,
                deg_depth_axis=th_deg, rho_kernel=float(rho_fin),
                rho_mirror_false=float(mirror_rho[False]),
                rho_mirror_true=float(mirror_rho[True]),
                mirror_gap=float(mirror_gap),
                mirror_decided=bool(mirror_gap >= MIRROR_MIN_GAP),
                outline_iou=float(iou_fin), frac_over_target=over,
                edge_l1_frac=float((lay_x_true[edge_mask(srcH)] == 0).mean()),
                n_spots_scored=int(n_opt), score_full=score_full)


# ============================================================================
#  figures that compare the two sections directly
# ============================================================================
def fig_panel_d(A, V, xt, yt, wm, layer_axis, lay_x_true, lab, scale, outdir):
    """
    Domains of both sections side by side in shared coordinates, then the
    same restricted to the overlapping region.

    Shared axis limits and one colour scale are what make the two panels
    comparable: a domain at the same figure position in D1 and D2 is at
    the same tissue position. Autoscaling each panel separately would
    defeat the purpose of putting them next to each other.

    The overlap is shown because the source section is larger than the
    target, so distance statistics computed over all cells are dominated
    by cells that have no counterpart at all. Two definitions of overlap
    are compared: geometric proximity to a spot, and the matching
    probability from the mixture model.
    """
    ref  = V.obsm["spatial"]
    srcD = np.c_[xt, yt]
    lay_v = lab - 1
    keep = wm > MATCH_THRESH

    allx = np.r_[srcD[:,0], ref[:,0]]/1000; ally = np.r_[srcD[:,1], ref[:,1]]/1000
    pad  = 0.03*max(np.ptp(allx), np.ptp(ally))
    XL = (allx.min()-pad, allx.max()+pad); YL = (ally.max()+pad, ally.min()-pad)
    fig, ax = plt.subplots(1, 2, figsize=(15, PANEL_H + 2.0))
    ax[0].scatter(srcD[:,0]/1000, srcD[:,1]/1000, s=1.2, c=layer_axis-1,
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    panel(ax[0], "a", f"Xenium, aligned ({A.n_obs:,} cells)", fontsize=10.5)
    ax[1].scatter(ref[:,0]/1000, ref[:,1]/1000, s=26, c=lay_v,
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    panel(ax[1], "b", f"Visium ({V.n_obs:,} spots)", fontsize=10.5)
    for a in ax:
        a.set_xlim(*XL); a.set_ylim(*YL); a.set_aspect("equal")
        add_scalebar(a, 1000.0, "1 mm")
    domain_legend(fig, y=0.03)
    fig.suptitle("Cortical domains of the two sections in a shared frame", y=0.99)
    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    savefig(fig, outdir, "fig_panelD_domains")

    # ---- overlap under two definitions ----
    geo  = cKDTree(ref).query(srcD)[0] <= SPOT_PITCH
    both = geo & keep
    geo_only, mat_only = geo & ~keep, keep & ~geo
    d_sp = (cKDTree(srcD[both]).query(ref)[0] if both.sum()
            else np.full(len(ref), np.inf))
    hit  = d_sp <= scale["spot_r_um"]

    allx = np.r_[srcD[both,0], ref[hit,0]]/1000
    ally = np.r_[srcD[both,1], ref[hit,1]]/1000
    pad  = 0.04*max(np.ptp(allx), np.ptp(ally))
    XL2 = (allx.min()-pad, allx.max()+pad); YL2 = (ally.max()+pad, ally.min()-pad)

    fig, ax = plt.subplots(1, 3, figsize=(19, PANEL_H + 1.0))
    ax[0].scatter(ref[:,0]/1000, ref[:,1]/1000, s=16, c="#dddddd", linewidths=0)
    for m, c, l in ((both, "#009E73", f"both ({both.sum()})"),
                    (geo_only, "#0072B2", f"geometric only ({geo_only.sum()})"),
                    (mat_only, "#D55E00", f"matched only ({mat_only.sum()})")):
        ax[0].scatter(srcD[m,0]/1000, srcD[m,1]/1000, s=.8, c=c,
                      linewidths=0, label=l)
    ax[0].legend(fontsize=8, markerscale=6)
    panel(ax[0], "a", "Agreement between overlap criteria", fontsize=10.5)
    ax[1].scatter(srcD[both,0]/1000, srcD[both,1]/1000, s=1.4,
                  c=(layer_axis-1)[both], cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    panel(ax[1], "b", f"Xenium within the overlap ({both.sum():,} cells)",
          fontsize=10.5)
    ax[2].scatter(ref[hit,0]/1000, ref[hit,1]/1000, s=28, c=lay_v[hit],
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    panel(ax[2], "c", f"Visium within the overlap ({hit.sum():,} spots)",
          fontsize=10.5)
    for a in ax:
        a.set_xlim(*XL2); a.set_ylim(*YL2); a.set_aspect("equal")
        add_scalebar(a, 1000.0, "1 mm")
    domain_legend(fig, y=0.02)
    fig.suptitle("Extent of overlap between the two sections", y=0.99)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    savefig(fig, outdir, "fig_panelD_overlap")

    return dict(overlap_both=int(both.sum()), overlap_geo_only=int(geo_only.sum()),
                overlap_matched_only=int(mat_only.sum()),
                spots_covered=int(hit.sum()), spot_coverage=float(hit.mean()))


def fig_landing(xt, yt, wm, lay_x_true, ref, lab, scale, outdir):
    """
    One panel per domain: aligned cells of that domain over the spots of
    the same domain. Cells on the rings indicate correspondence, cells
    beside them a laminar offset, cells spread across the section that
    the label is not spatially coherent -- which is what a tear does to
    the meninges-containing class.
    """
    d, jsp = cKDTree(ref).query(np.c_[xt, yt])
    sel = (d <= scale["spot_r_um"]) & (wm > MATCH_THRESH) & ~np.isnan(lay_x_true)
    if sel.sum() < 200: return
    xo = lay_x_true[sel].astype(int)
    vo = (lab - 1)[jsp[sel]].astype(int)

    fig, ax = plt.subplots(2, 4, figsize=(21, 10.5)); ax = ax.ravel()
    for i, l in enumerate(LAY):
        a = ax[i]
        a.scatter(ref[:,0]/1000, ref[:,1]/1000, s=7, c="#dddddd", linewidths=0)
        mv = (lab - 1) == i
        a.scatter(ref[mv,0]/1000, ref[mv,1]/1000, s=22, facecolors="none",
                  edgecolors="black", linewidths=0.35)
        m = (lay_x_true == i) & sel
        a.scatter(xt[m]/1000, yt[m]/1000, s=1.6, c=DOM_COLORS[i], linewidths=0)
        hit = (vo[xo == i] == i).mean()*100 if (xo == i).sum() > 20 else np.nan
        a.set_title(f"{l}\n{m.sum():,} cells, {hit:.0f}% exact",
                    fontsize=9, fontweight="bold", loc="left")
        a.set_aspect("equal"); a.invert_yaxis()
        add_scalebar(a, 1000.0, "1 mm")
    ax[7].axis("off")
    ax[7].text(0.02, 0.62,
               "coloured dots  aligned Xenium cells of that domain\n"
               "black rings    Visium spots of the same domain\n"
               "grey           all Visium spots\n\n"
               "Cells on the rings indicate correspondence; cells\n"
               "beside them a laminar offset; cells spread across\n"
               "the section that the domain label is not spatially\n"
               "coherent, as a tear does to the meninges class.",
               fontsize=9, va="top")
    fig.suptitle("Domain-by-domain correspondence after registration", y=1.0)
    plt.tight_layout()
    savefig(fig, outdir, "fig_layer_landing")


def fig_crosstech(A, V, xt, yt, keep, ref, genes, Pu, Vu, xy_u, pat,
                  cs, ss, nzX, nzV, moran, outdir):
    """
    Replica of the cross-technology figure in Clifton 2023: spatial maps
    of target and aligned source, the matched pseudospot-versus-spot
    scatter, and the cosine distributions with the shown genes marked.

    Rows are chosen by an explicit rule rather than by eye: highest,
    median and lowest cosine among patterned genes, plus one unpatterned
    gene as a negative control. Without the control a low-cosine gene
    reads as an alignment failure rather than as the expected behaviour
    of a gene with no spatial structure.

    A density requirement is applied first. Most genes are zero in most
    pseudospots, and a mostly-zero gene gives a flat scatter and a
    Spearman near zero whatever the alignment, so showing one would
    misattribute sampling depth to misregistration. Mitochondrial and
    ribosomal genes are excluded: abundant everywhere, their cosine
    approaches 1 while their Spearman is nil, which is the clearest
    illustration that cosine is mean-dominated across platforms.
    """
    HOUSEKEEP = r"^(MT-|MTRNR|RPL\d|RPS\d)"
    NZ_MIN, VMAX_MIN = 0.50, 10
    sel = pd.DataFrame({"gene": genes, "cosine": cs, "spearman": ss,
                        "moran": moran, "patterned": pat,
                        "nz_X": nzX, "nz_V": nzV, "vmax": Vu.max(0)})
    sel = sel[~sel.gene.str.match(HOUSEKEEP, case=True)]
    deep = (sel.nz_X >= NZ_MIN) & (sel.nz_V >= NZ_MIN) & (sel.vmax >= VMAX_MIN)
    patd = sel[sel.patterned & deep].sort_values("cosine", ascending=False).reset_index(drop=True)
    nonp = sel[~sel.patterned & deep].sort_values("cosine", ascending=False).reset_index(drop=True)
    if len(patd) < 3:
        log(f"crosstech figure skipped: only {len(patd)} patterned genes pass "
            f"the {NZ_MIN*100:.0f}% density requirement", 1)
        return None

    G = [patd.iloc[0].gene, patd.iloc[len(patd)//2].gene, patd.iloc[-1].gene]
    GL = ["patterned, highest cosine", "patterned, median cosine",
          "patterned, lowest cosine"]
    if len(nonp):
        G.append(nonp.iloc[len(nonp)//2].gene); GL.append("not patterned (control)")

    Xall = to_dense(A.X); lib = Xall.sum(1, keepdims=True); lib[lib == 0] = 1
    Xcpm = Xall / lib * 1e6
    Pc, Vc_ = cpm(Pu), cpm(Vu)
    RED = plt.cm.Reds
    def n01(v):
        v = np.asarray(v, float).ravel()
        a_, b_ = np.percentile(v, [1, 99])
        return np.clip((v - a_)/(b_ - a_ + 1e-9), 0, 1)

    nrow = len(G)
    fig = plt.figure(figsize=(16, 3.7*nrow))
    GS = fig.add_gridspec(nrow, 4, width_ratios=[1, 1, 1.05, 1.25],
                          hspace=.30, wspace=.36)
    for i, (g, nm) in enumerate(zip(G, GL)):
        jg = genes.index(g); jx = A.var_names.get_loc(g)
        r_ = sel.set_index("gene").loc[g]

        a = fig.add_subplot(GS[i, 0])
        a.scatter(xy_u[:,0]/1000, xy_u[:,1]/1000, c=n01(Vc_[:, jg]),
                  cmap=RED, s=9, linewidths=0)
        a.set_aspect("equal"); a.invert_yaxis()
        a.set_xticks([]); a.set_yticks([])
        add_scalebar(a, 1000.0, "1 mm", clear_axes=False)   # keeps the ylabel
        a.set_ylabel(f"{g}\n{nm}", fontsize=9.5, rotation=0, labelpad=58,
                     va="center", ha="center")
        if i == 0: panel(a, "a", "Visium (measured)", fontsize=10.5)

        a = fig.add_subplot(GS[i, 1])
        a.scatter(xt[keep]/1000, yt[keep]/1000, c=n01(Xcpm[keep, jx]),
                  cmap=RED, s=.5, linewidths=0)
        a.set_aspect("equal"); a.invert_yaxis(); add_scalebar(a, 1000.0, "1 mm")
        if i == 0: panel(a, "b", "Xenium (aligned)", fontsize=10.5)

        a = fig.add_subplot(GS[i, 2])
        a.scatter(Pc[:, jg], Vc_[:, jg], s=9, c="#444444", alpha=.45, linewidths=0)
        a.set_xlabel("Xenium pseudospot (CPM)")
        a.set_ylabel("Visium spot (CPM)")
        a.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        a.text(.04, .93, f"cosine {r_.cosine:.2f}\nSpearman {r_.spearman:+.2f}\n"
                         f"nonzero {min(r_.nz_X, r_.nz_V)*100:.0f}%",
               transform=a.transAxes, fontsize=8.5, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", lw=.5))
        if i == 0: panel(a, "c", "Correspondence", fontsize=10.5)

    edges = np.linspace(0, 1, 26)
    for k_, (msk, lbl, refv, rows) in enumerate(
            [(pat,  "spatially patterned", 0.55, slice(0, max(nrow-1, 1))),
             (~pat, "not patterned",       0.06, slice(max(nrow-1, 1), nrow))]):
        axd = fig.add_subplot(GS[rows, 3])
        v_ = cs[msk][~np.isnan(cs[msk])]
        axd.barh((edges[:-1]+edges[1:])/2, np.histogram(v_, bins=edges)[0],
                 height=(edges[1]-edges[0])*.9, color="#c8c8c8",
                 edgecolor="white", linewidth=.3)
        axd.axhline(refv, color="#0072B2", lw=1.4, ls="--")
        axd.text(axd.get_xlim()[1]*.98, refv+.02, f"Clifton {refv:.2f}",
                 ha="right", fontsize=8, color="#0072B2")
        med = np.nanmedian(cs[msk])
        axd.axhline(med, color="#333", lw=1.4)
        axd.text(axd.get_xlim()[1]*.98, med-.05, f"ours {med:.2f}",
                 ha="right", fontsize=8)
        for g in G:
            if g in genes and msk[genes.index(g)]:
                c_ = sel.set_index("gene").loc[g, "cosine"]
                axd.axhline(c_, color="#E8000B", lw=2)
                axd.text(axd.get_xlim()[1]*1.02, c_, g, va="center",
                         fontsize=8.5, style="italic")
        axd.set_ylim(0, 1); axd.set_ylabel("Cosine similarity")
        axd.set_xlabel("Genes")
        panel(axd, "d" if k_ == 0 else "e",
              f"{lbl[0].upper()}{lbl[1:]} (n = {msk.sum()})", fontsize=9.5)

    cax = fig.add_axes([0.13, 0.028, 0.20, 0.010])
    plt.colorbar(plt.cm.ScalarMappable(cmap=RED), cax=cax, orientation="horizontal")
    cax.set_xticks([0, 1]); cax.set_xticklabels(["low", "high"])
    cax.set_title("Normalised expression, 1st to 99th percentile", fontsize=8)
    cax.tick_params(labelsize=8)
    fig.suptitle("Gene expression correspondence across platforms", y=1.0)
    savefig(fig, outdir, "fig_crosstech")
    return G


def rasterize(x, y, s, dx=DX_RASTER, blur=BLUR):
    X, Y, img = STalign.rasterize_with_signal(x, y, s=np.asarray(s, float),
                                              dx=dx, blur=blur)
    return X, Y, STalign.normalize(np.repeat(img, 3, axis=0))


def run_lddmm(xI, yI, layer_axis, ref, lab, init, args, device, outdir):
    """
    LDDMM from the edge-butted initialisation. sigmaA is raised above
    sigmaM because a substantial fraction of source tissue has no
    counterpart in the target and must be free to be called artifact.
    """
    XI, YI, I = rasterize(xI, yI, layer_axis)
    XJ, YJ, J = rasterize(ref[:,0], ref[:,1], lab)
    occ_I, occ_J = float((I[0] > 0).mean()), float((J[0] > 0).mean())
    ceiling = ((J.shape[1]*J.shape[2]*DX_RASTER**2*occ_J) /
               (I.shape[1]*I.shape[2]*DX_RASTER**2*occ_I))

    sigma_m = args.sigma_m if args.sigma_m is not None else LDDMM_SIGMA_M
    sigma_b = args.sigma_b if args.sigma_b is not None else LDDMM_SIGMA_B
    sigma_a = args.sigma_a if args.sigma_a is not None else LDDMM_SIGMA_A
    log(f"sigmaM {sigma_m}, sigmaB {sigma_b}, sigmaA {sigma_a} "
        f"(ratio A/M {sigma_a/sigma_m:.1f}; tissue-area ratio {ceiling*100:.0f}%)", 1)

    kw = dict(niter=args.niter, diffeo_start=min(300, args.niter//4),
              a=LDDMM_A_UM, sigmaM=sigma_m, sigmaB=sigma_b, sigmaA=sigma_a,
              sigmaR=1e5,
              epL=1e-8, epT=2e-1, epV=3e1,
              muB=torch.zeros(3, device=device), device=device)
    t0 = time.time()
    out = STalign.LDDMM([YI, XI], I, [YJ, XJ], J,
                        A=torch.tensor(init["A"], dtype=DTYPE, device=device), **kw)
    runtime = time.time() - t0

    tp = STalign.transform_points_source_to_target(
        out["xv"], out["v"], out["A"], np.stack([yI, xI], 1))
    tp = (tp.cpu() if tp.is_cuda else tp).numpy()
    w  = STalign.interp([YJ, XJ], out["WM"][None].float(),
                        torch.tensor(tp)[None].permute(-1, 0, 1).float())
    wm = (w.cpu() if w.is_cuda else w)[0, 0].numpy()
    xt, yt = tp[:, 1], tp[:, 0]

    A_fit = out["A"].cpu().numpy() if hasattr(out["A"], "cpu") else np.asarray(out["A"])
    det_A = float(np.linalg.det(A_fit[:2, :2]))

    # Jacobian on a coarse grid: folding would invalidate the diffeomorphism
    gsd = 200.0
    gx = np.arange(xI.min(), xI.max()+gsd, gsd)
    gy = np.arange(yI.min(), yI.max()+gsd, gsd)
    GX, GY = np.meshgrid(gx, gy)
    gt = STalign.transform_points_source_to_target(
        out["xv"], out["v"], out["A"], np.stack([GY.ravel(), GX.ravel()], 1))
    gt = (gt.cpu() if gt.is_cuda else gt).numpy()
    TY, TX = gt[:,0].reshape(GY.shape), gt[:,1].reshape(GX.shape)
    dXx, dXy = np.gradient(TX, gsd, axis=(1, 0))
    dYx, dYy = np.gradient(TY, gsd, axis=(1, 0))
    Jdet = dXx*dYy - dXy*dYx
    occ = (cKDTree(np.c_[xI, yI]).query(np.c_[GX.ravel(), GY.ravel()])[0]
           .reshape(GX.shape) < gsd)
    Jv = Jdet[occ]

    # Folding means the map is not injective, which shows as a sign change
    # WITHIN the Jacobian field. A uniformly negative field is a reflection:
    # orientation-reversing, but still a valid diffeomorphism. Sections
    # mounted face down are genuinely reflected relative to their pair,
    # which is why the initialisation searches both mirror states at all.
    # Testing Jv > 0 would therefore report every mirrored pair as 100%
    # folded, which is what an earlier version did.
    folded = float((np.sign(Jv) != np.sign(np.median(Jv))).mean())

    # The sign of the determinant records which mirror state was used and is
    # not itself a fault. What would be a fault is the optimiser crossing
    # det = 0, abandoning the state settled at initialisation.
    det_init = float(np.linalg.det(np.asarray(init["A"])[:2, :2]))
    if np.sign(det_A) != np.sign(det_init):
        log(f"CHECK fitted determinant {det_A:+.3f} has the opposite sign to "
            f"the initial {det_init:+.3f}: the optimiser reversed the mirror "
            f"state rather than refining it", 1)

    keep = wm > MATCH_THRESH
    wm_p1 = float(np.percentile(wm, 1))
    saturated = wm_p1 > 0.99
    # A mixture that returns almost nothing between 0.2 and 0.8 is making a
    # hard decision rather than estimating a probability, and the threshold
    # then selects nothing that the sign of the value had not already
    # decided. The fraction in that band is the diagnostic.
    wm_mid = float(((wm > 0.2) & (wm < 0.8)).mean())
    log(f"LDDMM {runtime:.0f} s | matched {keep.mean()*100:.1f}% "
        f"(area ceiling {ceiling*100:.1f}%) | det {det_A:.4f} | "
        f"folding {(Jv<=0).mean()*100:.2f}%", 1)
    if keep.mean() < 0.25:
        log(f"CHECK only {keep.mean()*100:.0f}% of cells matched. Either the "
            f"two sections overlap too little to register, or the placement "
            f"is wrong; inspect fig_initial_placement before using the output.",
            1)
    if saturated:
        log(f"NOTE match_prob saturated (1st pct {wm_p1:.3f}); the mixture "
            f"model did not discriminate, so use the geometric criterion "
            f"downstream rather than the threshold", 1)
    if wm_mid < 0.02:
        log(f"CHECK only {wm_mid*100:.1f}% of cells score between 0.2 and 0.8: "
            f"the mixture is behaving as a hard classifier, so the "
            f"{MATCH_THRESH} threshold is selecting little that the values "
            f"had not already decided. Raising sigmaM softens it.", 1)

    if args.figures:
        fig, ax = plt.subplots(1, 3, figsize=(17, PANEL_H))
        s_ = ax[0].scatter(xt/1000, yt/1000, s=.6, c=wm, cmap="RdYlGn",
                           vmin=0, vmax=1, linewidths=0)
        panel(ax[0], "a", f"Matching probability "
                          f"({keep.mean()*100:.0f}% above {MATCH_THRESH})")
        cbar(s_, ax[0], "Probability")
        ax[1].scatter(ref[:,0]/1000, ref[:,1]/1000, s=9, c=lab-1, cmap=CMAP_DOM,
                      norm=NORM_DOM, alpha=.35, linewidths=0)
        ax[1].scatter(xt[keep]/1000, yt[keep]/1000, s=.6, c=(layer_axis-1)[keep],
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.55, linewidths=0)
        panel(ax[1], "b", "Aligned Xenium over Visium domains")
        im = ax[2].imshow(np.where(occ, Jdet, np.nan), cmap="coolwarm",
                          origin="lower", vmin=0.6, vmax=1.4,
                          extent=[gx.min()/1000, gx.max()/1000,
                                  gy.min()/1000, gy.max()/1000])
        ax[2].invert_yaxis(); ax[2].set_aspect("equal")
        cbar(im, ax[2], "Local area change")
        panel(ax[2], "c", f"Jacobian determinant "
                          f"(median {np.median(Jv):.3f})")
        for a in (ax[0], ax[1], ax[2]):
            a.set_aspect("equal")
            if a is not ax[2]: a.invert_yaxis()
            add_scalebar(a, 1000.0, "1 mm")
        domain_legend(fig, y=0.005)
        fig.suptitle("Diffeomorphic registration of Xenium to Visium", y=1.0)
        savefig(fig, outdir, "fig_registration")

    return dict(x=xt, y=yt, wm=wm, keep=keep, runtime=runtime,
                det_A=det_A, mirrored=bool(det_A < 0),
                ceiling=float(ceiling), sigma_a=float(sigma_a),
                wm_p1=wm_p1, wm_saturated=bool(saturated), wm_mid=wm_mid,
                sigma_m=float(sigma_m), sigma_b=float(sigma_b),
                jac_med=float(np.median(Jv)), jac_p5=float(np.percentile(Jv,5)),
                jac_p95=float(np.percentile(Jv,95)),
                folded=folded, det_init=det_init,
                mirror_reversed=bool(np.sign(det_A) != np.sign(det_init)))


# ============================================================================
#  evaluation
# ============================================================================
def eval_layers(xt, yt, wm, lay_x_true, ref, lab, scale, outdir, args):
    """
    Laminar concordance. These labels drive the registration, so this
    bounds alignment quality from above rather than testing it
    independently; the orthogonal test is expression concordance.
    """
    d, jsp = cKDTree(ref).query(np.c_[xt, yt])
    sel = (d <= scale["spot_r_um"]) & (wm > MATCH_THRESH) & ~np.isnan(lay_x_true)
    if sel.sum() < 500:
        return dict(n=int(sel.sum()))
    xo = lay_x_true[sel].astype(int)
    vo = (lab - 1)[jsp[sel]].astype(int)

    same = float((xo == vo).mean())
    adj  = float((np.abs(xo - vo) <= 1).mean())
    rho  = float(spearmanr(xo, vo).statistic)
    rng  = np.random.default_rng(0)
    base = np.array([(np.abs(rng.permutation(xo) - vo) <= 1).mean() for _ in range(200)])
    lift = float((adj - base.mean()) / base.std())

    gm, wmi = [1,2,3,4], [5,6]
    gx_, wx_ = np.isin(xo, gm), np.isin(xo, wmi)
    gv_ = np.isin(vo, gm)
    both = gx_ | wx_
    gmwm = float(((gx_ == gv_)[both]).mean()) if both.sum() else np.nan
    # A two-class accuracy has a floor set by the class balance, not by 50%:
    # where one compartment dominates, guessing that compartment already
    # scores highly. Reporting the accuracy alone leaves a near-chance
    # result and a strong one looking alike, so the permutation baseline
    # and the excess over it are carried alongside.
    if both.sum() > 50:
        rng2 = np.random.default_rng(0)
        gb = np.array([(gx_[both] == rng2.permutation(gv_[both])).mean()
                       for _ in range(200)])
        gmwm_base, gmwm_sigma = float(gb.mean()), float((gmwm - gb.mean())/gb.std())
    else:
        gmwm_base, gmwm_sigma = np.nan, np.nan
    mm = gx_ & gv_
    rho_gm = (float(spearmanr(xo[mm], vo[mm]).statistic)
              if mm.sum() > 50 and len(np.unique(xo[mm])) > 1 else np.nan)

    per = {}
    for i, l in enumerate(LAY):
        m = xo == i
        if m.sum() < 20: continue
        per[l] = dict(n=int(m.sum()), exact=float((vo[m] == i).mean()),
                      mean_landed=float(vo[m].mean()), bias=float(vo[m].mean() - i))
    log(f"layers: exact {same*100:.1f}%, +/-1 {adj*100:.1f}% ({lift:.0f} sigma), "
        f"GM/WM {gmwm*100:.1f}% (baseline {gmwm_base*100:.0f}%, "
        f"{gmwm_sigma:.0f} sigma), within-GM rho {rho_gm:+.3f}", 1)

    if args.figures:
        C = np.zeros((7, 7)); np.add.at(C, (xo, vo), 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, PANEL_H))
        im = ax[0].imshow(C / C.sum(1, keepdims=True).clip(1), cmap="magma",
                          vmin=0, vmax=1)
        ax[0].set_xticks(range(7)); ax[0].set_xticklabels(LAY, rotation=45, fontsize=7)
        ax[0].set_yticks(range(7)); ax[0].set_yticklabels(LAY, fontsize=7)
        ax[0].set_xlabel("Visium domain at landing site")
        ax[0].set_ylabel("Xenium domain of origin")
        ax[0].plot([-.5,6.5], [-.5,6.5], c="#00E5FF", lw=.7, alpha=.6)
        ax[0].axvline(4.5, c="#00E5FF", lw=.7, ls="--", alpha=.5)
        ax[0].axhline(4.5, c="#00E5FF", lw=.7, ls="--", alpha=.5)
        for sp in ax[0].spines.values():
            sp.set_visible(True); sp.set_color("#141413"); sp.set_linewidth(0.8)
        cbar(im, ax[0], "Fraction of domain")
        panel(ax[0], "a", f"Confusion matrix ({same*100:.0f}% exact, "
                          f"{adj*100:.0f}% within one domain)")
        mu = np.array([vo[xo == i].mean() if (xo == i).sum() > 20 else np.nan
                       for i in range(7)])
        sd = np.array([vo[xo == i].std() if (xo == i).sum() > 20 else np.nan
                       for i in range(7)])
        ax[1].plot([0,6], [0,6], c="#bbb", ls="--", lw=1, label="no offset")
        ax[1].errorbar(range(7), mu, yerr=sd, fmt="-o", ms=5, c="#0072B2",
                       capsize=3, label="observed, mean \u00b1 SD")
        ax[1].axvspan(0.5, 4.5, color="#009E73", alpha=.07)
        ax[1].text(2.5, 6.55, "grey matter", ha="center", fontsize=8,
                   color="#009E73")
        ax[1].set_xticks(range(7)); ax[1].set_xticklabels(LAY, rotation=45, fontsize=8)
        ax[1].set_yticks(range(7)); ax[1].set_yticklabels(LAY, fontsize=8)
        ax[1].set_xlabel("Xenium domain of origin")
        ax[1].set_ylabel("Mean Visium domain at landing site")
        ax[1].legend(loc="upper left")
        for sp in ax[1].spines.values():
            sp.set_visible(True); sp.set_color("#141413"); sp.set_linewidth(0.8)
        panel(ax[1], "b", "Systematic laminar offset")
        fig.suptitle("Laminar concordance between aligned cells and spots", y=1.0)
        savefig(fig, outdir, "fig_layer_concordance")

    return dict(n=int(sel.sum()), exact=same, within1=adj, spearman=rho,
                sigma=lift, gm_wm_acc=gmwm, gm_wm_baseline=gmwm_base,
                gm_wm_sigma=gmwm_sigma, within_gm_rho=rho_gm, per_layer=per)


def eval_expression(A, V, xt, yt, keep, ref, lab, scale, outdir, args):
    """
    Orthogonal evaluation: shared-gene expression, which never enters the
    registration. Cross-technology protocol from Clifton 2023 -- source
    cells summed into pseudospots at the target spot positions, CPM and
    log10, cosine compared between patterned and unpatterned strata.
    """
    shared = sorted(set(A.var_names) & set(V.var_names))
    Xk = to_dense(A.X[keep])
    d, sp = cKDTree(ref).query(np.c_[xt[keep], yt[keep]])
    inr = d <= scale["spot_r_um"]
    xi = [A.var_names.get_loc(g) for g in shared]
    vi = [V.var_names.get_loc(g) for g in shared]

    P = np.zeros((V.n_obs, len(shared)))
    np.add.at(P, sp[inr], Xk[np.ix_(inr, xi)])
    Vm = to_dense(V.X)[:, vi]
    ncell = np.bincount(sp[inr], minlength=V.n_obs)
    occ = ncell >= 1
    gene_ok = ((P[occ].sum(0) >= 1) & (Vm[occ].sum(0) >= 1) &
               ((Vm[occ] > 0).sum(0) > 1))
    genes = [g for g, k in zip(shared, gene_ok) if k]
    Pu, Vu = P[np.ix_(occ, gene_ok)], Vm[np.ix_(occ, gene_ok)]
    xy = ref[occ]
    if len(genes) < 50 or occ.sum() < 200:
        return dict(n_genes=len(genes), n_pseudospots=int(occ.sum()))

    # Moran's I, MERINGUE-style: Delaunay adjacency and a randomization z-test
    tri = Delaunay(xy); E = set()
    for s_ in tri.simplices:
        for a_ in range(3):
            for b_ in range(a_+1, 3):
                E.add(tuple(sorted((s_[a_], s_[b_]))))
    E = np.array(sorted(E))
    E = E[np.linalg.norm(xy[E[:,0]] - xy[E[:,1]], axis=1) <= 1.5*SPOT_PITCH]
    W = np.zeros((len(xy), len(xy)))
    W[E[:,0], E[:,1]] = 1; W[E[:,1], E[:,0]] = 1
    S0 = W.sum(); S1 = 0.5*((W+W.T)**2).sum(); S2 = ((W.sum(1)+W.sum(0))**2).sum()
    n = len(xy); EI = -1.0/(n-1)

    Vc = cpm(Vu)
    mo, mp = np.zeros(len(genes)), np.ones(len(genes))
    for j in range(len(genes)):
        z = Vc[:, j] - Vc[:, j].mean(); zz = (z**2).sum()
        if zz == 0: continue
        I_ = (n/S0) * (z @ W @ z) / zz
        b2 = n*(z**4).sum()/zz**2
        Aq = n*((n**2-3*n+3)*S1 - n*S2 + 3*S0**2)
        Bq = b2*((n**2-n)*S1 - 2*n*S2 + 6*S0**2)
        VI = (Aq-Bq)/((n-1)*(n-2)*(n-3)*S0**2) - EI**2
        mo[j] = I_
        mp[j] = 1 - norm.cdf((I_-EI)/np.sqrt(VI)) if VI > 0 else 1.0
    o = np.argsort(mp); q = np.empty(len(genes))
    q[o] = np.minimum.accumulate((mp[o]*len(genes)/np.arange(1, len(genes)+1))[::-1])[::-1]
    pat = np.clip(q, 0, 1) < 0.05

    lg = lambda M: np.log10(cpm(M) + 1)
    Pn, Vn = lg(Pu), lg(Vu)
    cosf = lambda a_, b_: (a_@b_/(np.linalg.norm(a_)*np.linalg.norm(b_))
                           if a_.any() and b_.any() else np.nan)
    cs = np.array([cosf(Pn[:,j], Vn[:,j]) for j in range(len(genes))])
    ss = np.array([spearmanr(Pn[:,j], Vn[:,j]).statistic
                   if Pn[:,j].std() and Vn[:,j].std() else np.nan
                   for j in range(len(genes))])
    nzX, nzV = (Pu > 0).mean(0), (Vu > 0).mean(0)
    dense = (nzX >= 0.6) & (nzV >= 0.6)
    is_wm = np.isin(genes, WM_MARKERS)          # exact match, never substring
    u = (mannwhitneyu(cs[pat][~np.isnan(cs[pat])], cs[~pat][~np.isnan(cs[~pat])],
                      alternative="greater") if pat.sum() and (~pat).sum() else None)

    res = dict(n_genes=len(genes), n_pseudospots=int(occ.sum()),
               n_patterned=int(pat.sum()),
               cells_per_pseudospot=float(np.median(ncell[occ])),
               cosine_patterned=float(np.nanmedian(cs[pat])),
               cosine_unpatterned=float(np.nanmedian(cs[~pat])),
               cosine_separation=float(np.nanmedian(cs[pat]) - np.nanmedian(cs[~pat])),
               spearman_patterned=float(np.nanmedian(ss[pat])),
               spearman_dense=float(np.nanmedian(ss[dense])) if dense.sum() else np.nan,
               n_dense=int(dense.sum()),
               cosine_wm_markers=float(np.nanmedian(cs[is_wm])) if is_wm.sum() else np.nan,
               wilcoxon_p=float(u.pvalue) if u else np.nan)
    log(f"expression: {pat.sum()}/{len(genes)} patterned, cosine "
        f"{res['cosine_patterned']:.3f} vs {res['cosine_unpatterned']:.3f} "
        f"(separation {res['cosine_separation']:.3f})", 1)

    if args.figures:
        shown = fig_crosstech(A, V, xt, yt, keep, ref, genes, Pu, Vu, xy,
                              pat, cs, ss, nzX, nzV, mo, outdir)
        if shown: res["genes_shown"] = shown

    if args.figures:
        fig, ax = plt.subplots(1, 2, figsize=(12, PANEL_H))
        b_ = np.linspace(0, 1, 26)
        ax[0].hist(cs[pat][~np.isnan(cs[pat])], bins=b_, color="#0072B2", alpha=.7,
                   label=f"patterned (n={pat.sum()})")
        ax[0].hist(cs[~pat][~np.isnan(cs[~pat])], bins=b_, color="#BBBBBB", alpha=.7,
                   label=f"not patterned (n={(~pat).sum()})")
        ax[0].axvline(0.55, c="#D55E00", ls="--", lw=1.2,
                      label="published, patterned")
        ax[0].axvline(0.06, c="#999999", ls="--", lw=1.2,
                      label="published, non-patterned")
        ax[0].set_xlabel("Cosine similarity"); ax[0].set_ylabel("Genes")
        ax[0].legend(loc="upper left", fontsize=8)
        panel(ax[0], "a", "Correspondence by spatial patterning")
        for m_, c_, l_ in ((pat, "#0072B2", "spatially patterned"),
                           (~pat, "#BBBBBB", "not patterned")):
            ax[1].scatter(np.minimum(nzX, nzV)[m_]*100, ss[m_], s=11,
                          c=c_, linewidths=0, label=l_)
        ax[1].axvline(60, c="#D55E00", ls="--", lw=1.2,
                      label="60% detection")
        ax[1].axhline(0, c="#ccc", lw=.6)
        ax[1].set_xlabel("Genes detected, sparser platform (% of spots)")
        ax[1].set_ylabel("Rank correlation (Spearman $\\rho$)")
        ax[1].legend(loc="upper left", fontsize=8)
        panel(ax[1], "b", "Detection rate limits the rank statistic")
        for a_ in ax:
            for sp_ in a_.spines.values():
                sp_.set_visible(True); sp_.set_color("#141413"); sp_.set_linewidth(0.8)
        fig.suptitle("Shared-gene expression concordance", y=1.0)
        savefig(fig, outdir, "fig_expression_concordance")

    return res


# ============================================================================
#  outputs
# ============================================================================
def write_outputs(info, A, V, reg, init, scale, lay_x_true, lab, outdir, metrics):
    s = info["sample"]
    xt, yt, wm = reg["x"], reg["y"], reg["wm"]
    ref = V.obsm["spatial"]

    A.obs["match_prob"]   = wm
    A.obsm["aligned"]     = np.c_[xt, yt]
    A.obsm["aligned_px"]  = np.c_[xt, yt] * scale["px_per_um"]
    d, jsp = cKDTree(ref).query(np.c_[xt, yt])
    # `matched` alone is not a safe filter: where the two sections have
    # comparable area the mixture model saturates and every cell passes the
    # threshold, including cells transformed outside the target. `usable`
    # adds the geometric requirement, which holds regardless.
    inside  = d <= SPOT_PITCH
    matched = wm > MATCH_THRESH
    usable  = matched & inside
    in_spot = (d <= scale["spot_r_um"]) & matched
    A.obs["visium_spot"]   = np.where(in_spot, np.array(V.obs_names)[jsp.clip(0)], None)
    A.obs["spot_dist_um"]  = d
    A.obs["inside_visium"] = inside
    A.obs["usable"]        = usable
    A.uns["alignment"] = {**metrics}

    exp = pd.DataFrame({
        "cell_id":      A.obs["cell_id_xe"].values,
        "orig_row":     A.obs["orig_row"].values,
        "lieber_key":   A.obs["lieber_key"].values,
        "x":            A.obsm["spatial"][:, 0],
        "y":            A.obsm["spatial"][:, 1],
        "x_aligned":    xt, "y_aligned": yt,
        "x_aligned_px": A.obsm["aligned_px"][:, 0],
        "y_aligned_px": A.obsm["aligned_px"][:, 1],
        "match_prob":   wm,
        "matched":      matched,
        "inside_visium": inside,
        "usable":       usable,
        "visium_spot":  A.obs["visium_spot"].values,
        "spot_dist_um": d,
        "layer":        A.obs["layer"].values,
        "cell_type":    A.obs["cell_type"].values})
    fcsv = os.path.join(outdir, f"{s}_aligned_coordinates.csv")
    exp.to_csv(fcsv, index=False)

    vexp = pd.DataFrame({
        "barcode": V.obs_names,
        "x": ref[:, 0], "y": ref[:, 1],
        "x_px": V.obsm["spatial_px"][:, 0], "y_px": V.obsm["spatial_px"][:, 1],
        "spd_label": V.obs["spd_label"].values if "spd_label" in V.obs else None,
        "domain": [LAY[int(v)-1] if np.isfinite(v) else None for v in lab]})
    vexp.to_csv(os.path.join(outdir, f"{s}_visium_coordinates.csv"), index=False)

    open(os.path.join(outdir, f"{s}_aligned_coordinates.README"), "w").write(f"""\
{s}_aligned_coordinates.csv
Xenium cells aligned to Visium capture area {info['capture']}.
Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}.

COLUMNS
  cell_id        Xenium cell_id, matches cells.parquet
  orig_row       1-based row in cells.parquet
  lieber_key     key used by the Lieber annotation tables
  x, y           Xenium native coordinates, MICRONS, unchanged
  x_aligned      aligned coordinates in the Visium frame, MICRONS
  y_aligned
  x_aligned_px   the same in Visium full-resolution image PIXELS
  y_aligned_px
  match_prob     STalign matching probability, 0 to 1
  matched        match_prob > {MATCH_THRESH}
  inside_visium  within {SPOT_PITCH:.0f} um of a Visium spot centre
  usable         matched AND inside_visium -- the column to filter on
  visium_spot    barcode of the spot the cell falls in, else empty
  spot_dist_um   distance to the nearest spot centre, microns
  layer          Lieber spatial domain
  cell_type      BANKSY cell type

FRAMES AND UNITS
  Aligned coordinates are microns in the Visium frame. Divide by
  {scale['um_per_px']:.6f} for full-resolution image pixels; the pixel
  columns have this applied already. For the hires image, multiply the
  pixel values by tissue_hires_scalef from scalefactors_json.json. The
  scale factor is specific to this capture area: across the cohort it
  varies, so it cannot be reused for another section.

RESTRICTIONS
  Filter on `usable`, not on `matched`. match_prob comes from a mixture
  model whose artifact component only has work to do when the source
  section exceeds the target; where the two are comparable it saturates
  and `matched` is True for every cell, including cells transformed
  outside the Visium tissue. `usable` adds the geometric requirement.
  Add spot_dist_um <= {scale['spot_r_um']:.2f} when a cell must sit
  inside a spot rather than merely near one.

  Spot discs sample {scale['disc_coverage']*100:.0f}% of tissue area at a
  {SPOT_PITCH:.0f} um pitch, so most cells fall between spots by
  construction rather than through misalignment.

  cells                {len(exp)}
  matched              {int(exp.matched.sum())}
  inside the target    {int(exp.inside_visium.sum())}
  usable               {int(exp.usable.sum())}
  assigned to a spot   {int(exp.visium_spot.notna().sum())}
  distinct spots hit   {exp.visium_spot.nunique()} of {V.n_obs}
""")

    # The h5ad objects are not written. Everything downstream needs is in
    # the two CSVs, and regenerating an object from them is cheaper than
    # storing several hundred megabytes per donor.
    json.dump(metrics, open(os.path.join(outdir, f"{s}_metrics.json"), "w"),
              indent=2, default=float)
    log(f"{s}_aligned_coordinates.csv   {len(exp)} cells, "
        f"{int(exp.matched.sum())} matched, {int(exp.usable.sum())} usable", 1)
    log(f"{s}_visium_coordinates.csv    {V.n_obs} spots", 1)
    log(f"{s}_metrics.json", 1)
    return exp


# ============================================================================
#  per-sample driver
# ============================================================================
def process_sample(sample, manifest, args, device):
    global _REPORT
    t_all = time.time()
    print(f"\n{'='*70}\n  {sample}\n{'='*70}", flush=True)
    info = resolve_sample(sample, manifest, args)
    log(f"capture area {info['capture']} | dx {info['dx']} | "
        f"{info['n_spots_expected']} spots expected")

    outdir = os.path.join(args.outdir, f"{sample}_STalign_result")
    os.makedirs(outdir, exist_ok=True)
    _REPORT = Report(sample, outdir)
    _REPORT.step("Inputs", code=
        "info  = resolve_sample(sample, manifest, args)\n"
        "vdir  = ensure_visium_export(info, args)      # Rscript spe_export.R\n"
        "srdir = ensure_spaceranger(info['capture'], args)")
    log(f"donor {info['donor']} | capture area {info['capture']} | "
        f"dx {info['dx']} | sex {info['sex']} | age {info['age']}")

    vdir  = ensure_visium_export(info, args)
    srdir = ensure_spaceranger(info["capture"], args)

    _REPORT.step("Loading and physical scale", code=
        "A, n_empty = load_xenium(info, args)       # centroids are microns as stored\n"
        "V, scale   = load_visium(info, vdir, srdir)\n"
        "#   um/px = SPOT_PITCH / median nearest-neighbour distance,\n"
        "#   recovered per capture area and checked against the array geometry\n"
        "marker_axis, cov, join_rho = attach_labels(A, sample, cache)\n"
        "layer_axis, lay_x_true, lab, bad = build_ordinals(A, V, marker_axis)")
    A, n_empty = load_xenium(info, args)
    V, scale   = load_visium(info, vdir, srdir)
    marker_axis, cov, join_rho = attach_labels(A, sample, os.path.join(args.workdir,
                                                                      "lieber_cache"))
    layer_axis, lay_x_true, lab, bad_spd = build_ordinals(A, V, marker_axis)
    if bad_spd.any():
        V = V[~bad_spd].copy(); lab = lab[~bad_spd]
    ref = V.obsm["spatial"]
    xI, yI = A.obsm["spatial"].T

    _REPORT.step("Initialisation", code=
        "init = initial_affine(xI, yI, layer_axis, lay_x_true, ref, lab,\n"
        "                      args, outdir)\n"
        "#   mirror by binned laminar correlation, then rotation and\n"
        "#   translation jointly, with the upper-left pial boundaries brought\n"
        "#   into contact and the boundary re-extracted at every angle\n"
        "#   scored by a Gaussian kernel, sigma = 150 um, so the score is\n"
        "#   continuous in the transform and sub-degree steps mean something")
    init = initial_affine(xI, yI, layer_axis, lay_x_true, ref, lab, args, outdir)
    _REPORT.step("Diffeomorphic registration", code=
        "reg = run_lddmm(xI, yI, layer_axis, ref, lab, init, args,\n"
        "                device, outdir)\n"
        "#   sigmaA follows the tissue-area ratio: unmatched source tissue\n"
        "#   needs somewhere to go, and only exists where the source is larger")
    reg  = run_lddmm(xI, yI, layer_axis, ref, lab, init, args, device, outdir)

    rho_pre,  _ = binned_corr(init["out"], layer_axis, ref, lab)
    rho_post, _ = binned_corr(np.c_[reg["x"], reg["y"]], layer_axis, ref, lab)
    _REPORT.step("Laminar concordance", code=
        "lay_eval = eval_layers(reg['x'], reg['y'], reg['wm'], lay_x_true,\n"
        "                       ref, lab, scale, outdir, args)\n"
        "#   these annotations drive the registration, so this bounds\n"
        "#   alignment quality from above rather than testing it")
    lay_eval = eval_layers(reg["x"], reg["y"], reg["wm"], lay_x_true, ref, lab,
                           scale, outdir, args)
    ov = {}
    if args.figures:
        _REPORT.step("Sections side by side", code=
        "ov = fig_panel_d(A, V, reg['x'], reg['y'], reg['wm'], layer_axis,\n"
        "                 lay_x_true, lab, scale, outdir)\n"
        "fig_landing(reg['x'], reg['y'], reg['wm'], lay_x_true, ref, lab,\n"
        "            scale, outdir)")
        ov = fig_panel_d(A, V, reg["x"], reg["y"], reg["wm"], layer_axis,
                         lay_x_true, lab, scale, outdir) or {}
        fig_landing(reg["x"], reg["y"], reg["wm"], lay_x_true, ref, lab,
                    scale, outdir)
    _REPORT.step("Expression concordance", code=
        "exp_eval = eval_expression(A, V, reg['x'], reg['y'], reg['keep'],\n"
        "                           ref, lab, scale, outdir, args)\n"
        "#   shared-gene expression enters the registration at no point,\n"
        "#   so this is the independent evaluation")
    exp_eval = eval_expression(A, V, reg["x"], reg["y"], reg["keep"], ref, lab,
                               scale, outdir, args)

    metrics = dict(
        sample=sample, capture_area=info["capture"], donor=info["donor"],
        dx=info["dx"], sex=info["sex"], age=info["age"],
        n_cells=int(A.n_obs), n_spots=int(V.n_obs), n_empty_cells=int(n_empty),
        n_genes_xenium=int(A.n_vars),
        xenium_source=("prepared: " + args.xenium_h5ad.format(sample=sample)
                       if args.xenium_h5ad else "raw parquet + h5"),
        qc_drop_empty=(not args.keep_empty) and not args.xenium_h5ad,
        qc_flag_outliers=(not args.no_qc) and not args.xenium_h5ad,
        annotation_coverage=cov, label_join_rho=join_rho,
        units="microns", **{f"scale_{k}": v for k, v in scale.items()},
        init_mirror=init["mirror"], init_mirror_gap=init["mirror_gap"],
        init_mirror_decided=init["mirror_decided"], init_deg=init["deg"],
        init_deg_lo=init["deg_lo"], init_deg_hi=init["deg_hi"],
        init_deg_depth_axis=init["deg_depth_axis"],
        init_rho_kernel=init["rho_kernel"],
        init_outline_iou=init["outline_iou"],
        init_frac_over_target=init["frac_over_target"],
        init_edge_l1_frac=init["edge_l1_frac"],
        lddmm_niter=args.niter, lddmm_a_um=LDDMM_A_UM,
        lddmm_runtime_s=reg["runtime"], device=str(device),
        matched_frac=float(reg["keep"].mean()), area_ceiling=reg["ceiling"],
        lddmm_sigmaA=reg["sigma_a"], lddmm_sigmaM=reg["sigma_m"],
        lddmm_sigmaB=reg["sigma_b"], match_prob_mid_frac=reg["wm_mid"],
        match_prob_p1=reg["wm_p1"],
        match_prob_saturated=reg["wm_saturated"],
        fitted_affine_det=reg["det_A"], mirrored=reg["mirrored"],
        initial_affine_det=reg["det_init"], mirror_reversed=reg["mirror_reversed"],
        jacobian_median=reg["jac_med"],
        jacobian_p5=reg["jac_p5"], jacobian_p95=reg["jac_p95"],
        folded_frac=reg["folded"],
        rho_binned_pre=float(rho_pre), rho_binned_post=float(rho_post),
        **{f"layer_{k}": v for k, v in lay_eval.items() if k != "per_layer"},
        layer_per_layer=lay_eval.get("per_layer", {}),
        **{f"expr_{k}": v for k, v in exp_eval.items() if not isinstance(v, list)},
        **{f"overlap_{k}" if not k.startswith("overlap") else k: v
           for k, v in ov.items()},
        total_runtime_s=time.time() - t_all)

    _REPORT.step("Outputs", code=
        "exp = write_outputs(info, A, V, reg, init, scale, lay_x_true, lab,\n"
        "                    outdir, metrics)\n"
        "#   CSV only; filter on `usable`, which is `matched` and within one\n"
        "#   spot pitch, since match_prob saturates on comparable areas")
    exp = write_outputs(info, A, V, reg, init, scale, lay_x_true, lab,
                        outdir, metrics)
    metrics["n_usable"] = int(exp.usable.sum())
    metrics["n_inside_visium"] = int(exp.inside_visium.sum())
    json.dump(metrics, open(os.path.join(outdir, f"{sample}_metrics.json"), "w"),
              indent=2, default=float)

    _REPORT.param(**{
        "donor":                    info["donor"],
        "capture area":             info["capture"],
        "diagnosis":                info["dx"],
        "spot pitch (um)":          SPOT_PITCH,
        "pixel scale (um/px)":      f"{scale['um_per_px']:.6f}",
        "spot radius (um)":         f"{scale['spot_r_um']:.2f}",
        "raster step (um)":         DX_RASTER,
        "raster blur (px)":         BLUR,
        "kernel sigma (um)":        KERNEL_SIGMA,
        "kernel min support":       KERNEL_MINSUP,
        "coarse angle step (deg)":  args.coarse_step,
        "fine angle step (deg)":    args.fine_step,
        "translation span (um)":    f"+/-{args.trans_span:.0f}",
        "LDDMM iterations":         args.niter,
        "LDDMM kernel a (um)":      LDDMM_A_UM,
        "sigmaM":                   reg["sigma_m"],
        "sigmaB":                   reg["sigma_b"],
        "sigmaA":                   reg["sigma_a"],
        "sigmaA / sigmaM":          f"{reg['sigma_a']/reg['sigma_m']:.1f}",
        "sigmaR":                   "1e5",
        "matching threshold":       MATCH_THRESH,
        "Xenium source":            metrics["xenium_source"],
        "QC: drop empty":           metrics["qc_drop_empty"],
        "QC: flag outliers":        metrics["qc_flag_outliers"],
        "device":                   str(device),
    })
    _REPORT.metric(**{
        "cells / spots":            f"{A.n_obs} / {V.n_obs}",
        "annotation coverage":      f"{cov*100:.1f}%",
        "label join rho":           f"{join_rho:.3f}",
        "array check x, y":         f"{scale['array_dev_x']*100:+.2f}%, "
                                    f"{scale['array_dev_y']*100:+.2f}%",
        "rotation (deg)":           f"{init['deg']:.1f} "
                                    f"[{init['deg_lo']:.1f}, {init['deg_hi']:.1f}]",
        "mirror":                   f"{init['mirror']} (gap "
                                    f"{init['mirror_gap']:.3f}"
                                    + ("" if init["mirror_decided"] else ", UNDECIDED")
                                    + ")",
        "tissue-area ceiling":      f"{reg['ceiling']*100:.1f}%",
        "matched fraction":         f"{reg['keep'].mean()*100:.1f}%",
        "match_prob saturated":     reg["wm_saturated"],
        "match_prob in 0.2-0.8":    f"{reg['wm_mid']*100:.1f}%",
        "usable cells":             f"{metrics['n_usable']} of {A.n_obs}",
        "fitted affine det":        f"{reg['det_A']:.4f}"
                                    + ("  (mirrored)" if reg["mirrored"] else "")
                                    + ("  SIGN REVERSED" if reg["mirror_reversed"] else ""),
        "outline IoU at placement":  f"{init['outline_iou']:.2f}",
        "cells over the target":     f"{init['frac_over_target']*100:.0f}%",
        "Jacobian median":          f"{reg['jac_med']:.3f}",
        "Jacobian p5 - p95":        f"{reg['jac_p5']:.3f} - {reg['jac_p95']:.3f}",
        "folded fraction":          f"{reg['folded']*100:.2f}%",
        "layer signal rho":         f"{rho_pre:.3f} -> {rho_post:.3f}",
        "GM/WM accuracy":           f"{lay_eval.get('gm_wm_acc', float('nan'))*100:.1f}% "
                                    f"(baseline "
                                    f"{lay_eval.get('gm_wm_baseline', float('nan'))*100:.0f}%, "
                                    f"{lay_eval.get('gm_wm_sigma', float('nan')):.0f} sigma)",
        "within-GM rho":            f"{lay_eval.get('within_gm_rho', float('nan')):+.3f}",
        "exact / within one layer": f"{lay_eval.get('exact', float('nan'))*100:.1f}% / "
                                    f"{lay_eval.get('within1', float('nan'))*100:.1f}%",
        "patterned genes":          f"{exp_eval.get('n_patterned', 0)} of "
                                    f"{exp_eval.get('n_genes', 0)}",
        "cosine separation":        f"{exp_eval.get('cosine_separation', float('nan')):.3f}",
        "LDDMM runtime (s)":        f"{reg['runtime']:.0f}",
    })
    _REPORT.write()
    log(f"report -> {sample}_report.html, log -> {sample}_run.log", 1)
    log(f"done in {metrics['total_runtime_s']:.0f} s")
    _REPORT = None
    return metrics


# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", nargs="*", help="donor name(s), e.g. Br8667")
    ap.add_argument("--all", action="store_true", help="every donor with both platforms")
    ap.add_argument("--list", action="store_true", help="show the manifest and exit")
    ap.add_argument("--root", default=os.path.expanduser("~/Guo_xenium"))
    ap.add_argument("--rds", default=None, help="default <root>/fnl_spe_kept_spots_only.rds")
    ap.add_argument("--xenium-dir", default=None, help="default <root>/xenium")
    ap.add_argument("--workdir", default=None, help="exports and caches; default <root>")
    ap.add_argument("--outdir", default=None, help="results; default <root>/stalign_results")
    ap.add_argument("--export-root", default=None,
                    help="where the per-donor Visium exports go; "
                         "default <workdir>/visium_exports")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="donors to leave out, e.g. one whose Xenium files are "
                         "known bad and are being re-downloaded")
    ap.add_argument("--rscript", default=None, help="default alongside this script")
    ap.add_argument("--niter", type=int, default=2000)
    ap.add_argument("--coarse-step", type=float, default=5.0,
                    help="degrees, first pass over the full circle; the "
                         "translation grid is searched at every angle, so "
                         "this is coarser than when translation was fixed")
    ap.add_argument("--fine-step", type=float, default=0.1,
                    help="degrees, final refinement")
    ap.add_argument("--trans-span", type=float, default=3000.0,
                    help="microns, half-width of the translation search. Must "
                         "exceed the largest plausible offset between the two "
                         "sections; an optimum reached at the bound means the "
                         "true one lies outside it")
    ap.add_argument("--xenium-h5ad", default=None,
                    help="prepared Xenium AnnData instead of the raw files; "
                         "{sample} is substituted. Skips loading and QC entirely, "
                         "which is the route for an imputed object. Must carry "
                         "obsm['spatial'] in microns.")
    ap.add_argument("--no-qc", action="store_true",
                    help="skip MAD outlier flagging on the raw path. Outliers are "
                         "never removed in any case, since deleting cells punches "
                         "holes in the density field the registration uses.")
    ap.add_argument("--keep-empty", action="store_true",
                    help="retain zero-count cells on the raw path; they give a "
                         "position but no signal, diluting the layer ordinal")
    ap.add_argument("--no-figures", dest="figures", action="store_false")
    ap.add_argument("--re-export", action="store_true", help="redo the Visium export")
    ap.add_argument("--refresh-manifest", action="store_true")
    ap.add_argument("--sigma-m", type=float, default=None,
                    help=f"matching variance, on images normalised to 0-1; "
                         f"default {LDDMM_SIGMA_M} follows the STalign "
                         f"package. Smaller values make the mixture behave "
                         f"as a hard classifier")
    ap.add_argument("--sigma-b", type=float, default=None,
                    help=f"background variance; default {LDDMM_SIGMA_B}")
    ap.add_argument("--sigma-a", type=float, default=None,
                    help=f"artifact variance; default {LDDMM_SIGMA_A}. The "
                         f"ratio to sigmaM governs how readily unmatched "
                         f"source tissue is set aside rather than forced "
                         f"onto the target")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    a = ap.parse_args()

    a.rds        = a.rds        or os.path.join(a.root, "fnl_spe_kept_spots_only.rds")
    a.xenium_dir = a.xenium_dir or os.path.join(a.root, "xenium")
    a.workdir    = a.workdir    or a.root
    a.outdir     = a.outdir     or os.path.join(a.root, "stalign_results")
    a.export_root = a.export_root or os.path.join(a.workdir, "visium_exports")
    os.makedirs(a.export_root, exist_ok=True)
    a.rscript    = a.rscript    or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "spe_export.R")
    os.makedirs(a.outdir, exist_ok=True)

    device = "cpu" if a.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    manifest = build_manifest(a)
    if a.list:
        print(manifest.to_string(index=False)); return

    if a.all:
        avail = {os.path.basename(p).split("_")[1].split("-")[0]
                 for p in glob.glob(f"{a.xenium_dir}/GSM*-cells.parquet*")}
        samples = sorted(d for d in manifest.donor.astype(str).unique()
                         if any(d in v or v in d for v in avail))
        log(f"{len(samples)} donors have both platforms: {samples}")
    elif a.sample:
        samples = a.sample
    else:
        ap.error("give --sample, --all or --list")

    if a.skip:
        dropped = [x for x in samples if x in a.skip]
        samples = [x for x in samples if x not in a.skip]
        if dropped: log(f"skipping {len(dropped)}: {dropped}")

    rows, failed = [], []
    for s in samples:
        try:
            rows.append(process_sample(s, manifest, a, device))
        except Exception as e:
            failed.append((s, str(e)))
            log(f"FAILED {s}: {e}")
            traceback.print_exc()

    if rows:
        flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
        summ = os.path.join(a.outdir, "cohort_summary.csv")
        pd.DataFrame(flat).to_csv(summ, index=False)
        log(f"\ncohort summary -> {summ}  ({len(rows)} donors)")
        cols = ["sample", "capture_area", "n_cells", "n_spots", "init_deg",
                "matched_frac", "layer_gm_wm_acc", "expr_cosine_separation"]
        print(pd.DataFrame(flat)[[c for c in cols if c in flat[0]]].to_string(index=False))
    if failed:
        log(f"\n{len(failed)} failed:")
        for s, e in failed: log(f"  {s}: {e}", 1)
    sys.exit(1 if failed and not rows else 0)


if __name__ == "__main__":
    main()
