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
#    <donor>_xenium_aligned.h5ad         full object with uns['alignment']
#    <donor>_visium.h5ad
#    <donor>_metrics.json                every number this run produced
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

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600,
                     "axes.grid": False, "font.size": 10})


# ============================================================================
#  small shared helpers
# ============================================================================
def log(msg, indent=0):
    print(("  " * indent) + msg, flush=True)


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


def savefig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


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
    if not os.path.exists(f"{stem}-cell_feature_matrix.h5"):
        raise FileNotFoundError(f"missing {stem}-cell_feature_matrix.h5")

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
    """Flat files from the SPE, written once per donor."""
    vdir = os.path.join(args.workdir, f"{info['sample']}_export")
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


def initial_affine(xI, yI, layer_axis, lay_x_true, ref, lab, args, outdir):
    """
    Mirror by binned correlation, then rotation and translation jointly
    under the constraint that the upper-left boundary of each section is
    the pial surface. The boundary is re-extracted at every angle, since
    rotation changes which cells face that direction.
    """
    src = np.c_[xI, yI]

    # --- depth axis, for the starting angle and the mirror test ---
    def depth_dir(x, y, s):
        P = np.c_[x - x.mean(), y - y.mean()]
        g = np.linalg.lstsq(P, s - s.mean(), rcond=None)[0]
        return g / np.linalg.norm(g)
    dI = depth_dir(xI, yI, layer_axis)
    dJ = depth_dir(ref[:,0], ref[:,1], lab)
    th_deg = float(np.degrees(np.arctan2(dJ[1], dJ[0]) - np.arctan2(dI[1], dI[0])))

    mirror_rho = {}
    for m in (False, True):
        M = rot2(th_deg)
        if m:
            n = np.array([-dI[1], dI[0]])
            M = M @ (np.eye(2) - 2*np.outer(n, n))
        t = ref.mean(0) - (src @ M.T).mean(0)
        mirror_rho[m], _ = binned_corr(src @ M.T + t, layer_axis, ref, lab)
    mirror = max(mirror_rho, key=lambda m: mirror_rho[m])
    log(f"mirror={mirror} (binned rho {mirror_rho[False]:.3f} vs "
        f"{mirror_rho[True]:.3f})", 1)

    mv_edge = edge_mask(ref)
    eV_c    = ref[mv_edge].mean(0)

    def apply_mirror(M):
        if not mirror: return M
        n = np.array([-dI[1], dI[0]])
        return M @ (np.eye(2) - 2*np.outer(n, n))

    def butt(M, pts):
        proj = pts @ (M.T @ U_EDGE)
        m = proj >= np.quantile(proj, 0.97)
        return eV_c - pts[m].mean(0) @ M.T, m

    rng = np.random.default_rng(0)
    sub = rng.choice(len(src), size=min(30000, len(src)), replace=False)
    score_sub  = make_kernel_scorer(src[sub], layer_axis[sub], ref, lab)
    score_full = make_kernel_scorer(src,      layer_axis,      ref, lab)

    t0 = time.time()
    coarse = []
    for d in np.arange(0, 360, args.coarse_step):
        M = apply_mirror(rot2(d)); t, _ = butt(M, src[sub])
        r, _ = score_sub(M, t)
        coarse.append((d, r))
    C = pd.DataFrame(coarse, columns=["deg", "rho"]).dropna()
    d0 = float(C.loc[C.rho.idxmax(), "deg"])

    fine = []
    for d in np.arange(d0 - 5, d0 + 5.001, args.fine_step):
        M = apply_mirror(rot2(d)); t, _ = butt(M, src)
        r, _ = score_full(M, t)
        fine.append((d, r))
    F = pd.DataFrame(fine, columns=["deg", "rho"]).dropna()
    k = int(F.rho.values.argmax())
    if 0 < k < len(F) - 1:
        y0, y1, y2 = F.rho.values[k-1:k+2]
        den = y0 - 2*y1 + y2
        deg_opt = float(F.deg.values[k] + (0.5*(y0-y2)/den if den else 0.0)*args.fine_step)
    else:
        deg_opt = float(F.deg.values[k])

    M_opt = apply_mirror(rot2(deg_opt))
    t_opt, me_opt = butt(M_opt, src)
    rho_opt, n_sc = score_full(M_opt, t_opt)

    # the width of the optimum is a property of the problem, not noise:
    # an equally spaced ordinal is a ramp, which fixes depth but is nearly
    # invariant to small rotations about the depth axis
    half = F[F.rho >= rho_opt - 0.005]
    ang_lo, ang_hi = float(half.deg.min()), float(half.deg.max())

    off = np.arange(-800, 801, 25.0)
    o_tan = off[np.nanargmax([score_full(M_opt, t_opt + o*V_TAN)[0] for o in off])]
    o_per = off[np.nanargmax([score_full(M_opt, t_opt + o_tan*V_TAN + o*U_EDGE)[0]
                              for o in off])]
    t_fin = t_opt + o_tan*V_TAN + o_per*U_EDGE
    rho_fin, _ = score_full(M_opt, t_fin)
    log(f"angle {deg_opt:.1f} deg (flat over {ang_lo:.1f}-{ang_hi:.1f}), "
        f"kernel rho {rho_fin:.4f}, {time.time()-t0:.0f} s", 1)

    if args.figures:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        ax[0].plot(C.deg, C.rho, lw=1.3, c="#0072B2")
        ax[0].axvline(deg_opt, c="#7A0177", lw=1.4)
        ax[0].set_xlabel("rotation (deg)"); ax[0].set_ylabel("kernel rho")
        ax[0].set_title(f"a   coarse sweep, {args.coarse_step:.0f} deg",
                        loc="left", fontweight="bold", fontsize=10.5)
        ax[1].plot(F.deg, F.rho, lw=1.4, c="#0072B2")
        ax[1].axvspan(ang_lo, ang_hi, color="#7A0177", alpha=.12)
        ax[1].axvline(deg_opt, c="#7A0177", lw=1.4)
        ax[1].set_xlabel("rotation (deg)"); ax[1].set_ylabel("kernel rho")
        ax[1].set_title(f"b   fine sweep, flat over {ang_hi-ang_lo:.1f} deg",
                        loc="left", fontweight="bold", fontsize=10.5)
        out0 = src @ M_opt.T + t_fin
        ax[2].scatter(ref[:,0]/1000, ref[:,1]/1000, s=20, c=lab-1,
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.32, linewidths=0)
        ax[2].scatter(out0[:,0]/1000, out0[:,1]/1000, s=.8, c=layer_axis-1,
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.55, linewidths=0)
        ax[2].scatter(out0[me_opt,0]/1000, out0[me_opt,1]/1000, s=2, c="#141413",
                      linewidths=0)
        ax[2].scatter(ref[mv_edge,0]/1000, ref[mv_edge,1]/1000, s=34,
                      facecolors="none", edgecolors="#141413", linewidths=.8)
        ax[2].set_aspect("equal"); ax[2].invert_yaxis()
        add_scalebar(ax[2], 1000.0, "1 mm")
        ax[2].set_title(f"c   edges butted, {deg_opt:.1f} deg",
                        loc="left", fontweight="bold", fontsize=10.5)
        fig.suptitle("Initial placement: mirror, rotation and edge contact",
                     fontsize=12.5, fontweight="bold", y=1.0)
        savefig(fig, outdir, "fig_initial_placement")

    return dict(A=affine3(M_opt, t_fin), M=M_opt, t=t_fin,
                out=src @ M_opt.T + t_fin, mirror=bool(mirror),
                deg=deg_opt, deg_lo=ang_lo, deg_hi=ang_hi,
                deg_depth_axis=th_deg, rho_kernel=float(rho_fin),
                rho_mirror_false=float(mirror_rho[False]),
                rho_mirror_true=float(mirror_rho[True]),
                tangential_um=float(o_tan), perpendicular_um=float(o_per),
                edge_l1_frac=float((lay_x_true[me_opt] == 0).mean()),
                n_spots_scored=int(n_sc),
                score_full=score_full)


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
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=DOM_COLORS[i], label=l)
               for i, l in enumerate(LAY)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.4))
    ax[0].scatter(srcD[:,0]/1000, srcD[:,1]/1000, s=1.2, c=layer_axis-1,
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    ax[0].set_title(f"D1   Xenium, spatial domain\naligned, {A.n_obs} cells",
                    loc="left", fontweight="bold", fontsize=11)
    ax[1].scatter(ref[:,0]/1000, ref[:,1]/1000, s=26, c=lay_v,
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    ax[1].set_title(f"D2   Visium, spatial domain\n{V.n_obs} spots",
                    loc="left", fontweight="bold", fontsize=11)
    for a in ax:
        a.set_xlim(*XL); a.set_ylim(*YL); a.set_aspect("equal")
        add_scalebar(a, 1000.0, "1 mm")
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Spatial domain annotation of both sections, shared coordinates",
                 fontsize=13, fontweight="bold", y=0.99)
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

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    ax[0].scatter(ref[:,0]/1000, ref[:,1]/1000, s=16, c="#dddddd", linewidths=0)
    for m, c, l in ((both, "#009E73", f"both ({both.sum()})"),
                    (geo_only, "#0072B2", f"geometric only ({geo_only.sum()})"),
                    (mat_only, "#D55E00", f"matched only ({mat_only.sum()})")):
        ax[0].scatter(srcD[m,0]/1000, srcD[m,1]/1000, s=.8, c=c,
                      linewidths=0, label=l)
    ax[0].legend(fontsize=8, markerscale=6)
    ax[0].set_title("a   where the two overlap criteria differ",
                    loc="left", fontweight="bold", fontsize=11)
    ax[1].scatter(srcD[both,0]/1000, srcD[both,1]/1000, s=1.4,
                  c=(layer_axis-1)[both], cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    ax[1].set_title(f"b   Xenium over the overlap ({both.sum()} cells)",
                    loc="left", fontweight="bold", fontsize=11)
    ax[2].scatter(ref[hit,0]/1000, ref[hit,1]/1000, s=28, c=lay_v[hit],
                  cmap=CMAP_DOM, norm=NORM_DOM, linewidths=0)
    ax[2].set_title(f"c   Visium over the overlap ({hit.sum()} spots)",
                    loc="left", fontweight="bold", fontsize=11)
    for a in ax:
        a.set_xlim(*XL2); a.set_ylim(*YL2); a.set_aspect("equal")
        add_scalebar(a, 1000.0, "1 mm")
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("The overlapping region, and each section restricted to it",
                 fontsize=13, fontweight="bold", y=0.99)
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
        a.set_title(f"{l}   n={m.sum()}   exact {hit:.0f}%", fontsize=9)
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
    fig.suptitle("Where each Xenium domain lands on the Visium section",
                 fontsize=12.5, fontweight="bold", y=1.0)
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
        if i == 0: a.set_title("a   Visium", fontsize=11, loc="left",
                               fontweight="bold")

        a = fig.add_subplot(GS[i, 1])
        a.scatter(xt[keep]/1000, yt[keep]/1000, c=n01(Xcpm[keep, jx]),
                  cmap=RED, s=.5, linewidths=0)
        a.set_aspect("equal"); a.invert_yaxis(); add_scalebar(a, 1000.0, "1 mm")
        if i == 0: a.set_title("b   Xenium aligned", fontsize=11, loc="left",
                               fontweight="bold")

        a = fig.add_subplot(GS[i, 2])
        a.scatter(Pc[:, jg], Vc_[:, jg], s=9, c="#444444", alpha=.45, linewidths=0)
        a.set_xlabel("Xenium pseudospot (CPM)", fontsize=9)
        a.set_ylabel("Visium spot (CPM)", fontsize=9)
        a.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        a.text(.04, .93, f"cosine {r_.cosine:.2f}\nSpearman {r_.spearman:+.2f}\n"
                         f"nonzero {min(r_.nz_X, r_.nz_V)*100:.0f}%",
               transform=a.transAxes, fontsize=8.5, va="top",
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", lw=.5))
        if i == 0: a.set_title("c   Normalized expression\n     correspondence",
                               fontsize=10, loc="left", fontweight="bold")

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
        axd.set_ylim(0, 1); axd.set_ylabel("cosine similarity")
        axd.set_xlabel("gene count")
        axd.set_title(f"{'d' if k_ == 0 else 'e'}   {lbl} (n={msk.sum()})",
                      fontweight="bold", loc="left", fontsize=9.5)

    cax = fig.add_axes([0.13, 0.028, 0.20, 0.010])
    plt.colorbar(plt.cm.ScalarMappable(cmap=RED), cax=cax, orientation="horizontal")
    cax.set_xticks([0, 1]); cax.set_xticklabels(["0", "high"])
    cax.set_title("normalized expression (1st-99th pct)", fontsize=8)
    fig.suptitle("Cross-technology expression correspondence (Xenium to Visium)",
                 fontsize=13, fontweight="bold", y=1.0)
    savefig(fig, outdir, "fig_crosstech")
    return G


# ============================================================================
#  registration
# ============================================================================
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

    # sigmaA governs how readily source tissue is called artifact, and it
    # only bites when the source exceeds the target. Clifton et al. raise it
    # for partially matched sections for exactly that reason. Where the two
    # areas are comparable the artifact component has nothing to explain,
    # W_M saturates at 1, and match_prob stops discriminating, so the 0.85
    # threshold silently admits every cell -- including cells transformed
    # outside the target. The value therefore tracks the area ratio.
    sigma_a = 1.0 if ceiling < 0.85 else 0.3
    log(f"sigmaA {sigma_a} (tissue-area ceiling {ceiling*100:.0f}%)", 1)

    kw = dict(niter=args.niter, diffeo_start=min(300, args.niter//4),
              a=LDDMM_A_UM, sigmaM=0.3, sigmaB=0.3, sigmaA=sigma_a, sigmaR=1e5,
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

    keep = wm > MATCH_THRESH
    wm_p1 = float(np.percentile(wm, 1))
    saturated = wm_p1 > 0.99
    log(f"LDDMM {runtime:.0f} s | matched {keep.mean()*100:.1f}% "
        f"(area ceiling {ceiling*100:.1f}%) | det {det_A:.4f} | "
        f"folding {(Jv<=0).mean()*100:.2f}%", 1)
    if saturated:
        log(f"NOTE match_prob saturated (1st pct {wm_p1:.3f}); the mixture "
            f"model did not discriminate, so use the geometric criterion "
            f"downstream rather than the threshold", 1)

    if args.figures:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
        s_ = ax[0].scatter(xt/1000, yt/1000, s=.6, c=wm, cmap="RdYlGn",
                           vmin=0, vmax=1, linewidths=0)
        ax[0].set_title(f"a   match probability ({keep.mean()*100:.0f}% > "
                        f"{MATCH_THRESH})", loc="left", fontweight="bold", fontsize=10.5)
        plt.colorbar(s_, ax=ax[0], fraction=.046)
        ax[1].scatter(ref[:,0]/1000, ref[:,1]/1000, s=9, c=lab-1, cmap=CMAP_DOM,
                      norm=NORM_DOM, alpha=.35, linewidths=0)
        ax[1].scatter(xt[keep]/1000, yt[keep]/1000, s=.6, c=(layer_axis-1)[keep],
                      cmap=CMAP_DOM, norm=NORM_DOM, alpha=.55, linewidths=0)
        ax[1].set_title("b   aligned Xenium on Visium domains",
                        loc="left", fontweight="bold", fontsize=10.5)
        im = ax[2].imshow(np.where(occ, Jdet, np.nan), cmap="coolwarm",
                          origin="lower", vmin=0.6, vmax=1.4,
                          extent=[gx.min()/1000, gx.max()/1000,
                                  gy.min()/1000, gy.max()/1000])
        ax[2].invert_yaxis(); ax[2].set_aspect("equal")
        plt.colorbar(im, ax=ax[2], fraction=.046)
        ax[2].set_title(f"c   Jacobian det (median {np.median(Jv):.3f})",
                        loc="left", fontweight="bold", fontsize=10.5)
        for a in (ax[0], ax[1], ax[2]):
            a.set_aspect("equal")
            if a is not ax[2]: a.invert_yaxis()
            add_scalebar(a, 1000.0, "1 mm")
        fig.suptitle("Registration result and deformation",
                     fontsize=12.5, fontweight="bold", y=1.0)
        savefig(fig, outdir, "fig_registration")

    return dict(x=xt, y=yt, wm=wm, keep=keep, runtime=runtime,
                det_A=det_A, ceiling=float(ceiling), sigma_a=float(sigma_a),
                wm_p1=wm_p1, wm_saturated=bool(saturated),
                jac_med=float(np.median(Jv)), jac_p5=float(np.percentile(Jv,5)),
                jac_p95=float(np.percentile(Jv,95)),
                folded=float((Jv <= 0).mean()))


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
        f"GM/WM {gmwm*100:.1f}%, within-GM rho {rho_gm:+.3f}", 1)

    if args.figures:
        C = np.zeros((7, 7)); np.add.at(C, (xo, vo), 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
        im = ax[0].imshow(C / C.sum(1, keepdims=True).clip(1), cmap="magma",
                          vmin=0, vmax=1)
        ax[0].set_xticks(range(7)); ax[0].set_xticklabels(LAY, rotation=45, fontsize=7)
        ax[0].set_yticks(range(7)); ax[0].set_yticklabels(LAY, fontsize=7)
        ax[0].set_xlabel("landed Visium domain"); ax[0].set_ylabel("Xenium domain")
        ax[0].plot([-.5,6.5], [-.5,6.5], c="#00E5FF", lw=.7, alpha=.6)
        ax[0].axvline(4.5, c="#00E5FF", lw=.7, ls="--", alpha=.5)
        ax[0].axhline(4.5, c="#00E5FF", lw=.7, ls="--", alpha=.5)
        for sp in ax[0].spines.values():
            sp.set_visible(True); sp.set_color("#141413"); sp.set_linewidth(0.8)
        plt.colorbar(im, ax=ax[0], fraction=.046)
        ax[0].set_title(f"a   recall (exact {same*100:.0f}%, +/-1 {adj*100:.0f}%)",
                        loc="left", fontweight="bold", fontsize=10.5)
        mu = np.array([vo[xo == i].mean() if (xo == i).sum() > 20 else np.nan
                       for i in range(7)])
        sd = np.array([vo[xo == i].std() if (xo == i).sum() > 20 else np.nan
                       for i in range(7)])
        ax[1].plot([0,6], [0,6], c="#bbb", ls="--", lw=1)
        ax[1].errorbar(range(7), mu, yerr=sd, fmt="-o", ms=5, c="#0072B2", capsize=3)
        ax[1].axvspan(0.5, 4.5, color="#009E73", alpha=.07)
        ax[1].set_xticks(range(7)); ax[1].set_xticklabels(LAY, rotation=45, fontsize=7)
        ax[1].set_ylabel("mean landed domain")
        for sp in ax[1].spines.values():
            sp.set_visible(True); sp.set_color("#141413"); sp.set_linewidth(0.8)
        ax[1].set_title("b   systematic offset", loc="left",
                        fontweight="bold", fontsize=10.5)
        fig.suptitle("Laminar concordance (bounds quality from above)",
                     fontsize=12.5, fontweight="bold", y=1.0)
        savefig(fig, outdir, "fig_layer_concordance")

    return dict(n=int(sel.sum()), exact=same, within1=adj, spearman=rho,
                sigma=lift, gm_wm_acc=gmwm, within_gm_rho=rho_gm, per_layer=per)


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
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        b_ = np.linspace(0, 1, 26)
        ax[0].hist(cs[pat][~np.isnan(cs[pat])], bins=b_, color="#0072B2", alpha=.7,
                   label=f"patterned (n={pat.sum()})")
        ax[0].hist(cs[~pat][~np.isnan(cs[~pat])], bins=b_, color="#BBBBBB", alpha=.7,
                   label=f"not patterned (n={(~pat).sum()})")
        ax[0].axvline(0.55, c="#D55E00", ls="--", lw=1.2, label="Clifton 0.55")
        ax[0].axvline(0.06, c="#999999", ls="--", lw=1.2, label="Clifton 0.06")
        ax[0].set_xlabel("cosine similarity"); ax[0].set_ylabel("genes")
        ax[0].legend(fontsize=7.5)
        ax[0].set_title("a   cosine by stratum", loc="left",
                        fontweight="bold", fontsize=10.5)
        ax[1].scatter(np.minimum(nzX, nzV)*100, ss, s=11,
                      c=np.where(pat, "#0072B2", "#BBBBBB"), linewidths=0)
        ax[1].axvline(60, c="#D55E00", ls="--", lw=1.2)
        ax[1].axhline(0, c="#ccc", lw=.6)
        ax[1].set_xlabel("nonzero fraction, sparser platform (%)")
        ax[1].set_ylabel("Spearman")
        ax[1].set_title("b   sparsity caps Spearman", loc="left",
                        fontweight="bold", fontsize=10.5)
        for a_ in ax:
            for sp_ in a_.spines.values():
                sp_.set_visible(True); sp_.set_color("#141413"); sp_.set_linewidth(0.8)
        fig.suptitle("Shared-gene expression concordance (orthogonal)",
                     fontsize=12.5, fontweight="bold", y=1.0)
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

    A.obs["visium_spot"] = A.obs["visium_spot"].astype(object)
    for c in A.obs.columns:
        if A.obs[c].dtype == object:
            A.obs[c] = A.obs[c].where(A.obs[c].notna(), None)
    A.write(os.path.join(outdir, f"{s}_xenium_aligned.h5ad"))
    V.write(os.path.join(outdir, f"{s}_visium.h5ad"))
    json.dump(metrics, open(os.path.join(outdir, f"{s}_metrics.json"), "w"),
              indent=2, default=float)
    log(f"wrote coordinates for {len(exp)} cells "
        f"({int(exp.matched.sum())} matched, {int(exp.usable.sum())} usable) "
        f"-> {outdir}", 1)
    return exp


# ============================================================================
#  per-sample driver
# ============================================================================
def process_sample(sample, manifest, args, device):
    t_all = time.time()
    log(f"\n{'='*70}\n  {sample}\n{'='*70}")
    info = resolve_sample(sample, manifest, args)
    log(f"capture area {info['capture']} | dx {info['dx']} | "
        f"{info['n_spots_expected']} spots expected")

    outdir = os.path.join(args.outdir, f"{sample}_STalign_result")
    os.makedirs(outdir, exist_ok=True)

    vdir  = ensure_visium_export(info, args)
    srdir = ensure_spaceranger(info["capture"], args)

    A, n_empty = load_xenium(info, args)
    V, scale   = load_visium(info, vdir, srdir)
    marker_axis, cov, join_rho = attach_labels(A, sample, os.path.join(args.workdir,
                                                                      "lieber_cache"))
    layer_axis, lay_x_true, lab, bad_spd = build_ordinals(A, V, marker_axis)
    if bad_spd.any():
        V = V[~bad_spd].copy(); lab = lab[~bad_spd]
    ref = V.obsm["spatial"]
    xI, yI = A.obsm["spatial"].T

    init = initial_affine(xI, yI, layer_axis, lay_x_true, ref, lab, args, outdir)
    reg  = run_lddmm(xI, yI, layer_axis, ref, lab, init, args, device, outdir)

    rho_pre,  _ = binned_corr(init["out"], layer_axis, ref, lab)
    rho_post, _ = binned_corr(np.c_[reg["x"], reg["y"]], layer_axis, ref, lab)
    lay_eval = eval_layers(reg["x"], reg["y"], reg["wm"], lay_x_true, ref, lab,
                           scale, outdir, args)
    ov = {}
    if args.figures:
        ov = fig_panel_d(A, V, reg["x"], reg["y"], reg["wm"], layer_axis,
                         lay_x_true, lab, scale, outdir) or {}
        fig_landing(reg["x"], reg["y"], reg["wm"], lay_x_true, ref, lab,
                    scale, outdir)
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
        init_mirror=init["mirror"], init_deg=init["deg"],
        init_deg_lo=init["deg_lo"], init_deg_hi=init["deg_hi"],
        init_deg_depth_axis=init["deg_depth_axis"],
        init_rho_kernel=init["rho_kernel"],
        init_tangential_um=init["tangential_um"],
        init_perpendicular_um=init["perpendicular_um"],
        init_edge_l1_frac=init["edge_l1_frac"],
        lddmm_niter=args.niter, lddmm_a_um=LDDMM_A_UM,
        lddmm_runtime_s=reg["runtime"], device=str(device),
        matched_frac=float(reg["keep"].mean()), area_ceiling=reg["ceiling"],
        lddmm_sigmaA=reg["sigma_a"], match_prob_p1=reg["wm_p1"],
        match_prob_saturated=reg["wm_saturated"],
        fitted_affine_det=reg["det_A"], jacobian_median=reg["jac_med"],
        jacobian_p5=reg["jac_p5"], jacobian_p95=reg["jac_p95"],
        folded_frac=reg["folded"],
        rho_binned_pre=float(rho_pre), rho_binned_post=float(rho_post),
        **{f"layer_{k}": v for k, v in lay_eval.items() if k != "per_layer"},
        layer_per_layer=lay_eval.get("per_layer", {}),
        **{f"expr_{k}": v for k, v in exp_eval.items() if not isinstance(v, list)},
        **{f"overlap_{k}" if not k.startswith("overlap") else k: v
           for k, v in ov.items()},
        total_runtime_s=time.time() - t_all)

    exp = write_outputs(info, A, V, reg, init, scale, lay_x_true, lab,
                        outdir, metrics)
    metrics["n_usable"] = int(exp.usable.sum())
    metrics["n_inside_visium"] = int(exp.inside_visium.sum())
    json.dump(metrics, open(os.path.join(outdir, f"{sample}_metrics.json"), "w"),
              indent=2, default=float)
    log(f"done in {metrics['total_runtime_s']:.0f} s")
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
    ap.add_argument("--rscript", default=None, help="default alongside this script")
    ap.add_argument("--niter", type=int, default=2000)
    ap.add_argument("--coarse-step", type=float, default=2.0)
    ap.add_argument("--fine-step", type=float, default=0.1)
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
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    a = ap.parse_args()

    a.rds        = a.rds        or os.path.join(a.root, "fnl_spe_kept_spots_only.rds")
    a.xenium_dir = a.xenium_dir or os.path.join(a.root, "xenium")
    a.workdir    = a.workdir    or a.root
    a.outdir     = a.outdir     or os.path.join(a.root, "stalign_results")
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
