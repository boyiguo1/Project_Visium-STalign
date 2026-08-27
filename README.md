# Project_Visium-STalign


Registers a Xenium section onto the Visium section from the same donor,
evaluates the result, and writes aligned coordinates plus diagnostic
figures. One donor per run, or the whole cohort in a batch.

```bash
python stalign_pipeline.py --list                 # what donors exist
python stalign_pipeline.py --sample Br8667        # one donor
python stalign_pipeline.py --all                  # every paired donor
```

Two files are needed and must sit in the same directory:
`stalign_pipeline.py` and `spe_export.R`.

---

## What it does

1. Reads the donor-to-capture-area table from the SPE object, once, and
   caches it.
2. Exports that capture area from the SPE to flat files, if not already
   done, by calling `spe_export.R`.
3. Downloads `scalefactors_json.json` for that capture area from the
   LieberInstitute repository.
4. Recovers the pixel-to-micron scale, converts Visium to microns, and
   checks the result against the chip geometry.
5. Finds the initial placement: reflection, rotation, and translation
   with the upper-left tissue boundaries brought into contact.
6. Runs LDDMM.
7. Evaluates against layer annotations and against shared-gene
   expression.
8. Writes coordinates, objects, metrics and figures.

Everything downstream of step 4 works in **microns on both platforms**,
so the initial affine carries rotation and translation only and its
determinant is exactly one.

---

## Command-line options

### Choosing what to run

| Option | Default | Meaning |
|---|---|---|
| `--sample NAME [NAME ...]` | — | Donor name(s), e.g. `Br8667`. Matched against the donor column of the SPE by substring, so `Br8667` finds `Br8667_NTC`. |
| `--all` | — | Every donor that has both a Xenium file and a Visium capture area. |
| `--list` | — | Print the donor table and exit. Run this first. |

A donor with more than one capture area produces a warning and the one
with the most spots is used.

### Paths

| Option | Default | Meaning |
|---|---|---|
| `--root` | `~/Guo_xenium` | Base for the four paths below. |
| `--rds` | `<root>/fnl_spe_kept_spots_only.rds` | The SpatialExperiment object holding all Visium capture areas. |
| `--xenium-dir` | `<root>/xenium` | Where the `GSM*_<donor>-cells.parquet` and `-cell_feature_matrix.h5` files live. |
| `--workdir` | `<root>` | Where exports, the manifest and downloaded files are cached. Reused across runs. |
| `--outdir` | `<root>/stalign_results` | Where results go, one subdirectory per donor. |
| `--rscript` | alongside the script | Path to `spe_export.R`. |

### Registration

| Option | Default | Meaning |
|---|---|---|
| `--niter` | `2000` | LDDMM iterations. The diffeomorphism engages after `min(300, niter/4)`; before that only the affine is optimised. Lower values are fine for a smoke test — on well-matched sections 500 gives nearly the same answer as 2000, but sections needing more warping do not converge that fast. |
| `--coarse-step` | `2.0` | Degrees, for the full-circle rotation sweep. Coarser is faster and risks missing a narrow optimum; on these data the optimum is several degrees wide, so 2° is ample. |
| `--fine-step` | `0.1` | Degrees, for the ±5° refinement around the coarse winner. Finer than the width of the optimum buys nothing, which is why the reported angle carries that width rather than the sub-degree fit. |
| `--cpu` | — | Force CPU even if a GPU is visible. |

### Xenium input

| Option | Default | Meaning |
|---|---|---|
| `--xenium-h5ad PATH` | — | Load a prepared AnnData instead of the raw parquet and h5. `{sample}` in the path is substituted. Skips loading and all QC, which is the route for an imputed object. Must carry `obsm['spatial']` in microns. |
| `--no-qc` | — | Skip outlier flagging on the raw path. Outliers are flagged but never removed in any case, since deleting cells punches holes in the density field the registration uses. |
| `--keep-empty` | — | Keep zero-count cells on the raw path. They contribute a position but no signal, so they dilute the rasterised layer ordinal. |

**If the prepared object was imputed from Visium**, the shared-gene
evaluation is no longer independent of the registration — it compares
values derived from Visium against Visium itself. The script prints a
note when the gene count exceeds a Xenium panel. Read that evaluation as
a consistency check, not a validation.

### Housekeeping

| Option | Default | Meaning |
|---|---|---|
| `--no-figures` | figures on | Numbers only. Roughly halves the runtime. |
| `--re-export` | — | Redo the Visium export even if the flat files exist. |
| `--refresh-manifest` | — | Re-read the donor table from the SPE object. |

---

## Fixed parameters

These are constants at the top of the script rather than options,
because changing them changes what the numbers mean.

### Physical scale

| Name | Value | Why |
|---|---|---|
| `SPOT_PITCH` | `100.0` µm | Visium spot centre-to-centre distance, a manufacturing constant. This is the ruler: the pixel scale is recovered as `SPOT_PITCH / median nearest-neighbour distance`, per capture area. |

The pixel scale is **not** taken from `spot_diameter_fullres`, which
Space Ranger documents as an estimate for visualisation spanning 60–70 µm
across slide designs, with the explicit advice not to infer pixel size
from it. That field is read, but only for the in-spot radius, where it is
used in pixel space.

The recovered scale is checked against the chip geometry — hexagonal
spacings of 50.000 µm per array column and 86.603 µm per row predict the
tissue extent without reference to the registration. Agreement is printed
as `array-geometry check`; both axes should be within about 1%.

### Rasterisation

| Name | Value | Why |
|---|---|---|
| `DX_RASTER` | `50.0` µm | Grid step, the same for source and target so a pixel denotes the same physical size in both images. |
| `BLUR` | `2.0` | Gaussian blur in units of `DX_RASTER`, so 100 µm. Raise this first if the energy curves oscillate; that is the STalign authors' documented remedy, ahead of tuning `a` or `epV`. |

### Placement search

| Name | Value | Why |
|---|---|---|
| `KERNEL_SIGMA` | `150.0` µm | Width of the Gaussian used to score a candidate placement. Each target spot takes a weighted mean of the source layer ordinal nearby and is correlated with its own domain. Unlike binning, this varies continuously with the transform, so sub-degree differences are resolvable. |
| `KERNEL_MINSUP` | `8` | Minimum source cells within 3σ for a spot to contribute. |
| `U_EDGE` | `(-1,-1)` | The direction taken as upper-left. The boundary along it is treated as the pial surface and matched between sections. |

### LDDMM

| Name | Value | Why |
|---|---|---|
| `LDDMM_A_UM` | `300.0` µm | Smoothing kernel width, roughly one cortical layer, so the deformation is smooth within a layer and free to differ between layers. |
| `sigmaM`, `sigmaB` | `0.3` | Matching and background variances. |
| `sigmaA` | `1.0` or `0.3` | Artifact variance, chosen by the tissue-area ratio: `1.0` when the source exceeds the target by more than about 15%, otherwise `0.3`. See the note on saturation below. |
| `sigmaR` | `1e5` | Regularisation weight. |
| `epL`, `epT`, `epV` | `1e-8`, `0.2`, `30` | Gradient-descent step sizes for the linear part, the translation and the velocity field. |
| `MATCH_THRESH` | `0.85` | Matching-probability cutoff, following Clifton et al. |

### Evaluation

| Name | Value | Why |
|---|---|---|
| Moran's I FDR | `0.05` | Benjamini–Hochberg, on Visium counts per million, over a Delaunay graph filtered at 1.5 spot pitches. Reimplements MERINGUE in Python; verified on a synthetic hexagonal lattice where a linear gradient gives I = 0.977 and white noise I = 0.009. |
| Pseudospot radius | `spot_diameter_fullres / 2` | In pixels, converted to microns once. Avoids having to choose between the 55 µm and 65 µm spot diameters that appear in different 10x documents. |

---

## Two things to know before using the output

### `matched` is not always a safe filter — use `usable`

The matching probability comes from a mixture model whose artifact
component only has work to do when the source section exceeds the target.
Where the two have comparable area, the artifact component has nothing to
explain, the probability saturates at one for every cell, and the 0.85
threshold admits everything — including cells the transform placed
outside the target.

Observed across two donors:

| | tissue-area ratio | σ_A | matched | 1st pct of match_prob |
|---|---|---|---|---|
| source much larger | 0.67 | 1.0 | 70.2% | discriminating |
| comparable areas | 0.97 | 0.3 | 100.0% | 0.99997, saturated |

Lowering σ_A from 1.0 to 0.3 did not restore discrimination, confirming
that this is geometric rather than a parameter setting.

The exported CSV therefore carries three columns:

- `matched` — `match_prob > 0.85`, which may be true for every cell
- `inside_visium` — within one spot pitch of a Visium spot centre
- **`usable` — both. Filter on this.**

The script prints a note when saturation is detected and records
`match_prob_saturated` in the metrics.

### The reported rotation carries a width

The rotation optimum is flat over several degrees — 6.6° and 4.9° in the
two donors measured. This follows from the layer ordinal being equally
spaced: it produces a monotonic ramp from pia to white matter, which
fixes position along the depth axis but is nearly invariant to small
rotations about it. `init_deg_lo` and `init_deg_hi` bound the region
within 0.005 of the optimum. Report the angle with that width; the
sub-degree figure from the parabolic fit is below the resolving power of
the data.

---

## Outputs

Per donor, in `<outdir>/<donor>_STalign_result/`:

| File | Contents |
|---|---|
| `<donor>_aligned_coordinates.csv` | One row per cell: native and aligned x/y in microns, the same in Visium pixels, match probability, the three filter columns, nearest spot, layer, cell type. |
| `<donor>_aligned_coordinates.README` | Units, coordinate frames, and how to filter. Travels with the CSV. |
| `<donor>_visium_coordinates.csv` | Spot barcode, microns, pixels, domain. |
| `<donor>_xenium_aligned.h5ad` | Full object; `uns['alignment']` holds every parameter and metric. |
| `<donor>_visium.h5ad` | |
| `<donor>_metrics.json` | Everything the run produced, including the per-layer breakdown. |

And across donors, `<outdir>/cohort_summary.csv` — one row each, written
at the end of a run. **A single-donor run overwrites it**, so use
`--sample A B` or `--all` when a comparison table is wanted.

### Figures

| File | Shows |
|---|---|
| `fig_initial_placement` | Rotation sweeps, coarse and fine, and the resulting placement with both matched boundaries marked. |
| `fig_registration` | Match probability, aligned cells over Visium domains, Jacobian determinant. |
| `fig_panelD_domains` | Both sections' domains side by side in shared coordinates and one colour scale, so a domain at the same figure position is at the same tissue position. |
| `fig_panelD_overlap` | Where the geometric and probabilistic overlap criteria disagree, and each section restricted to the overlap. |
| `fig_layer_concordance` | Confusion matrix and the per-layer offset. |
| `fig_layer_landing` | One panel per domain: aligned cells over spots of the same domain. |
| `fig_expression_concordance` | Cosine by stratum, and Spearman against sparsity. |
| `fig_crosstech` | Replica of the cross-technology figure in Clifton et al.: three patterned genes spanning the cosine range plus one unpatterned control. |

All tissue panels carry a 1 mm scale bar and a black frame.

---

## Reading the metrics

| Field | Healthy value | Meaning |
|---|---|---|
| `scale_um_per_px` | 0.49–0.50 here | Recovered pixel scale. Varies between capture areas, which is why it is recovered per section. |
| `scale_array_dev_x/y` | within ±1% | Chip-geometry check on that scale. A larger value means the scale is wrong, not that the registration is. |
| `label_join_rho` | > 0.8 | Lieber annotations against an independent marker axis. Low means the annotation join failed, e.g. a key-prefix mismatch. |
| `init_deg_hi − init_deg_lo` | a few degrees | Width of the rotation optimum. |
| `fitted_affine_det` | near 1 | Rescaling LDDMM introduced on its own; the initial affine is exactly 1. Far from 1 would also flag an error in the pixel scale. |
| `folded_frac` | 0 | Non-zero means the transform is not a diffeomorphism. |
| `jacobian_p5`, `p95` | within ±25% of 1 | Local area change. A narrow spread means the deformation is nearly rigid. |
| `layer_gm_wm_acc` | — | Fraction landing on the correct side of the grey–white boundary. |
| `layer_within_gm_rho` | — | Laminar ordering within grey matter. |
| `expr_cosine_separation` | — | Patterned minus unpatterned cosine. Clifton et al. report 0.49 for MERFISH-to-Visium in mouse; tissue, species and panel size differ here, so treat it as a reference point rather than a target. |

### The two evaluations are not equivalent

`layer_*` uses the same annotations that drive the registration, so it
**bounds alignment quality from above** rather than testing it. Clifton
et al. raise this explicitly for alignment driven by expression or
cell-type features.

`expr_*` uses shared-gene expression, which enters the registration at no
point. That is the independent test.

### A known systematic offset

Per-layer bias decreases monotonically from pia to white matter in both
donors measured: superficial cells land deeper than their own domain,
deep cells land shallower. The layers are compressed toward the middle.
This is consistent with the equally spaced ordinal, which does not encode
the unequal physical thickness of the layers, so the registration aligns
two ramps by length rather than by layer boundary. It shows up as a high
`layer_within_gm_rho` alongside a low `layer_exact`: the ordering is
recovered, the exact position is not.

---

## Requirements

Python: `numpy pandas scipy matplotlib scanpy anndata torch STalign`.
R: `SpatialExperiment SingleCellExperiment Matrix`, with `Rscript` on the
path — activate the environment that has them before running.

Runtime per donor on CPU: roughly 4 minutes for the rotation search and
2–3 minutes for LDDMM at 2000 iterations, plus a minute for the Visium
export the first time. A failing donor is logged and the batch continues.

## Troubleshooting

**`no Xenium cells file for <donor>`** — the donor name must appear in
the filename as `GSM*_<donor>-cells.parquet`.

**`<donor> not found among the Visium donors`** — check `--list` for how
the name is written in the SPE object.

**`label_join_rho` is NaN, coverage 0%** — the Lieber annotation keys are
`<donor>_<row>`, built from `orig_row`. A different prefix in the
published tables breaks the join.

**R errors mentioning a JHPCE path** — the stored image paths in the SPE
do not resolve off-cluster. `spe_export.R` drops the image slot on load;
if you call the object from your own code, do the same first.

**Warnings from `torch` and `STalign`** — `torch.tensor(sourceTensor)`,
`meshgrid indexing`, and `cannot be log-scaled` all come from inside
those libraries and are harmless.
