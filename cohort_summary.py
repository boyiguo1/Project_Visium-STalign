#!/usr/bin/env python3
"""
Rebuild the cohort summary from every <donor>_metrics.json on disk.

The pipeline writes cohort_summary.csv at the end of a run, covering only
the donors that run touched, so a single-donor invocation overwrites the
table built by a batch. This reads the per-donor files instead, which are
never overwritten, and so always reflects everything completed so far.

    python cohort_summary.py                        # default results dir
    python cohort_summary.py --outdir path/to/results
    python cohort_summary.py --full                 # every column
"""
import argparse, glob, json, os
import pandas as pd

KEY = ["sample", "dx", "n_cells", "n_spots",
       "scale_um_per_px", "scale_array_dev_x", "scale_array_dev_y",
       "area_ceiling", "lddmm_sigmaA", "init_deg", "init_deg_lo", "init_deg_hi",
       "matched_frac", "match_prob_saturated", "n_usable",
       "fitted_affine_det", "jacobian_median", "folded_frac",
       "layer_gm_wm_acc", "layer_within_gm_rho", "layer_exact",
       "expr_n_patterned", "expr_cosine_separation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",
                    default=os.path.expanduser("~/Guo_xenium/stalign_results"))
    ap.add_argument("--full", action="store_true", help="keep every column")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.outdir, "*", "*_metrics.json")))
    if not files:
        raise SystemExit(f"no metrics files under {a.outdir}")
    rows = [json.load(open(f)) for f in files]
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
            for r in rows]
    d = pd.DataFrame(flat).sort_values("sample").reset_index(drop=True)

    out = os.path.join(a.outdir, "cohort_summary.csv")
    d.to_csv(out, index=False)
    print(f"{len(d)} donors -> {out}\n")

    cols = list(d.columns) if a.full else [c for c in KEY if c in d.columns]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(d[cols].to_string(index=False))

    # the three distributions worth reading across a cohort rather than
    # within one donor
    print("\n--- consistency across donors ---")
    if "scale_um_per_px" in d:
        v = d.scale_um_per_px
        print(f"  pixel scale        {v.min():.4f} to {v.max():.4f} um "
              f"(spread {(v.max()-v.min())*1000:.1f} nm)")
        print(f"    recovered per capture area; a narrow spread confirms the "
              f"calibration is stable, a wide one that it must be")
    if {"init_deg_lo", "init_deg_hi"} <= set(d.columns):
        w = d.init_deg_hi - d.init_deg_lo
        print(f"  angle optimum width {w.min():.1f} to {w.max():.1f} deg "
              f"(median {w.median():.1f})")
        print(f"    the width is a property of the ramp-like laminar signal, "
              f"not of any one section")
    if "match_prob_mid_frac" in d:
        m = d.match_prob_mid_frac
        print(f"  match_prob in 0.2-0.8 {m.min()*100:.1f}% to {m.max()*100:.1f}% "
              f"(median {m.median()*100:.1f}%)")
        print(f"    a mixture returning almost nothing in that band is making a")
        print(f"    hard decision, and the 0.85 threshold then selects little")
        print(f"    that the values had not already decided")
    if {"area_ceiling", "match_prob_saturated"} <= set(d.columns):
        sat = d[d.match_prob_saturated == True]
        uns = d[d.match_prob_saturated != True]
        print(f"  matching saturated in {len(sat)} of {len(d)} donors")
        if len(sat) and len(uns):
            lo, hi = sat.area_ceiling.min(), sat.area_ceiling.max()
            print(f"    saturated area ratio   {lo:.2f} to {hi:.2f}")
            print(f"    unsaturated area ratio {uns.area_ceiling.min():.2f} to "
                  f"{uns.area_ceiling.max():.2f}")
            n_over = int((uns.area_ceiling > lo).sum())
            if n_over:
                print(f"    {n_over} unsaturated donors sit above the lowest "
                      f"saturated ratio, so the area ratio does not predict")
                print(f"    saturation and no threshold on it is warranted")


if __name__ == "__main__":
    main()
