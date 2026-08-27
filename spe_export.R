#!/usr/bin/env Rscript
# =============================================================================
#  spe_export.R -- companion to stalign_pipeline.py
#
#  Two modes:
#    --manifest   write one CSV row per capture area (donor, dx, sex, age,
#                 spots). The Python pipeline reads this to pair each donor
#                 with its Visium capture area without hard-coded tables.
#    --export     write the flat files the pipeline loads: counts.mtx,
#                 logcounts.mtx, barcodes.tsv, genes.tsv, metadata.csv,
#                 coordinates.csv, gene_annotation.csv.
#
#  imgData() is dropped on load. The Lieber SPE stores StoredSpatialImage
#  objects pointing at JHPCE paths that do not exist off-cluster, and any
#  operation touching their print or coerce method throws. Nothing here
#  needs the images.
#
#  Barcodes are written WITHOUT the _<sample_id> suffix so they match the
#  10x barcodes in tissue_positions.csv; the suffixed form is kept in
#  metadata.csv as `key`. array_row and array_col are always included:
#  they encode position on the chip and are the only registration-free
#  way to validate the pixel-to-micron scale.
#
#  USAGE
#    Rscript spe_export.R --rds fnl.rds --manifest manifest.csv
#    Rscript spe_export.R --rds fnl.rds --export --capture V13M06-342_D1 \
#                         --outdir /path/Br8667_export --tag Br8667
# =============================================================================
suppressPackageStartupMessages({
    library(SpatialExperiment); library(SingleCellExperiment); library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(flag, default = NULL) {
    i <- match(flag, args)
    if (is.na(i)) return(default)
    if (i == length(args) || startsWith(args[i + 1], "--")) return(TRUE)
    args[i + 1]
}
has <- function(flag) flag %in% args

RDS      <- getopt("--rds")
MANIFEST <- getopt("--manifest")
CAPTURE  <- getopt("--capture")
OUTDIR   <- getopt("--outdir")
TAG      <- getopt("--tag")
NO_LOG   <- has("--no-logcounts")
stopifnot(!is.null(RDS), file.exists(RDS))

spe <- readRDS(RDS)
# the stored image paths point at JHPCE and do not resolve off-cluster;
# any operation touching their print or coerce method throws, and the
# assignment itself echoes NULL, so both are suppressed
if (methods::is(spe, "SpatialExperiment"))
    invisible(capture.output(try(imgData(spe) <- NULL, silent = TRUE)))
cd_all <- colData(spe)

pick <- function(cands) {
    hit <- intersect(cands, colnames(cd_all))
    if (length(hit)) as.character(cd_all[[hit[1]]]) else rep(NA_character_, ncol(spe))
}

# ------------------------------------------------------------- manifest -----
if (!is.null(MANIFEST) && !isTRUE(MANIFEST)) {
    donor <- pick(c("brnum", "BrNumbr", "sample_label"))
    caps  <- as.character(cd_all$sample_id)
    dfm <- do.call(rbind, lapply(sort(unique(caps)), function(ca) {
        k <- caps == ca
        data.frame(capture_area = ca,
                   donor = paste(unique(donor[k]), collapse = "/"),
                   dx    = paste(unique(pick(c("DX", "dx"))[k]), collapse = "/"),
                   sex   = paste(unique(pick(c("sex"))[k]), collapse = "/"),
                   age   = paste(unique(pick(c("age"))[k]), collapse = "/"),
                   spots = sum(k), stringsAsFactors = FALSE)
    }))
    write.csv(dfm, MANIFEST, row.names = FALSE)
    cat("manifest:", nrow(dfm), "capture areas ->", MANIFEST, "\n")
    quit(save = "no", status = 0)
}

# --------------------------------------------------------------- export -----
stopifnot(!is.null(CAPTURE), !is.null(OUTDIR))
if (is.null(TAG)) TAG <- CAPTURE
if (!CAPTURE %in% cd_all$sample_id)
    stop("capture area not present in the object: ", CAPTURE)

sub <- spe[, cd_all$sample_id == CAPTURE]
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)
p <- function(f) file.path(OUTDIR, paste0(TAG, "_", f))

keys <- colnames(sub)
bcs  <- sub("_.*$", "", keys)
if (anyDuplicated(bcs)) stop("duplicate barcodes after stripping the suffix")

writeMM(as(t(assay(sub, "counts")), "CsparseMatrix"), p("counts.mtx"))
if (!NO_LOG && "logcounts" %in% assayNames(sub))
    writeMM(as(t(assay(sub, "logcounts")), "CsparseMatrix"), p("logcounts.mtx"))
writeLines(bcs,           p("barcodes.tsv"))
writeLines(rownames(sub), p("genes.tsv"))

md <- as.data.frame(colData(sub))
md <- md[, vapply(md, function(v) is.atomic(v) && !is.list(v), logical(1)), drop = FALSE]
write.csv(cbind(barcode = bcs, key = keys, md), p("metadata.csv"), row.names = FALSE)

sc <- spatialCoords(sub); cn <- colnames(sc)
ix <- if ("pxl_col_in_fullres" %in% cn) "pxl_col_in_fullres" else cn[1]
iy <- if ("pxl_row_in_fullres" %in% cn) "pxl_row_in_fullres" else cn[2]
write.csv(data.frame(barcode = bcs, x = sc[, ix], y = sc[, iy]),
          p("coordinates.csv"), row.names = FALSE)

rd <- as.data.frame(rowData(sub))
rd <- rd[, vapply(rd, function(v) is.atomic(v) && !is.list(v), logical(1)), drop = FALSE]
rd <- cbind(gene_id = rownames(sub), rd)
if (!"gene_name" %in% colnames(rd)) {
    alt <- intersect(c("symbol", "Symbol", "gene_symbol"), colnames(rd))
    rd$gene_name <- if (length(alt)) rd[[alt[1]]] else rownames(sub)
}
write.csv(rd, p("gene_annotation.csv"), row.names = FALSE)

cat("exported", CAPTURE, "->", OUTDIR,
    "(", ncol(sub), "spots x", nrow(sub), "genes )\n")
cat("  x from", ix, "| y from", iy, "\n")
