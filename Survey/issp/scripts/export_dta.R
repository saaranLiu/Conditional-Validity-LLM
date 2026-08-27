#!/usr/bin/env Rscript
# Export ISSP Stata file to CSV + label JSON for Python.

suppressPackageStartupMessages({
  library(haven)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else
  "Survey/issp/ZA6980_v2-0-0.dta/ZA6980_v2-0-0.dta"
output_root <- if (length(args) >= 2) args[[2]] else
  "Survey/issp/processed"

if (!file.exists(input_path)) {
  stop("Input file not found: ", input_path)
}

script_dir <- dirname(normalizePath(
  sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])
))
source(file.path(script_dir, "../../gss/scripts/value_labels_common.R"))

export_labels_json <- function(df, path) {
  labels <- vapply(df, function(x) {
    lab <- attr(x, "label")
    if (is.null(lab) || length(lab) != 1) {
      return(NA_character_)
    }
    as.character(lab)
  }, character(1))
  labels <- labels[!is.na(labels)]
  write_json(as.list(labels), path, auto_unbox = TRUE, pretty = TRUE)
}

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

cat("Reading Stata file:", input_path, "\n")
df <- read_dta(input_path)

csv_path <- file.path(output_root, "survey.csv")
write_csv(df, csv_path)
export_labels_json(df, file.path(output_root, "variable_labels.json"))
n_value_labels <- export_value_labels_json(
  df,
  file.path(output_root, "value_labels.json")
)

manifest <- list(
  study = "ZA6980",
  version = unique(as.character(df$version)),
  doi = unique(as.character(df$doi)),
  source_dta = normalizePath(input_path, winslash = "/"),
  rows = nrow(df),
  cols = ncol(df),
  countries = sort(unique(as.character(df$c_alphan))),
  value_label_vars = n_value_labels
)
write_json(
  manifest,
  file.path(output_root, "manifest.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)

cat(
  sprintf(
    "Exported survey.csv (%d rows, %d cols), variable_labels, value_labels (%d vars)\n",
    nrow(df),
    ncol(df),
    n_value_labels
  )
)
