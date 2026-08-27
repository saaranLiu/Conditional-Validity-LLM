#!/usr/bin/env Rscript
# Summarize the 2010 Panel cumulative file (longitudinal wide format).

suppressPackageStartupMessages({
  library(haven)
  library(readr)
  library(jsonlite)
})

panel_path <- Sys.getenv(
  "GSS_2010_PANEL",
  unset = "Survey/gss/processed/by_year/2010/GSS_panel2010w123_R6 - stata.dta"
)
output_dir <- Sys.getenv(
  "GSS_2010_OUTPUT",
  unset = "Survey/gss/processed/by_year/2010"
)

if (!file.exists(panel_path)) {
  stop("Panel file not found: ", panel_path)
}

d <- read_dta(panel_path)
waves <- list(
  wave1 = sum(!is.na(d$id_1)),
  wave2 = sum(!is.na(d$id_2)),
  wave3 = sum(!is.na(d$id_3))
)
if ("panstat_2" %in% names(d)) {
  waves$panstat_2 <- as.list(table(d$panstat_2, useNA = "ifany"))
}

meta <- list(
  source_file = basename(panel_path),
  n_panel_starters = nrow(d),
  waves = waves,
  note = paste(
    "Panel cumulative file: variables asked in 2+ waves only.",
    "Ego-network module (numgiven, sex1, close12, ...) is NOT in this file;",
    "see ego.csv from GSS2010merged_R8.dta (network-module respondents, full survey)."
  )
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_json(meta, file.path(output_dir, "panel_source.json"), auto_unbox = TRUE, pretty = TRUE)
cat("Wrote panel_source.json\n")
