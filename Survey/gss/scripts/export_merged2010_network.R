#!/usr/bin/env Rscript
# Extract 2010 ego-network tables from GSS 2010 Merged data.
#
# The 2010 Panel cumulative file omits variables asked only once (incl. numgiven).
# Network module responses live in GSS2010merged_R8.dta (samptype=2006, n=1272).

suppressPackageStartupMessages({
  library(haven)
  library(readr)
  library(jsonlite)
})

merged_path <- Sys.getenv(
  "GSS_2010_MERGED",
  unset = "Survey/gss/processed/by_year/2010/GSS2010merged_R8.dta"
)
output_dir <- Sys.getenv(
  "GSS_2010_OUTPUT",
  unset = "Survey/gss/processed/by_year/2010/network"
)

script_dir <- dirname(normalizePath(
  sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])
))
source(file.path(script_dir, "value_labels_common.R"))

alter_attr_bases <- c(
  "sex", "age", "race", "educ", "relig",
  "spouse", "parent", "sibling", "child", "othfam",
  "cowork", "memgrp", "neighbr", "friend", "advisor", "other",
  "talkto", "known", "rclose"
)
kin_bases <- c("parent", "sibling", "child", "othfam")

detect_max_alters <- function(df) {
  for (n in 5:1) {
    cols <- paste0(alter_attr_bases, n)
    if (any(cols %in% names(df))) {
      return(n)
    }
  }
  0L
}

active_alter_bases <- function(df, max_alters, min_count = 50L) {
  bases <- character(0)
  for (base in alter_attr_bases) {
    cols <- paste0(base, seq_len(max_alters))
    cols <- intersect(cols, names(df))
    if (length(cols) == 0) {
      next
    }
    if (sum(!is.na(df[[cols[[1]]]])) >= min_count) {
      bases <- c(bases, base)
    }
  }
  bases
}

add_kin_column <- function(alter_df) {
  present <- intersect(kin_bases, names(alter_df))
  if (length(present) == 0) {
    return(alter_df)
  }
  is_kin <- rep(FALSE, nrow(alter_df))
  for (col in present) {
    is_kin <- is_kin | (alter_df[[col]] == 1)
  }
  alter_df$kin <- ifelse(is_kin, 1L, 2L)
  alter_df
}

add_ego_network_flags <- function(ego_df, max_alters = 5L) {
  numgiven <- as.integer(ego_df$numgiven)
  capped <- pmin(numgiven, max_alters, na.rm = FALSE)
  ego_df$in_network_module <- !is.na(numgiven)
  ego_df$has_alters <- !is.na(numgiven) & numgiven > 0L
  ego_df$numgiven_capped <- capped
  ego_df$numgiven_is_topcoded <- !is.na(numgiven) & numgiven >= 6L
  ego_df
}

network_wide_cols <- function(df, max_alters) {
  ego_keep <- intersect(
    c(
      "ego_id", "id", "year", "age", "sex", "race", "educ", "relig",
      "partyid", "happy", "health", "numgiven", "eqclose", "frndknow",
      "talkpol", "wtssall", "samptype", "version", "form"
    ),
    names(df)
  )
  alter_keep <- unlist(lapply(alter_attr_bases, function(b) paste0(b, seq_len(max_alters))))
  alter_keep <- intersect(alter_keep, names(df))
  rclose_keep <- intersect(paste0("rclose", seq_len(max_alters)), names(df))
  aa_keep <- intersect(
    paste0("close", c(12, 13, 14, 15, 23, 24, 25, 34, 35, 45)),
    names(df)
  )
  unique(c(ego_keep, alter_keep, rclose_keep, aa_keep))
}

create_alter_table <- function(df, ego_id_col = "ego_id", max_alters = 5L) {
  bases <- active_alter_bases(df, max_alters)
  if (length(bases) == 0) {
    return(NULL)
  }
  netsize <- pmin(as.integer(df$numgiven), max_alters, na.rm = FALSE)
  keep <- which(!is.na(netsize) & netsize > 0)
  if (length(keep) == 0) {
    return(NULL)
  }
  rows <- list()
  for (i in keep) {
    n_alt <- netsize[[i]]
    ego_id <- df[[ego_id_col]][[i]]
    for (alt_id in seq_len(n_alt)) {
      row <- list(ego_id = ego_id, alter_id = alt_id)
      for (base in bases) {
        col <- paste0(base, alt_id)
        if (col %in% names(df)) {
          row[[base]] <- df[[col]][[i]]
        }
      }
      rows[[length(rows) + 1]] <- row
    }
  }
  out <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  rownames(out) <- NULL
  add_kin_column(out)
}

if (!file.exists(merged_path)) {
  stop("Merged 2010 file not found: ", merged_path)
}

cat("Reading", merged_path, "\n")
raw <- read_dta(merged_path)
if (!"numgiven" %in% names(raw)) {
  stop("Variable `numgiven` not found in merged file.")
}

net_idx <- which(!is.na(raw$numgiven))
sub <- raw[net_idx, , drop = FALSE]
sub_zap <- zap_labels(sub)
sub_zap$ego_id <- sprintf("2010_%s", sub_zap$id)
sub_zap$year <- 2010L

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

max_alt <- detect_max_alters(sub_zap)
bases <- active_alter_bases(sub_zap, max_alt)

# Full survey columns for network-module respondents only (no whitelist, no aliases).
ego_df <- add_ego_network_flags(sub_zap, max_alt)
lead <- intersect(
  c(
    "ego_id", "id", "year", "numgiven",
    "in_network_module", "has_alters", "numgiven_capped", "numgiven_is_topcoded"
  ),
  names(ego_df)
)
ego_df <- ego_df[, c(lead, setdiff(names(ego_df), lead)), drop = FALSE]
write_csv(ego_df, file.path(output_dir, "ego.csv"), na = "")

alter_df <- create_alter_table(sub_zap, max_alters = max_alt)
n_alter <- 0L
if (!is.null(alter_df) && nrow(alter_df) > 0) {
  write_csv(alter_df, file.path(output_dir, "alter.csv"), na = "")
  n_alter <- nrow(alter_df)
}

# Drop legacy redundant tables if present from older exports.
for (name in c("ego_network", "survey_network", "network_wide", "alter_alter")) {
  for (ext in c(".csv", ".parquet")) {
    p <- file.path(output_dir, paste0(name, ext))
    if (file.exists(p)) {
      file.remove(p)
    }
  }
}

meta <- list(
  source_file = basename(merged_path),
  note = paste(
    "2010 ego-network from GSS 2010 Merged (not cross-section cumulative).",
    "All network respondents are samptype=2006 (2006 panel wave 3 interviewed in 2010).",
    "The 2010 Panel cumulative file excludes these variables (asked only once).",
    "Tables: ego (network-module only, full survey) + alter. No alter_alter (close12..) in this release."
  ),
  n_network_respondents = nrow(ego_df),
  n_ego = nrow(ego_df),
  n_alter = n_alter,
  n_alter_alter = 0L,
  max_alters = max_alt,
  alter_bases = c(bases, "kin"),
  samptype_counts = as.list(table(sub_zap$samptype))
)
write_json(meta, file.path(output_dir, "network_source.json"), auto_unbox = TRUE, pretty = TRUE)

n_net_value_labels <- export_network_value_labels_json(
  sub,
  file.path(output_dir, "value_labels.json"),
  max_alt,
  bases
)

cat(sprintf(
  "Wrote 2010 network tables: ego=%d alter=%d max_alters=%d bases=%s value_labels=%d\n",
  nrow(ego_df), n_alter, max_alt,
  paste(c(bases, "kin"), collapse = ","), n_net_value_labels
))
