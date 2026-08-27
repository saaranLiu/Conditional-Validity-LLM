#!/usr/bin/env Rscript
# Backfill value_labels.json for cross-section, network, and panel exports.
#
# Does not rewrite survey.csv; reads Stata sources and writes JSON alongside
# existing processed outputs.

suppressPackageStartupMessages({
  library(haven)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "Survey/gss/GSS_stata/gss7224_r3.dta"
by_year_root <- if (length(args) >= 2) args[[2]] else "Survey/gss/processed/by_year"
panel_root <- if (length(args) >= 3) args[[3]] else "Survey/gss/processed/panel"

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

has_ego_alter_module <- function(df) {
  if (!"numgiven" %in% names(df)) {
    return(FALSE)
  }
  if (sum(!is.na(df$numgiven)) < 50) {
    return(FALSE)
  }
  max_alt <- detect_max_alters(df)
  if (max_alt == 0L) {
    return(FALSE)
  }
  length(active_alter_bases(df, max_alt)) > 0
}

export_cross_section <- function(input_path, by_year_root) {
  if (!file.exists(input_path)) {
    warning("Cross-section input not found: ", input_path)
    return(invisible(NULL))
  }

  cat("Cross-section:", input_path, "\n")
  raw <- read_dta(input_path)
  if (!"year" %in% names(raw)) {
    stop("Expected variable `year` not found in ", input_path)
  }

  for (yr in sort(unique(raw$year))) {
    year_dir <- file.path(by_year_root, as.character(yr))
    if (!dir.exists(year_dir)) {
      next
    }

    sub <- raw[raw$year == yr, , drop = FALSE]
    active_cols <- names(sub)[vapply(sub, function(x) any(!is.na(x)), logical(1))]
    sub <- sub[, active_cols, drop = FALSE]

    n_all <- export_value_labels_json(sub, file.path(year_dir, "value_labels.json"))
    msg <- sprintf("  [%d] survey value_labels: %d vars", yr, n_all)

    if (has_ego_alter_module(sub)) {
      net_dir <- file.path(year_dir, "network")
      if (dir.exists(net_dir)) {
        max_alt <- detect_max_alters(sub)
        bases <- active_alter_bases(sub, max_alt)
        n_net <- export_network_value_labels_json(
          sub,
          file.path(net_dir, "value_labels.json"),
          max_alt,
          bases
        )
        msg <- paste0(msg, sprintf("; network: %d vars", n_net))
      }
    }
    cat(msg, "\n")
  }
}

export_merged_2010_network <- function(by_year_root) {
  merged_path <- file.path(by_year_root, "2010", "GSS2010merged_R8.dta")
  net_dir <- file.path(by_year_root, "2010", "network")
  if (!file.exists(merged_path) || !dir.exists(net_dir)) {
    return(invisible(NULL))
  }

  cat("2010 merged network:", merged_path, "\n")
  raw <- read_dta(merged_path)
  net_idx <- which(!is.na(raw$numgiven))
  sub <- raw[net_idx, , drop = FALSE]
  max_alt <- detect_max_alters(sub)
  bases <- active_alter_bases(sub, max_alt)
  n_net <- export_network_value_labels_json(
    sub,
    file.path(net_dir, "value_labels.json"),
    max_alt,
    bases
  )
  cat(sprintf("  network value_labels: %d vars\n", n_net))
}

cohort_specs <- list(
  list(
    cohort = 2006L,
    waves = list(
      list(suffix = "1", interview_year = 2006L),
      list(suffix = "2", interview_year = 2008L),
      list(suffix = "3", interview_year = 2010L)
    )
  ),
  list(
    cohort = 2008L,
    waves = list(
      list(suffix = "1", interview_year = 2008L),
      list(suffix = "2", interview_year = 2010L),
      list(suffix = "3", interview_year = 2012L)
    )
  ),
  list(
    cohort = 2010L,
    waves = list(
      list(suffix = "1", interview_year = 2010L),
      list(suffix = "2", interview_year = 2012L),
      list(suffix = "3", interview_year = 2014L)
    )
  ),
  list(
    cohort = 2016L,
    waves = list(
      list(suffix = "1a", interview_year = 2016L),
      list(suffix = "1b", interview_year = 2018L),
      list(suffix = "2", interview_year = 2020L)
    )
  )
)

find_panel_dta <- function(cohort_year, raw_dir) {
  cohort_raw <- file.path(raw_dir, paste0("cohort_", cohort_year))
  if (!dir.exists(cohort_raw)) {
    return(NULL)
  }
  dtas <- list.files(cohort_raw, pattern = "\\.dta$", full.names = TRUE, ignore.case = TRUE)
  if (length(dtas) == 0) {
    return(NULL)
  }
  dtas[[1]]
}

wave_suffix_pattern <- function(suffix) {
  paste0("_", suffix, "$")
}

is_wave_var <- function(names, suffixes) {
  Reduce(`|`, lapply(suffixes, function(s) grepl(wave_suffix_pattern(s), names)))
}

extract_wave <- function(d, wave_suffix, cohort_year, wave_num, interview_year) {
  id_col <- paste0("id_", wave_suffix)
  year_col <- paste0("year_", wave_suffix)
  panstat_col <- paste0("panstat_", wave_suffix)

  if (!id_col %in% names(d)) {
    return(NULL)
  }

  keep <- which(!is.na(d[[id_col]]))
  if (length(keep) == 0) {
    return(NULL)
  }

  sub <- d[keep, , drop = FALSE]
  ws <- paste0("_", wave_suffix)
  wave_cols <- grep(paste0(ws, "$"), names(sub), value = TRUE)
  wave_cols <- setdiff(wave_cols, c(id_col, year_col, panstat_col))

  static_cols <- names(sub)[!is_wave_var(names(sub), c("1", "2", "3", "1a", "1b"))]
  static_cols <- setdiff(static_cols, wave_cols)

  out <- sub[, static_cols, drop = FALSE]
  for (col in wave_cols) {
    new_name <- sub(paste0(ws, "$"), "", col)
    out[[new_name]] <- sub[[col]]
  }

  out$id <- sub[[id_col]]
  out$year <- sub[[year_col]]
  if (panstat_col %in% names(sub)) {
    out$panstat <- sub[[panstat_col]]
  }
  out$cohort_year <- cohort_year
  out$wave_num <- wave_num
  out$interview_year <- interview_year
  out$panel_id <- sprintf("%d_%s", cohort_year, out$id)
  out$ego_id <- sprintf("%d_%s", out$year, out$id)
  out
}

export_panels <- function(panel_root) {
  raw_root <- file.path(panel_root, "raw")
  if (!dir.exists(panel_root)) {
    return(invisible(NULL))
  }

  cat("Panel:", panel_root, "\n")
  for (spec in cohort_specs) {
    cohort <- spec$cohort
    dta_path <- find_panel_dta(cohort, raw_root)
    cohort_dir <- file.path(panel_root, paste0("cohort_", cohort))
    if (is.null(dta_path) || !dir.exists(cohort_dir)) {
      cat(sprintf("  cohort %d: skipped (missing dta or export dir)\n", cohort))
      next
    }

    cat(sprintf("  cohort %d: %s\n", cohort, basename(dta_path)))
    d <- read_dta(dta_path)
    n_cohort <- export_value_labels_json(d, file.path(cohort_dir, "value_labels.json"))

    for (w in spec$waves) {
      wave_dir_name <- sprintf("wave_%s_%d", w$suffix, w$interview_year)
      wave_dir <- file.path(cohort_dir, wave_dir_name)
      if (!dir.exists(wave_dir)) {
        next
      }

      wave_df <- extract_wave(
        d,
        wave_suffix = w$suffix,
        cohort_year = cohort,
        wave_num = w$suffix,
        interview_year = w$interview_year
      )
      if (is.null(wave_df) || nrow(wave_df) == 0) {
        next
      }

      n_wave <- export_panel_wave_value_labels_json(
        d,
        wave_suffix = w$suffix,
        wave_df = wave_df,
        path = file.path(wave_dir, "value_labels.json")
      )
      cat(sprintf("    %s: %d vars (cohort wide: %d)\n", wave_dir_name, n_wave, n_cohort))
    }
  }
}

export_cross_section(input_path, by_year_root)
export_merged_2010_network(by_year_root)
export_panels(panel_root)
cat("Done.\n")
