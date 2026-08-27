#!/usr/bin/env Rscript
# Split GSS cumulative Stata file by survey year and export Python-friendly tables.

suppressPackageStartupMessages({
  library(haven)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else
  "Survey/gss/GSS_stata/gss7224_r3.dta"
output_root <- if (length(args) >= 2) args[[2]] else
  "Survey/gss/processed/by_year"

if (!file.exists(input_path)) {
  stop("Input file not found: ", input_path)
}

script_dir <- dirname(normalizePath(
  sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])
))
source(file.path(script_dir, "value_labels_common.R"))

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

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

aa_pairs <- as.data.frame(t(combn(1:5, 2)))
names(aa_pairs) <- c("a", "b")
aa_varnames <- paste0("close", aa_pairs$a, aa_pairs$b)

has_full_network_module <- function(df) {
  if (!"numgiven" %in% names(df)) {
    return(FALSE)
  }
  n_numgiven <- sum(!is.na(df$numgiven))
  if (n_numgiven < 50) {
    return(FALSE)
  }
  aa_vars <- intersect(aa_varnames, names(df))
  if (length(aa_vars) == 0) {
    return(FALSE)
  }
  any(vapply(aa_vars, function(v) sum(!is.na(df[[v]])) > 0, logical(1)))
}

has_network_module <- has_full_network_module

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
  if (!"numgiven" %in% names(ego_df)) {
    ego_df$in_network_module <- FALSE
    ego_df$has_alters <- FALSE
    ego_df$numgiven_capped <- NA_integer_
    ego_df$numgiven_is_topcoded <- FALSE
    return(ego_df)
  }
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
      "talkpol", "wtssall"
    ),
    names(df)
  )
  alter_keep <- unlist(lapply(alter_attr_bases, function(b) paste0(b, seq_len(max_alters))))
  alter_keep <- intersect(alter_keep, names(df))
  rclose_keep <- intersect(paste0("rclose", seq_len(max_alters)), names(df))
  aa_keep <- intersect(aa_varnames, names(df))
  unique(c(ego_keep, alter_keep, rclose_keep, aa_keep))
}

create_alter_table <- function(df, ego_id_col = "ego_id", max_alters = 5L) {
  if (!"numgiven" %in% names(df)) {
    return(NULL)
  }

  bases <- active_alter_bases(df, max_alters)
  if (length(bases) == 0) {
    return(NULL)
  }

  netsize <- pmin(as.integer(df$numgiven), max_alters, na.rm = FALSE)
  keep <- which(!is.na(netsize) & netsize > 0)
  if (length(keep) == 0) {
    return(NULL)
  }

  alter_vars <- unlist(lapply(bases, function(b) paste0(b, seq_len(max_alters))))
  alter_vars <- intersect(alter_vars, names(df))
  if (length(alter_vars) == 0) {
    return(NULL)
  }

  rows <- list()
  for (i in keep) {
    n_alt <- netsize[[i]]
    ego_id <- df[[ego_id_col]][[i]]
    for (alt_id in seq_len(n_alt)) {
      row <- list(
        ego_id = ego_id,
        alter_id = alt_id
      )
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

create_alter_alter_table <- function(df, ego_id_col = "ego_id") {
  if (!"numgiven" %in% names(df)) {
    return(NULL)
  }

  netsize <- pmin(as.integer(df$numgiven), 5L, na.rm = FALSE)
  keep <- which(!is.na(netsize) & netsize > 1)
  if (length(keep) == 0) {
    return(NULL)
  }

  aa_vars <- intersect(aa_varnames, names(df))
  if (length(aa_vars) == 0) {
    return(NULL)
  }

  rows <- list()
  for (i in keep) {
    n_alt <- netsize[[i]]
    ego_id <- df[[ego_id_col]][[i]]
    for (v in aa_vars) {
      digits <- gsub("[^0-9]", "", v)
      src <- as.integer(substr(digits, 1, 1))
      tgt <- as.integer(substr(digits, 2, 2))
      if (src > n_alt || tgt > n_alt) {
        next
      }
      weight <- df[[v]][[i]]
      if (is.na(weight) || weight == 0) {
        next
      }
      rows[[length(rows) + 1]] <- data.frame(
        ego_id = ego_id,
        src_id = src,
        tgt_id = tgt,
        weight = weight,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0) {
    return(NULL)
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

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

cat("Reading Stata file:", input_path, "\n")
raw <- read_dta(input_path)
if (!"year" %in% names(raw)) {
  stop("Expected variable `year` not found in input file.")
}

years <- sort(unique(raw$year))
manifest <- data.frame(
  year = integer(),
  n_cases = integer(),
  n_variables = integer(),
  has_ego_alter = logical(),
  has_network_module = logical(),
  max_alters = integer(),
  alter_bases = character(),
  n_ego = integer(),
  n_ego_network = integer(),
  n_alter = integer(),
  n_alter_alter = integer(),
  stringsAsFactors = FALSE
)

for (yr in years) {
  cat(sprintf("\n[%d] extracting...\n", yr))
  sub <- raw[raw$year == yr, , drop = FALSE]

  active_cols <- names(sub)[vapply(sub, function(x) any(!is.na(x)), logical(1))]
  sub <- sub[, active_cols, drop = FALSE]

  sub_zap <- zap_labels(sub)
  sub_zap$ego_id <- sprintf("%d_%s", yr, sub_zap$id)

  year_dir <- file.path(output_root, as.character(yr))
  dir.create(year_dir, recursive = TRUE, showWarnings = FALSE)

  survey_path <- file.path(year_dir, "survey.csv")
  write_csv(sub_zap, survey_path, na = "")

  export_labels_json(sub, file.path(year_dir, "variable_labels.json"))
  n_value_labels <- export_value_labels_json(sub, file.path(year_dir, "value_labels.json"))

  ego_alter_flag <- has_ego_alter_module(sub)
  network_flag <- has_full_network_module(sub)
  max_alt <- detect_max_alters(sub)
  bases <- if (max_alt > 0) {
    b <- active_alter_bases(sub, max_alt)
    if (length(b) > 0) paste(c(b, "kin"), collapse = ",") else ""
  } else ""
  n_ego <- 0L
  n_ego_network <- 0L
  n_alter <- 0L
  n_aa <- 0L

  if (ego_alter_flag) {
    net_dir <- file.path(year_dir, "network")
    dir.create(net_dir, recursive = TRUE, showWarnings = FALSE)

    ego_cols <- intersect(
      c(
        "ego_id", "id", "year", "age", "sex", "race", "educ", "degree",
        "marital", "relig", "partyid", "region", "happy", "health",
        "numgiven", "wtssall", "vpsu", "vstrat", "ballot", "form"
      ),
      names(sub_zap)
    )
    ego_df <- add_ego_network_flags(sub_zap[, ego_cols, drop = FALSE], max_alt)
    write_csv(ego_df, file.path(net_dir, "ego.csv"), na = "")

    ego_network_df <- ego_df[ego_df$in_network_module, , drop = FALSE]
    write_csv(ego_network_df, file.path(net_dir, "ego_network.csv"), na = "")

    wide_cols <- network_wide_cols(sub_zap, max_alt)
    wide_df <- sub_zap[sub_zap$numgiven %in% 0:6 | !is.na(sub_zap$numgiven), wide_cols, drop = FALSE]
    wide_df <- wide_df[!is.na(wide_df$numgiven), , drop = FALSE]
    write_csv(wide_df, file.path(net_dir, "network_wide.csv"), na = "")

    alter_df <- create_alter_table(sub_zap, max_alters = max_alt)
    if (!is.null(alter_df) && nrow(alter_df) > 0) {
      write_csv(alter_df, file.path(net_dir, "alter.csv"), na = "")
      n_alter <- nrow(alter_df)
    }

    n_ego <- nrow(ego_df)
    n_ego_network <- nrow(ego_network_df)

    if (network_flag) {
      aa_df <- create_alter_alter_table(sub_zap)
      if (!is.null(aa_df) && nrow(aa_df) > 0) {
        write_csv(aa_df, file.path(net_dir, "alter_alter.csv"), na = "")
        n_aa <- nrow(aa_df)
      }
    }

    alter_bases <- active_alter_bases(sub, max_alt)
    n_net_value_labels <- export_network_value_labels_json(
      sub,
      file.path(net_dir, "value_labels.json"),
      max_alt,
      alter_bases
    )

    cat(sprintf(
      "  ego_alter: ego=%d alter=%d alter_alter=%d (max_alters=%d) value_labels=%d network_value_labels=%d\n",
      n_ego, n_alter, n_aa, max_alt, n_value_labels, n_net_value_labels
    ))
  } else {
    cat(sprintf("  value_labels: %d variables\n", n_value_labels))
  }

  manifest <- rbind(manifest, data.frame(
    year = yr,
    n_cases = nrow(sub_zap),
    n_variables = ncol(sub_zap),
    has_ego_alter = ego_alter_flag,
    has_network_module = network_flag,
    max_alters = max_alt,
    alter_bases = bases,
    n_ego = n_ego,
    n_ego_network = n_ego_network,
    n_alter = n_alter,
    n_alter_alter = n_aa,
    stringsAsFactors = FALSE
  ))

  cat(sprintf(
    "  saved survey.csv (%d cases x %d vars)\n",
    nrow(sub_zap), ncol(sub_zap)
  ))
}

manifest_path <- file.path(dirname(output_root), "manifest.csv")
write_csv(manifest, manifest_path, na = "")
cat("\nManifest written to:", manifest_path, "\n")
cat("Done.\n")
