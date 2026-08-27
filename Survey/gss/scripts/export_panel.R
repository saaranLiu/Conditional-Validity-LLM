#!/usr/bin/env Rscript
# Export GSS panel wide files into cohort/wave folders.
#
# Output layout:
#   Survey/gss/processed/panel/
#     manifest.csv
#     cohort_{year}/
#       panel_source.json
#       panel_wide.csv          # full wide file (zap_labels)
#       raw/                    # original .dta (+ optional .zip)
#       wave_{n}_{interview_year}/
#         survey.csv
#         variable_labels.json

suppressPackageStartupMessages({
  library(haven)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
panel_root <- if (length(args) >= 1) args[[1]] else "Survey/gss/processed/panel"
raw_root <- file.path(panel_root, "raw")

script_dir <- dirname(normalizePath(
  sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))])
))
source(file.path(script_dir, "value_labels_common.R"))

dir.create(panel_root, recursive = TRUE, showWarnings = FALSE)
dir.create(raw_root, recursive = TRUE, showWarnings = FALSE)

var_label <- function(x, v) {
  lab <- attr(x[[v]], "label")
  if (is.null(lab)) {
    return("")
  }
  lab <- as.character(lab)
  if (length(lab) == 0) {
    return("")
  }
  paste(lab, collapse = " ")
}

export_labels_json <- function(df, path) {
  labels <- vapply(names(df), function(v) var_label(df, v), character(1))
  write_json(as.list(labels), path, auto_unbox = TRUE, pretty = TRUE)
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
    stop("Missing ", id_col, " in cohort ", cohort_year)
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

cohort_specs <- list(
  list(
    cohort = 2006L,
    waves = list(
      list(suffix = "1", wave = 1L, interview_year = 2006L),
      list(suffix = "2", wave = 2L, interview_year = 2008L),
      list(suffix = "3", wave = 3L, interview_year = 2010L)
    ),
    url = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2006_Panel_Stata.zip"
  ),
  list(
    cohort = 2008L,
    waves = list(
      list(suffix = "1", wave = 1L, interview_year = 2008L),
      list(suffix = "2", wave = 2L, interview_year = 2010L),
      list(suffix = "3", wave = 3L, interview_year = 2012L)
    ),
    url = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2008_Panel_stata.zip"
  ),
  list(
    cohort = 2010L,
    waves = list(
      list(suffix = "1", wave = 1L, interview_year = 2010L),
      list(suffix = "2", wave = 2L, interview_year = 2012L),
      list(suffix = "3", wave = 3L, interview_year = 2014L)
    ),
    url = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2010_Panel_stata.zip"
  ),
  list(
    cohort = 2016L,
    dual_baseline = TRUE,
    waves = list(
      list(suffix = "1a", wave = "1a", interview_year = 2016L),
      list(suffix = "1b", wave = "1b", interview_year = 2018L),
      list(suffix = "2", wave = 2L, interview_year = 2020L)
    ),
    url = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2020_panel_stata_1a.zip"
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

download_panel_zip <- function(url, dest_zip) {
  if (file.exists(dest_zip)) {
    return(TRUE)
  }
  dir.create(dirname(dest_zip), recursive = TRUE, showWarnings = FALSE)
  cat("Downloading", url, "\n")
  tryCatch(
    {
      download.file(url, dest_zip, mode = "wb", quiet = TRUE)
      TRUE
    },
    error = function(e) {
      warning("Download failed: ", conditionMessage(e))
      FALSE
    }
  )
}

unpack_panel_zip <- function(zip_path, dest_dir) {
  if (!file.exists(zip_path)) {
    return(character())
  }
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  unzip(zip_path, exdir = dest_dir)
  list.files(dest_dir, pattern = "\\.dta$", full.names = TRUE, recursive = TRUE, ignore.case = TRUE)
}

ensure_panel_raw <- function(spec, raw_dir) {
  cohort <- spec$cohort
  cohort_raw <- file.path(raw_dir, paste0("cohort_", cohort))
  dir.create(cohort_raw, recursive = TRUE, showWarnings = FALSE)

  existing <- find_panel_dta(cohort, raw_dir)
  if (!is.null(existing)) {
    return(existing)
  }

  # Legacy path: 2010 panel already under by_year/2010
  if (cohort == 2010L) {
    legacy <- list.files(
      "Survey/gss/processed/by_year/2010",
      pattern = "panel.*\\.dta$",
      full.names = TRUE,
      ignore.case = TRUE
    )
    if (length(legacy) > 0) {
      dest <- file.path(cohort_raw, basename(legacy[[1]]))
      if (!file.exists(dest)) {
        file.copy(legacy[[1]], dest)
      }
      return(dest)
    }
  }

  zip_name <- basename(spec$url)
  zip_path <- file.path(cohort_raw, zip_name)
  if (download_panel_zip(spec$url, zip_path)) {
    dtas <- unpack_panel_zip(zip_path, cohort_raw)
    if (length(dtas) > 0) {
      return(dtas[[1]])
    }
  }

  NULL
}

manifest_rows <- list()

for (spec in cohort_specs) {
  cohort <- spec$cohort
  cat(sprintf("\n=== Cohort %d ===\n", cohort))

  dta_path <- ensure_panel_raw(spec, raw_root)
  if (is.null(dta_path)) {
    warning("No panel file for cohort ", cohort, "; skipping.")
    next
  }

  cat("Reading", dta_path, "\n")
  d <- read_dta(dta_path)
  d_zap <- zap_labels(d)

  cohort_dir <- file.path(panel_root, paste0("cohort_", cohort))
  dir.create(cohort_dir, recursive = TRUE, showWarnings = FALSE)

  write_csv(d_zap, file.path(cohort_dir, "panel_wide.csv"), na = "")
  n_cohort_value_labels <- export_value_labels_json(
    d,
    file.path(cohort_dir, "value_labels.json")
  )
  cat(sprintf("  panel_wide value_labels: %d variables\n", n_cohort_value_labels))

  wave_stats <- list()
  for (w in spec$waves) {
    wave_dir_name <- sprintf("wave_%s_%d", w$wave, w$interview_year)
    wave_dir <- file.path(cohort_dir, wave_dir_name)
    dir.create(wave_dir, recursive = TRUE, showWarnings = FALSE)

    wave_df <- extract_wave(
      d,
      wave_suffix = w$suffix,
      cohort_year = cohort,
      wave_num = w$wave,
      interview_year = w$interview_year
    )

    if (is.null(wave_df) || nrow(wave_df) == 0) {
      cat("  ", wave_dir_name, ": no cases\n")
      next
    }

    write_csv(zap_labels(wave_df), file.path(wave_dir, "survey.csv"), na = "")
    export_labels_json(wave_df, file.path(wave_dir, "variable_labels.json"))
    n_wave_value_labels <- export_panel_wave_value_labels_json(
      d,
      wave_suffix = w$suffix,
      wave_df = wave_df,
      path = file.path(wave_dir, "value_labels.json")
    )

    n_cases <- nrow(wave_df)
    n_vars <- ncol(wave_df)
    cat(sprintf(
      "  %s: %d cases x %d vars (value_labels=%d)\n",
      wave_dir_name, n_cases, n_vars, n_wave_value_labels
    ))

    wave_stats[[wave_dir_name]] <- list(
      wave = w$wave,
      interview_year = w$interview_year,
      n_cases = n_cases,
      n_variables = n_vars
    )

    manifest_rows[[length(manifest_rows) + 1]] <- data.frame(
      cohort_year = cohort,
      wave = as.character(w$wave),
      interview_year = w$interview_year,
      wave_dir = wave_dir_name,
      n_cases = n_cases,
      n_variables = n_vars,
      source_file = basename(dta_path),
      stringsAsFactors = FALSE
    )
  }

  panel_meta <- list(
    cohort_year = cohort,
    source_file = basename(dta_path),
    source_path = dta_path,
    dual_baseline = isTRUE(spec$dual_baseline),
    waves = wave_stats,
    note = if (cohort == 2010L) {
      "Panel cumulative: vars asked in 2+ waves. Ego-network (numgiven) not included; see by_year/2010/network/."
    } else {
      "Exported from GSS panel wide file; one survey.csv per wave."
    }
  )
  write_json(
    panel_meta,
    file.path(cohort_dir, "panel_source.json"),
    auto_unbox = TRUE,
    pretty = TRUE
  )
}

if (length(manifest_rows) > 0) {
  manifest <- do.call(rbind, manifest_rows)
  write_csv(manifest, file.path(panel_root, "manifest.csv"), na = "")
  cat("\nWrote panel manifest:", nrow(manifest), "wave exports\n")
} else {
  cat("\nNo panel waves exported.\n")
}
