#!/usr/bin/env Rscript
# Export ICPSR .rda files to CSV + label JSON for Python.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
labels_only <- "--labels-only" %in% args
if (labels_only) {
  args <- args[args != "--labels-only"]
}
root <- if (length(args) >= 1) args[[1]] else "Survey/ucnets"
out_root <- if (length(args) >= 2) args[[2]] else file.path(root, "processed")

dataset_map <- list(
  "DS0001" = list(wave = "Wave1", name = "alters"),
  "DS0002" = list(wave = "Wave2", name = "alters"),
  "DS0003" = list(wave = "Wave3", name = "alters"),
  "DS0004" = list(wave = "combined", name = "alters"),
  "DS0005" = list(wave = "Wave1", name = "ego"),
  "DS0006" = list(wave = "Wave2", name = "ego"),
  "DS0007" = list(wave = "Wave3", name = "ego"),
  "DS0008" = list(wave = "combined", name = "ego")
)

extract_value_labels <- function(x) {
  if (!is.factor(x)) {
    return(NULL)
  }
  levels <- sort(unique(levels(x)))
  codes <- suppressWarnings(as.numeric(sub("^\\(0*([0-9]+)\\).+$", "\\1", levels)))
  labels <- sub("^\\([0-9]+\\) +(.+$)", "\\1", levels)
  ok <- !is.na(codes)
  if (!any(ok)) {
    return(NULL)
  }
  stats::setNames(as.list(labels[ok]), as.character(codes[ok]))
}

convert_icpsr_column <- function(x) {
  if (is.factor(x)) {
    chars <- as.character(x)
    codes <- sub("^\\(0*([0-9]+)\\).+$", "\\1", chars)
    has_numeric_code <- !is.na(chars) & chars != codes
    if (all(has_numeric_code | is.na(chars))) {
      return(list(
        data = suppressWarnings(as.numeric(codes)),
        value_labels = extract_value_labels(x)
      ))
    }
    return(list(data = chars, value_labels = NULL))
  }
  list(data = x, value_labels = NULL)
}

extract_variable_labels <- function(df) {
  variable_labels <- list()

  for (col in names(df)) {
    var_lab <- attr(df[[col]], "label")
    if (!is.null(var_lab) && length(var_lab) == 1 && nzchar(as.character(var_lab))) {
      variable_labels[[col]] <- as.character(var_lab)
    }
  }

  # ICPSR R exports store question text on the data.frame as variable.labels.
  df_var_labels <- attr(df, "variable.labels")
  if (!is.null(df_var_labels) && length(df_var_labels) > 0) {
    for (col in names(df_var_labels)) {
      lab <- as.character(df_var_labels[[col]])
      if (length(lab) == 1 && nzchar(lab)) {
        variable_labels[[col]] <- lab
      }
    }
  }

  variable_labels
}

convert_icpsr_df <- function(df) {
  value_labels <- list()
  out <- df

  for (col in names(df)) {
    converted <- convert_icpsr_column(df[[col]])
    out[[col]] <- converted$data
    if (!is.null(converted$value_labels)) {
      value_labels[[col]] <- converted$value_labels
    }
  }

  list(
    data = out,
    value_labels = value_labels,
    variable_labels = extract_variable_labels(df)
  )
}

find_rda_files <- function(root) {
  pattern <- file.path(root, "ICPSR_*", "DS*", "*-Data.rda")
  sort(Sys.glob(pattern))
}

export_one <- function(path, out_root, labels_only = FALSE) {
  parts <- strsplit(path, "/", fixed = TRUE)[[1]]
  ds_tag <- parts[grep("^DS[0-9]+$", parts)][1]
  if (is.null(ds_tag) || !ds_tag %in% names(dataset_map)) {
    stop("Unknown dataset tag for ", path)
  }

  info <- dataset_map[[ds_tag]]
  wave <- info$wave
  dataset_name <- info$name

  load(path)
  obj_name <- ls()[1]
  raw <- get(obj_name)
  if (!is.data.frame(raw)) {
    stop("Expected a data.frame in ", path, ", got ", class(raw))
  }

  converted <- convert_icpsr_df(raw)
  df <- converted$data

  out_dir <- file.path(out_root, "by_wave", wave)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  csv_path <- file.path(out_dir, paste0(dataset_name, ".csv"))
  if (!labels_only) {
    write.csv(df, csv_path, row.names = FALSE, na = "")
  }

  if (length(converted$variable_labels) > 0) {
    write_json(
      converted$variable_labels,
      file.path(out_dir, paste0("variable_labels_", dataset_name, ".json")),
      auto_unbox = TRUE,
      pretty = TRUE
    )
  }
  if (length(converted$value_labels) > 0) {
    write_json(
      converted$value_labels,
      file.path(out_dir, paste0("value_labels_", dataset_name, ".json")),
      auto_unbox = TRUE,
      pretty = TRUE
    )
  }

  list(
    wave = wave,
    dataset = dataset_name,
    ds = ds_tag,
    source_rda = path,
    object = obj_name,
    rows = nrow(df),
    cols = ncol(df),
    csv = csv_path
  )
}

root <- normalizePath(root, winslash = "/", mustWork = TRUE)
out_root <- normalizePath(out_root, winslash = "/", mustWork = FALSE)
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

rda_files <- find_rda_files(root)
if (length(rda_files) == 0) {
  stop("No *-Data.rda files found under ", root)
}

manifest <- do.call(
  rbind,
  lapply(rda_files, export_one, out_root = out_root, labels_only = labels_only)
)
if (!labels_only) {
  manifest_path <- file.path(out_root, "manifest.csv")
  write.csv(manifest, manifest_path, row.names = FALSE)
  cat("Manifest:", manifest_path, "\n")
}

cat(
  if (labels_only) "Updated label JSON for" else "Exported",
  nrow(manifest),
  "dataset(s) to",
  out_root,
  "\n"
)
