#!/usr/bin/env Rscript
# Export ICPSR .rda files under Survey/nshap/Round* to CSV + label JSON for Python.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "Survey/nshap"
out_root <- if (length(args) >= 2) args[[2]] else file.path(root, "processed")

dataset_names <- list(
  "Round1/DS0001" = "core",
  "Round1/DS0004" = "social_networks",
  "Round2/DS0001" = "core",
  "Round2/DS0003" = "partner_disposition_r1",
  "Round2/DS0005" = "social_networks",
  "Round2/DS0007" = "social_networks_update",
  "Round2/DS0012" = "proxy",
  "Round2/DS0014" = "sleep_statistics",
  "Round3/DS0001" = "core",
  "Round3/DS0003" = "social_networks",
  "Round3/DS0005" = "partner_disposition",
  "Round3/DS0007" = "proxy",
  "Round3/DS0011" = "covid"
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

convert_icpsr_df <- function(df) {
  value_labels <- list()
  variable_labels <- list()
  out <- df

  for (col in names(df)) {
    converted <- convert_icpsr_column(df[[col]])
    out[[col]] <- converted$data

    var_lab <- attr(df[[col]], "label")
    if (!is.null(var_lab) && length(var_lab) == 1 && nzchar(as.character(var_lab))) {
      variable_labels[[col]] <- as.character(var_lab)
    }
    if (!is.null(converted$value_labels)) {
      value_labels[[col]] <- converted$value_labels
    }
  }

  list(data = out, value_labels = value_labels, variable_labels = variable_labels)
}

find_rda_files <- function(root) {
  pattern <- file.path(root, "Round*", "ICPSR_*", "DS*", "*-Data.rda")
  sort(Sys.glob(pattern))
}

infer_dataset_name <- function(path, root) {
  rel <- sub(paste0("^", normalizePath(root, winslash = "/"), "/?"), "", normalizePath(path, winslash = "/"))
  parts <- strsplit(rel, "/", fixed = TRUE)[[1]]
  round_dir <- parts[1]
  ds_dir <- parts[3]
  key <- paste(round_dir, ds_dir, sep = "/")
  if (!is.null(dataset_names[[key]])) {
    return(dataset_names[[key]])
  }
  tolower(gsub("^DS", "ds", ds_dir))
}

export_one <- function(path, root, out_root) {
  round_dir <- strsplit(sub(paste0("^", normalizePath(root, winslash = "/"), "/?"), "", normalizePath(path, winslash = "/")), "/", fixed = TRUE)[[1]][1]
  ds_tag <- strsplit(path, "/", fixed = TRUE)[[1]]
  ds_tag <- ds_tag[grep("^DS[0-9]+$", ds_tag)][1]
  dataset_name <- infer_dataset_name(path, root)

  load(path)
  obj_name <- ls()[1]
  raw <- get(obj_name)
  if (!is.data.frame(raw)) {
    stop("Expected a data.frame in ", path, ", got ", class(raw))
  }

  converted <- convert_icpsr_df(raw)
  df <- converted$data

  out_dir <- file.path(out_root, "by_round", round_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  csv_path <- file.path(out_dir, paste0(dataset_name, ".csv"))
  write.csv(df, csv_path, row.names = FALSE, na = "")

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
    round = round_dir,
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

manifest <- do.call(rbind, lapply(rda_files, export_one, root = root, out_root = out_root))
manifest_path <- file.path(out_root, "manifest.csv")
write.csv(manifest, manifest_path, row.names = FALSE)

cat("Exported", nrow(manifest), "dataset(s) to", out_root, "\n")
cat("Manifest:", manifest_path, "\n")
