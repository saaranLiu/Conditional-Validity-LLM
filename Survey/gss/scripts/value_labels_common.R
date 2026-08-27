# Shared helpers for exporting GSS value labels (code -> meaning) to JSON.

extract_value_labels_map <- function(x) {
  labs <- attr(x, "labels")
  if (is.null(labs) || length(labs) == 0) {
    return(NULL)
  }
  vals <- as.numeric(labs)
  names <- names(labs)
  ok <- !is.na(vals)
  if (!any(ok)) {
    return(NULL)
  }
  stats::setNames(as.list(names[ok]), as.character(vals[ok]))
}

variable_value_label_entry <- function(x) {
  var_lab <- attr(x, "label")
  val_labs <- extract_value_labels_map(x)
  if (is.null(var_lab) && is.null(val_labs)) {
    return(NULL)
  }
  entry <- list()
  if (!is.null(var_lab) && length(var_lab) == 1) {
    entry$label <- as.character(var_lab)
  }
  if (!is.null(val_labs)) {
    entry$values <- val_labs
  }
  entry
}

derived_value_labels <- function() {
  list(
    kin = list(
      label = "alter is family member (derived from parent/sibling/child/othfam)",
      values = list(`1` = "kin", `2` = "not kin")
    ),
    in_network_module = list(
      label = "respondent entered ego-network module (derived)",
      values = list(`TRUE` = "yes", `FALSE` = "no")
    ),
    has_alters = list(
      label = "respondent named at least one alter (derived)",
      values = list(`TRUE` = "yes", `FALSE` = "no")
    ),
    numgiven_is_topcoded = list(
      label = "numgiven is top-coded at 6+ (derived)",
      values = list(`TRUE` = "yes", `FALSE` = "no")
    )
  )
}

export_value_labels_json <- function(df, path, vars = NULL) {
  if (is.null(vars)) {
    vars <- names(df)
  }
  vars <- intersect(vars, names(df))
  out <- list()
  for (v in vars) {
    entry <- variable_value_label_entry(df[[v]])
    if (!is.null(entry)) {
      out[[v]] <- entry
    }
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(out, path, auto_unbox = TRUE, pretty = TRUE)
  invisible(length(out))
}

aa_pair_varnames <- function() {
  pairs <- as.data.frame(t(combn(1:5, 2)))
  paste0("close", pairs[[1]], pairs[[2]])
}

export_network_value_labels_json <- function(
    sub,
    path,
    max_alters,
    bases,
    alter_attr_bases = c(
      "sex", "age", "race", "educ", "relig",
      "spouse", "parent", "sibling", "child", "othfam",
      "cowork", "memgrp", "neighbr", "friend", "advisor", "other",
      "talkto", "known", "rclose"
    )) {
  entry_for <- function(x) {
    entry <- variable_value_label_entry(x)
    if (is.null(entry)) {
      return(NULL)
    }
    entry
  }

  alter_long_label <- function(base, entry) {
    if (is.null(entry) || is.null(entry$label)) {
      return(entry)
    }
    entry$label <- sub("person number 1", "person #n", entry$label, ignore.case = TRUE)
    entry$label <- sub("person #1", "person #n", entry$label, ignore.case = TRUE)
    entry$label <- sub("person number n", "person #n", entry$label, ignore.case = TRUE)
    entry
  }

  ego_vars <- c(
    "ego_id", "id", "year", "age", "sex", "race", "educ", "degree",
    "marital", "relig", "partyid", "region", "happy", "health",
    "numgiven", "wtssall", "vpsu", "vstrat", "ballot", "form",
    "eqclose", "frndknow", "talkpol", "samptype", "version"
  )
  ego_out <- list()
  for (v in intersect(ego_vars, names(sub))) {
    entry <- entry_for(sub[[v]])
    if (!is.null(entry)) {
      ego_out[[v]] <- entry
    }
  }
  derived <- derived_value_labels()
  for (name in c("in_network_module", "has_alters", "numgiven_capped", "numgiven_is_topcoded")) {
    ego_out[[name]] <- derived[[name]]
  }

  alter_out <- list(
    ego_id = list(label = "ego identifier ({year}_{id})"),
    alter_id = list(label = "alter nomination rank within ego (1..max_alters)")
  )
  for (base in intersect(bases, alter_attr_bases)) {
    ref <- paste0(base, 1)
    if (ref %in% names(sub)) {
      entry <- alter_long_label(base, entry_for(sub[[ref]]))
      if (!is.null(entry)) {
        alter_out[[base]] <- entry
      }
    }
  }
  alter_out[["kin"]] <- derived[["kin"]]

  wide_out <- list()
  for (v in intersect(c("ego_id", "id", "year"), names(sub))) {
    entry <- entry_for(sub[[v]])
    if (!is.null(entry)) {
      wide_out[[v]] <- entry
    }
  }
  for (base in intersect(bases, alter_attr_bases)) {
    for (n in seq_len(max_alters)) {
      wide <- paste0(base, n)
      if (wide %in% names(sub)) {
        entry <- entry_for(sub[[wide]])
        if (!is.null(entry)) {
          wide_out[[wide]] <- entry
        }
      }
    }
  }
  for (v in intersect(c("numgiven", "eqclose", "frndknow", "talkpol", "wtssall"), names(sub))) {
    entry <- entry_for(sub[[v]])
    if (!is.null(entry)) {
      wide_out[[v]] <- entry
    }
  }
  for (n in seq_len(max_alters)) {
    wide <- paste0("rclose", n)
    if (wide %in% names(sub)) {
      entry <- entry_for(sub[[wide]])
      if (!is.null(entry)) {
        wide_out[[wide]] <- entry
      }
    }
  }
  aa_vars <- intersect(aa_pair_varnames(), names(sub))
  for (wide in aa_vars) {
    entry <- entry_for(sub[[wide]])
    if (!is.null(entry)) {
      wide_out[[wide]] <- entry
    }
  }

  aa_out <- list(
    ego_id = list(label = "ego identifier ({year}_{id})"),
    src_id = list(label = "source alter nomination rank"),
    tgt_id = list(label = "target alter nomination rank")
  )
  if ("close12" %in% names(sub)) {
    entry <- entry_for(sub[["close12"]])
    if (!is.null(entry)) {
      entry$label <- "alter-alter tie strength (from close{i}{j})"
      aa_out[["weight"]] <- entry
    }
  }

  out <- list(
    ego = ego_out,
    alter = alter_out,
    alter_alter = aa_out
  )

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(out, path, auto_unbox = TRUE, pretty = TRUE)
  invisible(length(ego_out) + length(alter_out) + length(aa_out))
}

export_panel_wave_value_labels_json <- function(d, wave_suffix, wave_df, path) {
  ws <- paste0("_", wave_suffix)
  out <- list()

  for (v in names(wave_df)) {
    src_col <- NULL
    if (v %in% names(d)) {
      src_col <- v
    } else {
      candidate <- paste0(v, ws)
      if (candidate %in% names(d)) {
        src_col <- candidate
      }
    }
    if (is.null(src_col)) {
      next
    }
    entry <- variable_value_label_entry(d[[src_col]])
    if (!is.null(entry)) {
      out[[v]] <- entry
    }
  }

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(out, path, auto_unbox = TRUE, pretty = TRUE)
  invisible(length(out))
}
