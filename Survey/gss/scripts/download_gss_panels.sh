#!/usr/bin/env bash
# Download GSS panel Stata archives into Survey/gss/processed/panel/raw/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/processed/panel/raw"

download() {
  local cohort="$1"
  local url="$2"
  local dest="$RAW/cohort_${cohort}/$(basename "$url")"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    echo "exists: $dest"
    return 0
  fi
  echo "downloading: $url"
  curl -fsSL "$url" -o "$dest"
  unzip -o "$dest" -d "$RAW/cohort_${cohort}"
}

download 2006 "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2006_Panel_Stata.zip"
download 2008 "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2008_Panel_stata.zip"
download 2010 "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2010_Panel_stata.zip"
download 2016 "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_2020_panel_stata_1a.zip"

LEGACY=$(find "$ROOT/processed/by_year/2010" -maxdepth 1 -iname '*panel*.dta' 2>/dev/null | head -1 || true)
if [[ -n "${LEGACY:-}" ]]; then
  mkdir -p "$RAW/cohort_2010"
  cp -n "$LEGACY" "$RAW/cohort_2010/" 2>/dev/null || true
fi

echo "Panel raw files ready under $RAW"
