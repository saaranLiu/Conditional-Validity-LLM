"""Shared path helpers for processed GSS layout."""

from __future__ import annotations

from pathlib import Path

NETWORK_TABLES = (
    "ego",
    "alter",
    "alter_alter",
)

NETWORK_META = ("network_source.json",)


def network_dir(year_dir: Path) -> Path:
    return year_dir / "network"


def network_table_path(year_dir: Path, name: str, ext: str = ".csv") -> Path:
    return network_dir(year_dir) / f"{name}{ext}"


def legacy_network_table_path(year_dir: Path, name: str, ext: str = ".csv") -> Path:
    return year_dir / f"{name}{ext}"
