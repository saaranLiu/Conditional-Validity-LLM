"""Shared logic for GSS ego-network table construction."""

from __future__ import annotations

import polars as pl

# Demographics + relationship types + interaction (wide suffix 1..N).
ALTER_ATTR_BASES: tuple[str, ...] = (
    "sex",
    "age",
    "race",
    "educ",
    "relig",
    "spouse",
    "parent",
    "sibling",
    "child",
    "othfam",
    "cowork",
    "memgrp",
    "neighbr",
    "friend",
    "advisor",
    "other",
    "talkto",
    "known",
    "rclose",
)

KIN_BASES: tuple[str, ...] = ("parent", "sibling", "child", "othfam")

# Legacy "core" ego columns kept for documentation / optional slim views.
# Network exports now keep the full survey on ego / ego_network; do not use
# this whitelist to drop variables.
EGO_COLS: tuple[str, ...] = (
    "ego_id",
    "id",
    "year",
    "age",
    "sex",
    "race",
    "educ",
    "degree",
    "marital",
    "relig",
    "partyid",
    "region",
    "happy",
    "health",
    "numgiven",
    "wtssall",
    "vpsu",
    "vstrat",
    "ballot",
    "form",
)

NETWORK_WIDE_EGO_COLS: tuple[str, ...] = (
    "ego_id",
    "id",
    "year",
    "age",
    "sex",
    "race",
    "educ",
    "relig",
    "partyid",
    "happy",
    "health",
    "numgiven",
    "eqclose",
    "frndknow",
    "talkpol",
    "wtssall",
)

CLOSE_PAIR_VARS: tuple[str, ...] = (
    "close12",
    "close13",
    "close14",
    "close15",
    "close23",
    "close24",
    "close25",
    "close34",
    "close35",
    "close45",
)


def detect_max_alters(columns: list[str]) -> int:
    for n in range(5, 0, -1):
        if any(f"{base}{n}" in columns for base in ALTER_ATTR_BASES):
            return n
    return 0


def present_alter_bases(df: pl.DataFrame, max_alters: int) -> list[str]:
    """All alter attribute bases that exist as wide columns in the survey."""
    bases: list[str] = []
    for base in ALTER_ATTR_BASES:
        cols = [f"{base}{i}" for i in range(1, max_alters + 1) if f"{base}{i}" in df.columns]
        if cols:
            bases.append(base)
    return bases


def active_alter_bases(df: pl.DataFrame, max_alters: int, min_count: int = 50) -> list[str]:
    """Bases with enough non-null responses (used for module detection)."""
    bases: list[str] = []
    for base in present_alter_bases(df, max_alters):
        col0 = f"{base}1" if f"{base}1" in df.columns else None
        if col0 is None:
            continue
        if df[col0].drop_nulls().len() >= min_count:
            bases.append(base)
    return bases


def add_ego_network_flags(df: pl.DataFrame, max_alters: int) -> pl.DataFrame:
    if "numgiven" not in df.columns:
        return df.with_columns(
            in_network_module=pl.lit(False),
            has_alters=pl.lit(False),
            numgiven_capped=pl.lit(None).cast(pl.Int64),
            numgiven_is_topcoded=pl.lit(False),
        )

    numgiven = pl.col("numgiven").cast(pl.Int64, strict=False)
    capped = numgiven.clip(upper_bound=max_alters) if max_alters > 0 else numgiven

    return df.with_columns(
        in_network_module=numgiven.is_not_null(),
        has_alters=numgiven.is_not_null() & (numgiven > 0),
        numgiven_capped=capped,
        numgiven_is_topcoded=numgiven.is_not_null() & (numgiven >= 6),
    )


def add_kin_column(alter_df: pl.DataFrame) -> pl.DataFrame:
    present = [c for c in KIN_BASES if c in alter_df.columns]
    if not present:
        return alter_df

    is_kin = pl.lit(False)
    for col in present:
        is_kin = is_kin | (pl.col(col) == 1)

    return alter_df.with_columns(
        kin=pl.when(is_kin).then(1).otherwise(2).cast(pl.Int64)
    )


def build_alter_table(
    df: pl.DataFrame,
    max_alters: int,
    bases: list[str] | None = None,
) -> pl.DataFrame:
    """Long alter table from survey wide slots (sex1.. / spouse1..).

    Uses every alter attribute base present in the survey (not a slim subset).
    Structural convenience only: original wide columns remain on ego.
    """
    if "numgiven" not in df.columns:
        raise ValueError("survey table missing `numgiven`")

    if bases is None:
        bases = present_alter_bases(df, max_alters)

    netsize = pl.col("numgiven").cast(pl.Int64, strict=False).clip(upper_bound=max_alters)
    work = df.with_columns(n_alt=netsize).filter(
        pl.col("n_alt").is_not_null() & (pl.col("n_alt") > 0)
    )

    frames: list[pl.DataFrame] = []
    for alt_id in range(1, max_alters + 1):
        exprs: list[pl.Expr] = [
            pl.col("ego_id"),
            pl.lit(alt_id).alias("alter_id"),
        ]
        has_any = False
        for base in bases:
            col = f"{base}{alt_id}"
            if col in work.columns:
                exprs.append(pl.col(col).alias(base))
                has_any = True
        if not has_any:
            continue

        part = (
            work.filter(pl.col("n_alt") >= alt_id)
            .select(exprs)
            .with_columns(pl.col("alter_id").cast(pl.Int64))
        )
        frames.append(part)

    if not frames:
        return pl.DataFrame()

    alter_df = pl.concat(frames, how="vertical_relaxed")
    return add_kin_column(alter_df)


def build_alter_alter_table(df: pl.DataFrame, max_alters: int) -> pl.DataFrame:
    """Edge list from close12..close45, keeping the original survey variable name."""
    if "numgiven" not in df.columns:
        return pl.DataFrame()

    close_cols = [c for c in CLOSE_PAIR_VARS if c in df.columns]
    if not close_cols:
        return pl.DataFrame()

    netsize = pl.col("numgiven").cast(pl.Int64, strict=False).clip(upper_bound=max_alters)
    work = df.with_columns(n_alt=netsize).filter(
        pl.col("n_alt").is_not_null() & (pl.col("n_alt") > 1)
    )
    if work.height == 0:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for col in close_cols:
        digits = "".join(ch for ch in col if ch.isdigit())
        if len(digits) != 2:
            continue
        src_id = int(digits[0])
        tgt_id = int(digits[1])
        part = (
            work.filter(pl.col("n_alt") >= tgt_id)
            .select(
                pl.col("ego_id"),
                pl.lit(src_id).cast(pl.Int64).alias("src_id"),
                pl.lit(tgt_id).cast(pl.Int64).alias("tgt_id"),
                pl.lit(col).alias("close_var"),
                pl.col(col).alias("weight"),
            )
            .filter(pl.col("weight").is_not_null() & (pl.col("weight") != 0))
        )
        if part.height:
            frames.append(part)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def network_wide_columns(df: pl.DataFrame, max_alters: int) -> list[str]:
    cols = [c for c in NETWORK_WIDE_EGO_COLS if c in df.columns]
    for base in ALTER_ATTR_BASES:
        for i in range(1, max_alters + 1):
            col = f"{base}{i}"
            if col in df.columns:
                cols.append(col)
    for i in range(1, max_alters + 1):
        col = f"rclose{i}"
        if col in df.columns:
            cols.append(col)
    for pair in CLOSE_PAIR_VARS:
        if pair in df.columns:
            cols.append(pair)
    # preserve order, dedupe
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def build_ego_table(df: pl.DataFrame, max_alters: int) -> pl.DataFrame:
    """Full survey rows plus network flags (no column whitelist)."""
    flagged = add_ego_network_flags(df, max_alters)
    lead = [
        c
        for c in (
            "ego_id",
            "id",
            "year",
            "numgiven",
            "in_network_module",
            "has_alters",
            "numgiven_capped",
            "numgiven_is_topcoded",
        )
        if c in flagged.columns
    ]
    rest = [c for c in flagged.columns if c not in lead]
    return flagged.select(lead + rest)


def build_network_wide(df: pl.DataFrame, max_alters: int) -> pl.DataFrame:
    """Convenience wide slice of network-module columns (full survey stays on ego)."""
    flagged = add_ego_network_flags(df, max_alters)
    wide_cols = network_wide_columns(flagged, max_alters)
    flag_cols = [
        c
        for c in (
            "in_network_module",
            "has_alters",
            "numgiven_capped",
            "numgiven_is_topcoded",
        )
        if c in flagged.columns and c not in wide_cols
    ]
    return flagged.filter(pl.col("in_network_module")).select(wide_cols + flag_cols)


def has_ego_alter_module(df: pl.DataFrame, max_alters: int, bases: list[str]) -> bool:
    if "numgiven" not in df.columns:
        return False
    if df["numgiven"].drop_nulls().len() < 50:
        return False
    if max_alters == 0:
        return False
    return len(bases) > 0
