"""
src/correlation_analysis.py
----------------------------
Analysis 9: Correlation Analysis.

Computes Pearson and Spearman correlation matrices for all numeric features.
Identifies highly correlated pairs and generates heatmaps.

Saves
-----
tables/correlation_pearson.csv
tables/correlation_spearman.csv
tables/highly_correlated_pairs_pearson.csv
tables/highly_correlated_pairs_spearman.csv
plots/correlation/pearson_heatmap.png
plots/correlation/spearman_heatmap.png
plots/correlation/top_corr_pairs.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import (
    CORR_HIGH,
    FIGURE_DPI,
    FIGURE_SIZE_SQUARE,
    HEATMAP_MAX_COLS,
    LOG_FILE,
    MATPLOTLIB_STYLE,
    PLOTS_CORRELATION_DIR,
    TABLES_DIR,
)
from src.utils import ensure_dirs, safe_corr, save_csv, save_figure, setup_logger

logger = setup_logger(__name__, LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_high_corr_pairs(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Extract upper-triangle pairs where |correlation| >= threshold.

    Returns
    -------
    pd.DataFrame  Columns: Feature A, Feature B, Correlation
    """
    mat = corr_matrix.copy()
    # Zero out lower triangle + diagonal
    mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    pairs = []
    for i, col_a in enumerate(mat.columns):
        for j, col_b in enumerate(mat.columns):
            if j <= i:
                continue
            val = mat.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({
                    "Feature A":    col_a,
                    "Feature B":    col_b,
                    "Correlation":  round(float(val), 6),
                    "Abs Corr":     round(abs(float(val)), 6),
                })
    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values("Abs Corr", ascending=False).reset_index(drop=True)
    return df_pairs


def _plot_corr_heatmap(
    corr_matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    max_cols: int = HEATMAP_MAX_COLS,
) -> None:
    """Plot a seaborn-style correlation heatmap (clipped to max_cols × max_cols)."""
    ensure_dirs(out_path.parent)

    # Clip to the most correlated max_cols columns (by absolute sum)
    if corr_matrix.shape[0] > max_cols:
        col_scores = corr_matrix.abs().sum(axis=0).sort_values(ascending=False)
        top_cols = col_scores.head(max_cols).index.tolist()
        mat = corr_matrix.loc[top_cols, top_cols]
        note = f" (top {max_cols} by total absolute correlation)"
    else:
        mat = corr_matrix
        note = ""

    n = len(mat)
    cell_size = max(0.2, min(0.5, 20 / n))
    fig_size = max(10, n * cell_size + 2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=max(4, 8 - n // 20))
    ax.set_yticklabels(mat.index,   rotation=0,  fontsize=max(4, 8 - n // 20))
    ax.set_title(f"{title}{note}", fontsize=12, fontweight="bold", pad=12)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Correlation Coefficient")
    save_figure(fig, out_path, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Heatmap saved: {out_path}")


def _plot_top_corr_pairs(pairs_df: pd.DataFrame, method: str) -> None:
    """Horizontal bar chart of top 40 highly correlated pairs."""
    ensure_dirs(PLOTS_CORRELATION_DIR)
    if pairs_df.empty:
        return

    top = pairs_df.head(40).copy()
    labels = [f"{r['Feature A']} ↔ {r['Feature B']}" for _, r in top.iterrows()]
    values = top["Correlation"].values

    colors = ["crimson" if v > 0 else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(14, max(8, len(labels) * 0.32)))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(CORR_HIGH,  color="grey", linestyle="--", linewidth=1, label=f"+{CORR_HIGH}")
    ax.axvline(-CORR_HIGH, color="grey", linestyle="--", linewidth=1, label=f"-{CORR_HIGH}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{method.capitalize()} Correlation", fontsize=11)
    ax.set_title(f"Top {len(top)} Highly Correlated Pairs ({method.capitalize()})", fontsize=13, fontweight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=9)

    out = PLOTS_CORRELATION_DIR / f"top_corr_pairs_{method}.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Top pairs chart saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_correlation_analysis(
    df: pd.DataFrame,
    sample_size: int = 30_000,
) -> dict[str, pd.DataFrame]:
    """
    Analysis 9: Pearson and Spearman correlation analysis.

    For large datasets, a random sample is used for Spearman to avoid
    excessive runtime (Pearson always uses the full dataset).

    Parameters
    ----------
    df          : The merged dataset (never modified).
    sample_size : Max rows to use for Spearman correlation.

    Returns
    -------
    dict with keys: 'pearson', 'spearman', 'pearson_pairs', 'spearman_pairs'
    """
    logger.info("Analysis 9 — Correlation Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_CORRELATION_DIR)

    numeric_df = df.select_dtypes(include="number")
    logger.info(f"  {len(numeric_df.columns)} numeric columns available for correlation.")

    # Drop columns with zero std (would produce NaN correlations)
    std = numeric_df.std()
    valid_cols = std[std > 0].index.tolist()
    numeric_clean = numeric_df[valid_cols]
    logger.info(f"  {len(valid_cols)} columns have non-zero std (will be used).")

    # ── Pearson ───────────────────────────────────────────────────────────────
    logger.info("  Computing Pearson correlation...")
    pearson_corr = numeric_clean.corr(method="pearson")
    save_csv(pearson_corr, TABLES_DIR / "correlation_pearson.csv", index=True)

    pearson_pairs = _extract_high_corr_pairs(pearson_corr, CORR_HIGH)
    save_csv(pearson_pairs, TABLES_DIR / "highly_correlated_pairs_pearson.csv")
    logger.info(f"  Pearson: {len(pearson_pairs)} pairs with |r| ≥ {CORR_HIGH}")

    _plot_corr_heatmap(pearson_corr, "Pearson Correlation Matrix", PLOTS_CORRELATION_DIR / "pearson_heatmap.png")
    _plot_top_corr_pairs(pearson_pairs, "pearson")

    # ── Spearman (sampled) ────────────────────────────────────────────────────
    logger.info("  Computing Spearman correlation (sampled)...")
    sample = numeric_clean
    if len(numeric_clean) > sample_size:
        sample = numeric_clean.sample(sample_size, random_state=42)
        logger.info(f"  Using sample of {sample_size:,} rows for Spearman.")

    spearman_corr = sample.corr(method="spearman")
    save_csv(spearman_corr, TABLES_DIR / "correlation_spearman.csv", index=True)

    spearman_pairs = _extract_high_corr_pairs(spearman_corr, CORR_HIGH)
    save_csv(spearman_pairs, TABLES_DIR / "highly_correlated_pairs_spearman.csv")
    logger.info(f"  Spearman: {len(spearman_pairs)} pairs with |r| ≥ {CORR_HIGH}")

    _plot_corr_heatmap(spearman_corr, "Spearman Correlation Matrix", PLOTS_CORRELATION_DIR / "spearman_heatmap.png")
    _plot_top_corr_pairs(spearman_pairs, "spearman")

    # Console
    print("\n" + "=" * 60)
    print("  CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"  Numeric columns used   : {len(valid_cols)}")
    print(f"  Pearson high-corr pairs: {len(pearson_pairs)}  (|r| ≥ {CORR_HIGH})")
    print(f"  Spearman high-corr pairs:{len(spearman_pairs)}  (|r| ≥ {CORR_HIGH})")
    if not pearson_pairs.empty:
        print("\n  Top 5 Pearson pairs:")
        for _, r in pearson_pairs.head(5).iterrows():
            print(f"    {r['Feature A'][:30]:<30} ↔ {r['Feature B'][:30]:<30} : {r['Correlation']:+.4f}")
    print("=" * 60 + "\n")

    logger.info("  ✔ Correlation analysis complete.")
    return {
        "pearson":        pearson_corr,
        "spearman":       spearman_corr,
        "pearson_pairs":  pearson_pairs,
        "spearman_pairs": spearman_pairs,
    }
