# analytics/descriptive/kpi.py
import os, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm

# ------- display defaults
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.bbox"] = "tight"

FIGSIZE = (10, 6)  # compact window for most plots

def _new_fig(figsize=FIGSIZE, left=0.12, right=0.98, top=0.92, bottom=0.35):
    """Compact figure with big bottom margin so rotated x-ticks never clip."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax.tick_params(labelsize=9)
    return fig, ax

def _finalize(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=240, bbox_inches="tight")
    print(f"✓ Saved figure: {path}")
    try:
        plt.show()
    finally:
        plt.close(fig)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _to_json(df: pd.DataFrame, path: Path):
    path.write_text(
        df.to_json(orient="records", date_format="iso", force_ascii=False, indent=2),
        encoding="utf-8"
    )

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["date","receipt","item_code","description","qty","sales",
                "cost","discount","profit","payment","cashier_id","category","tab"]:
        if col not in df.columns: df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["qty","sales","cost","discount","profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()
    df = df.dropna(subset=["date","description"])
    return df

def _fmt_int(ax, *, x_numeric: bool=False, y_numeric: bool=True):
    if y_numeric:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v)):,}"))
    if x_numeric:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v)):,}"))

def _set_zoom_limits(ax, ydata, lower_q=2, upper_q=98, pad_frac=0.06):
    """Zoom inward but always include true min/max (+padding) so nothing clips."""
    y = pd.Series(ydata).dropna().astype(float)
    if len(y) == 0: return
    y_min, y_max = float(y.min()), float(y.max())
    lo_q, hi_q = np.percentile(y, [lower_q, upper_q])
    lo = min(lo_q, y_min); hi = max(hi_q, y_max)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - pad_frac*span, hi + pad_frac*span)

# ---------------------------- heatmap (unchanged) ----------------------------
def _plot_season_index_heatmap(season: pd.DataFrame, out_dir: Path):
    if season.empty: return
    pivot = season.pivot(index="category", columns="month_num", values="season_index").sort_index()

    vals = pivot.values[np.isfinite(pivot.values)]
    vmin = max(0.6, np.nanpercentile(vals, 10)) if vals.size else 0.6
    vmax = min(1.6, np.nanpercentile(vals, 90)) if vals.size else 1.6
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig_height = max(12, len(pivot) * 0.65)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.38, right=0.95, top=0.93, bottom=0.08)

    im = ax.imshow(pivot.values, aspect="auto", norm=norm, cmap="viridis")
    ax.set_title("Season Index Heatmap (Category × Month)\n(1.0 = category average; >1 above, <1 below)")
    ax.set_xlabel("Month Number"); ax.set_ylabel("Category")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color="white")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Season Index (center=1.0)")
    _finalize(fig, out_dir / "fig_season_index_heatmap.png")

# ---------------------------- NEW: combined figures ----------------------------
def _plot_sales_month_vs_year(monthly: pd.DataFrame, yearly: pd.DataFrame, out_dir: Path):
    if monthly.empty or yearly.empty: return
    monthly = monthly.sort_values("month")
    years = yearly["year"].astype(int).to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.34, wspace=0.22)

    # Left: monthly sales
    ax1.plot(monthly["month"], monthly["total_sales"], marker="o", linewidth=1.6, markersize=4, label="Sales")
    ax1.set_title("Total Sales per Month")
    ax1.set_xlabel("Month"); ax1.set_ylabel("Sales")
    for label in ax1.get_xticklabels():
        label.set_rotation(60); label.set_ha("right")
    _fmt_int(ax1, y_numeric=True)
    _set_zoom_limits(ax1, monthly["total_sales"])

    # Right: yearly sales (force integer ticks & labels)
    ax2.plot(years, yearly["total_sales"], marker="o", linewidth=1.8, label="Sales")
    ax2.set_title("Total Sales per Year")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Sales")
    _fmt_int(ax2, y_numeric=True)
    _set_zoom_limits(ax2, yearly["total_sales"], lower_q=0, upper_q=100)

    # <<< this is the key >>>
    ax2.set_xticks(years)
    ax2.set_xticklabels([str(y) for y in years])
    ax2.xaxis.set_major_locator(mticker.FixedLocator(years))
    ax2.xaxis.set_major_formatter(mticker.FixedFormatter([str(y) for y in years]))
    # optional small padding so endpoints aren’t clipped
    ax2.set_xlim(years.min() - 0.2, years.max() + 0.2)

    ax2.legend()

    out_path = Path(out_dir) / "fig_sales_month_vs_year.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    print(f"✓ Saved figure: {out_path}")
    try:
        plt.show()
    finally:
        plt.close(fig)

def _plot_qty_month_vs_year(monthly: pd.DataFrame, yearly: pd.DataFrame, out_dir: Path):
    if monthly.empty or yearly.empty: return
    monthly = monthly.sort_values("month")
    years = yearly["year"].astype(int).to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.34, wspace=0.22)

    # Left: monthly qty
    ax1.plot(monthly["month"], monthly["total_qty"], marker="o", linewidth=1.6, markersize=4, label="Quantity")
    ax1.set_title("Total Quantity per Month")
    ax1.set_xlabel("Month"); ax1.set_ylabel("Quantity")
    for label in ax1.get_xticklabels():
        label.set_rotation(60); label.set_ha("right")
    _fmt_int(ax1, y_numeric=True)
    _set_zoom_limits(ax1, monthly["total_qty"])

    # Right: yearly qty (force integer ticks & labels)
    ax2.plot(years, yearly["total_qty"], marker="o", linewidth=1.8, label="Quantity")
    ax2.set_title("Total Quantity per Year")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Quantity")
    _fmt_int(ax2, y_numeric=True)
    _set_zoom_limits(ax2, yearly["total_qty"], lower_q=0, upper_q=100)

    ax2.set_xticks(years)
    ax2.set_xticklabels([str(y) for y in years])
    ax2.xaxis.set_major_locator(mticker.FixedLocator(years))
    ax2.xaxis.set_major_formatter(mticker.FixedFormatter([str(y) for y in years]))
    ax2.set_xlim(years.min() - 0.2, years.max() + 0.2)

    ax2.legend()

    out_path = Path(out_dir) / "fig_qty_month_vs_year.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    print(f"✓ Saved figure: {out_path}")
    try:
        plt.show()
    finally:
        plt.close(fig)

# ---------------------------- KPI compute ----------------------------
def compute_kpis(df: pd.DataFrame, out_dir: str) -> Dict[str, Any]:
    out_path = _ensure_dir(Path(out_dir))
    df = _normalize_columns(df)

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_num"] = df["date"].dt.month

    monthly = df.groupby("month", as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))
    yearly  = df.groupby("year", as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))

    base_cols = ["item_code","description","category","tab"]
    by_prod = df.groupby(base_cols, as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))
    top10_sales = by_prod.sort_values("total_sales", ascending=False).head(10).reset_index(drop=True)
    top10_qty   = by_prod.sort_values("total_qty",   ascending=False).head(10).reset_index(drop=True)

    # Top 10 by YEAR
    by_year_prod = df.groupby(["year"] + base_cols, as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))
    by_year_prod["rank_sales"] = by_year_prod.groupby("year")["total_sales"].rank(method="first", ascending=False)
    by_year_prod["rank_qty"]   = by_year_prod.groupby("year")["total_qty"].rank(method="first", ascending=False)
    top10_sales_year = by_year_prod[by_year_prod["rank_sales"] <= 10].drop(columns=["rank_qty"])
    top10_qty_year   = by_year_prod[by_year_prod["rank_qty"]   <= 10].drop(columns=["rank_sales"])

    # Top 10 by MONTH (YYYY-MM)
    by_month_prod = df.groupby(["month"] + base_cols, as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))
    by_month_prod["rank_sales"] = by_month_prod.groupby("month")["total_sales"].rank(method="first", ascending=False)
    by_month_prod["rank_qty"]   = by_month_prod.groupby("month")["total_qty"].rank(method="first", ascending=False)
    top10_sales_month = by_month_prod[by_month_prod["rank_sales"] <= 10].drop(columns=["rank_qty"])
    top10_qty_month   = by_month_prod[by_month_prod["rank_qty"]   <= 10].drop(columns=["rank_sales"])

    # Splits
    cat_month = df.groupby(["month","category"], as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))
    tab_month = df.groupby(["month","tab"], as_index=False).agg(
        total_sales=("sales","sum"), total_qty=("qty","sum"))

    # Active SKUs + item lists (monthly & yearly) for drill-down
    active_skus = (df.dropna(subset=["item_code"])
                     .groupby("month")["item_code"]
                     .nunique()
                     .reset_index(name="active_skus"))
    active_items_monthly = (df.dropna(subset=["item_code"])
                              [["month","item_code","description","category","tab"]]
                              .drop_duplicates()
                              .sort_values(["month","category","description"]))
    active_items_yearly = (df.dropna(subset=["item_code"]).assign(year=df["date"].dt.year)
                              [["year","item_code","description","category","tab"]]
                              .drop_duplicates()
                              .sort_values(["year","category","description"]))

    # Season index per category × month
    cat_month2 = df.groupby(["category","month_num"], as_index=False).agg(month_sales=("sales","sum"))
    cat_avg = (cat_month2.groupby("category", as_index=False)["month_sales"]
               .mean().rename(columns={"month_sales":"avg_sales"}))
    season = cat_month2.merge(cat_avg, on="category", how="left")
    season["season_index"] = season.apply(
        lambda r: (r["month_sales"]/r["avg_sales"]) if r["avg_sales"] else np.nan, axis=1
    )
    season = season[["category","month_num","month_sales","avg_sales","season_index"]]\
                 .sort_values(["category","month_num"])

    # Avg monthly growth (CAGR-style)
    mser = monthly.set_index("month")["total_sales"].sort_index()
    if len(mser) >= 2 and pd.notnull(mser.iloc[0]) and mser.iloc[0] != 0:
        months_n = len(mser) - 1
        avg_monthly_growth = (mser.iloc[-1] / mser.iloc[0]) ** (1/months_n) - 1
    else:
        avg_monthly_growth = np.nan

    # Save CSV + JSON (unchanged)
    def save(df_, name):
        (out_path / f"{name}.csv").write_text(df_.to_csv(index=False), encoding="utf-8")
        _to_json(df_, out_path / f"{name}.json")

    out_path = out_dir = _ensure_dir(out_path)  # keep alias used above
    save(monthly, "kpi_monthly_sales_qty")
    save(yearly,  "kpi_yearly_sales_qty")
    save(top10_sales, "kpi_top10_by_sales")
    save(top10_qty,   "kpi_top10_by_qty")
    save(top10_sales_year,  "kpi_top10_by_sales_by_year")
    save(top10_qty_year,    "kpi_top10_by_qty_by_year")
    save(top10_sales_month, "kpi_top10_by_sales_by_month")
    save(top10_qty_month,   "kpi_top10_by_qty_by_month")
    save(cat_month, "kpi_category_split_monthly")
    save(tab_month, "kpi_tab_split_monthly")
    save(active_skus,          "kpi_active_skus_monthly")
    save(active_items_monthly, "kpi_active_skus_items_monthly")
    save(active_items_yearly,  "kpi_active_skus_items_yearly")
    save(season,               "kpi_season_index_category")

    # --------- ONLY TWO FIGURES NOW (each has 2 plots) ---------
    _plot_sales_month_vs_year(monthly, yearly, out_path)
    _plot_qty_month_vs_year(monthly, yearly, out_path)
    _plot_season_index_heatmap(season, out_path)  # keep this single heatmap

    # ---------- Manifest for frontend filters ----------
    manifest = {
        "years": sorted([int(y) for y in df["year"].dropna().unique().tolist()]),
        "months": sorted(df["month"].dropna().unique().tolist()),
        "categories": sorted([str(x) for x in df["category"].dropna().unique().tolist()]),
        "tabs": sorted([str(x) for x in df["tab"].dropna().unique().tolist()]),
        "files": sorted([p.name for p in out_path.glob("*.*")])
    }
    Path(out_path / "kpi_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "profile": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()) if pd.notnull(df["date"].min()) else None,
            "date_max": str(df["date"].max().date()) if pd.notnull(df["date"].max()) else None,
            "unique_products": int(df["description"].nunique()),
        },
        "avg_monthly_growth_rate": float(avg_monthly_growth) if pd.notnull(avg_monthly_growth) else None,
        "outputs": [str(p) for p in out_path.glob("*.csv")]
                + [str(p) for p in out_path.glob("*.json")]
                + [str(p) for p in out_path.glob("*.png")]
    }

if __name__ == "__main__":
    pass
