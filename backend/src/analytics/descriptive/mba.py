# analytics/descriptive/mba.py
import os, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.constrained_layout.use"] = True

def _new_fig(figsize=(18, 12)):
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=10)
    return fig, ax

def _finalize(fig: plt.Figure, path: Path):
    fig.savefig(path, dpi=240)
    print(f"✓ Saved figure: {path}")
    try:
        plt.show()
    finally:
        plt.close(fig)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _to_json(df: pd.DataFrame, path: Path):
    path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["receipt","description","category","tab","qty"]:
        if c not in df.columns: df[c] = np.nan
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df = df[df["qty"].fillna(1) > 0]
    df = df.dropna(subset=["receipt","description"])
    df["receipt"] = df["receipt"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str)
    df["tab"] = df["tab"].astype(str)
    df = df.drop_duplicates(subset=["receipt","description"])
    return df

def _pairs_mba(df: pd.DataFrame, min_support=0.005, min_lift=1.1, max_pairs=5000) -> pd.DataFrame:
    tx = df.groupby("receipt")["description"].apply(list)
    tx = tx[tx.apply(lambda x: len(set(x)) >= 2)]
    n_tx = len(tx)
    if n_tx == 0:
        return pd.DataFrame(columns=[
            "item_a","item_b","support","lift","confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ])

    item_counts = Counter(); pair_counts = Counter()
    for items in tx:
        u = list(set(items))
        item_counts.update(u)
        for a,b in combinations(u[:80], 2):
            pair_counts[tuple(sorted((a,b)))] += 1
            if len(pair_counts) > max_pairs * 50: break

    if not pair_counts:
        return pd.DataFrame(columns=[
            "item_a","item_b","support","lift","confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ])

    items_df = pd.DataFrame([{"item":k, "count":v, "support": v/n_tx} for k,v in item_counts.items()])
    sup_map = items_df.set_index("item")["support"].to_dict()

    meta = (df.groupby(["description"])
              .agg(category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
                   tab=("tab",       lambda s: s.mode().iloc[0] if not s.mode().empty else None))
              .reset_index()
              .rename(columns={"description":"item"})).set_index("item")

    rows = []
    for (a,b), cnt in pair_counts.items():
        sab = cnt / n_tx
        sa = sup_map.get(a, 1e-12); sb = sup_map.get(b, 1e-12)
        lift = sab / (sa * sb) if sa and sb else 0.0
        if sab >= min_support and lift >= min_lift:
            rows.append({
                "item_a": a, "item_b": b,
                "support": sab, "lift": lift,
                "confidence_a_to_b": sab / sa if sa else 0.0,
                "confidence_b_to_a": sab / sb if sb else 0.0,
                "category_a": meta.loc[a, "category"] if a in meta.index else None,
                "tab_a":       meta.loc[a, "tab"]      if a in meta.index else None,
                "category_b": meta.loc[b, "category"] if b in meta.index else None,
                "tab_b":       meta.loc[b, "tab"]      if b in meta.index else None,
            })

    rules = pd.DataFrame(rows)
    if rules.empty: return rules
    return rules.sort_values(["lift","support"], ascending=False).reset_index(drop=True)

def _anchors_view(rules: pd.DataFrame, topn=5) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(columns=["anchor","complement","lift","confidence_pct","support_pct"])
    a_rows = []
    for item in pd.unique(rules[["item_a","item_b"]].values.ravel()):
        sub_a = rules.loc[rules["item_a"] == item, ["item_b","lift","confidence_a_to_b","support"]].rename(
            columns={"item_b":"complement","confidence_a_to_b":"confidence"})
        sub_b = rules.loc[rules["item_b"] == item, ["item_a","lift","confidence_b_to_a","support"]].rename(
            columns={"item_a":"complement","confidence_b_to_a":"confidence"})
        sub = pd.concat([sub_a, sub_b], ignore_index=True).sort_values(["lift","support"], ascending=False).head(topn)
        sub.insert(0, "anchor", item)
        sub["confidence_pct"] = (sub["confidence"] * 100).round(2)
        sub["support_pct"] = (sub["support"] * 100).round(2)
        a_rows.append(sub[["anchor","complement","lift","confidence_pct","support_pct"]])
    return pd.concat(a_rows, ignore_index=True) if a_rows else pd.DataFrame(columns=["anchor","complement","lift","confidence_pct","support_pct"])

def _plot_rules(rules: pd.DataFrame, out_path: Path):
    if rules.empty: return
    top = rules.head(20).copy()
    labels = (top["item_a"] + " + " + top["item_b"]).str.slice(0,60)

    fig, ax = _new_fig(figsize=(18, 12))
    ax.barh(labels.iloc[::-1], top["lift"].iloc[::-1])
    ax.set_title("Top 20 Association Rules by Lift (↑ stronger than chance)")
    ax.set_xlabel("Lift"); ax.set_ylabel("Item Pair")
    _finalize(fig, out_path / "fig_mba_top20_lift.png")

    fig, ax = _new_fig(figsize=(18, 12))
    ax.barh(labels.iloc[::-1], (top["confidence_a_to_b"]*100).iloc[::-1])
    ax.set_title("Top 20 Association Rules by Confidence A→B (percent)")
    ax.set_xlabel("Confidence (%)"); ax.set_ylabel("Item Pair")
    _finalize(fig, out_path / "fig_mba_top20_confidence_a_to_b.png")

def run_mba(df: pd.DataFrame, out_dir: str,
            min_support=None, min_lift=None, topn_anchor=5) -> Dict[str, any]:
    out_path = _ensure_dir(Path(out_dir))
    df = _normalize(df)

    tx_items = df.groupby("receipt")["description"].nunique()
    n_tx = int((tx_items >= 2).sum())
    if n_tx == 0:
        empty = pd.DataFrame(columns=[
            "item_a","item_b","support","lift","confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ])
        (out_path / "mba_rules.csv").write_text(empty.to_csv(index=False), encoding="utf-8")
        _to_json(empty, out_path / "mba_rules.json")
        (out_path / "mba_anchors.csv").write_text(empty.to_csv(index=False), encoding="utf-8")
        _to_json(empty, out_path / "mba_anchors.json")
        return {"transactions": 0, "items": int(df["description"].nunique()), "rules_count": 0, "outputs": []}

    if min_support is None:
        min_support = max(0.0005, min(0.01, 5.0 / n_tx))
    if min_lift is None:
        min_lift = 1.1

    tries = [
        (min_support,       min_lift),
        (min_support * 0.5, 1.05),
        (min_support * 0.25,1.00),
    ]
    rules = pd.DataFrame()
    for sup, lift in tries:
        rules = _pairs_mba(df, min_support=sup, min_lift=lift)
        if not rules.empty: break

    anchors = _anchors_view(rules, topn=topn_anchor) if not rules.empty else pd.DataFrame()

    (out_path / "mba_rules.csv").write_text(rules.to_csv(index=False), encoding="utf-8")
    _to_json(rules, out_path / "mba_rules.json")
    (out_path / "mba_anchors.csv").write_text(anchors.to_csv(index=False), encoding="utf-8")
    _to_json(anchors, out_path / "mba_anchors.json")

    _plot_rules(rules, out_path)

    return {
        "transactions": n_tx,
        "items": int(df["description"].nunique()),
        "rules_count": int(len(rules)),
        "outputs": [str(p) for p in out_path.glob("*.csv")] + [str(p) for p in out_path.glob("*.json")] + [str(p) for p in out_path.glob("*.png")]
    }

if __name__ == "__main__":
    pass
