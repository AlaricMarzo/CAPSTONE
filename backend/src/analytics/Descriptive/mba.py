# analytics/descriptive/mba.py
import os, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.bbox"] = "tight"

FIGSIZE = (10, 8)  # compact but tall enough for labels

def _new_fig(figsize=FIGSIZE, left=0.35, right=0.98, top=0.92, bottom=0.14):
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
    path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["receipt","description","category","tab","qty"]:
        if c not in df.columns: df[c] = np.nan
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df[df["qty"].fillna(1) > 0]
    df = df.dropna(subset=["receipt","description"])
    df["receipt"] = df["receipt"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str)
    df["tab"] = df["tab"].astype(str)
    # Deduplicate items within a receipt to avoid false inflation of support
    return df.drop_duplicates(subset=["receipt","description"])

# -------------------- Core mining --------------------
def _pairs_mba(
    df: pd.DataFrame,
    *,
    min_support=0.005,
    min_lift=1.1,
    min_item_tx=5,     # each item appears in ≥ this many transactions
    min_pair_tx=2,     # pair appears together in ≥ this many transactions
    max_pairs=5000
) -> Tuple[pd.DataFrame, Counter, int]:
    """
    Returns (rules_df, item_counts, n_tx)
    n_tx is the number of valid transactions (receipts with ≥2 distinct items).
    item_counts is a Counter of item -> transaction frequency (on valid transactions).
    """
    # Keep only receipts with ≥ 2 distinct items
    tx = df.groupby("receipt")["description"].apply(list)
    tx = tx[tx.apply(lambda x: len(set(x)) >= 2)]
    n_tx = len(tx)
    if n_tx == 0:
        return pd.DataFrame(columns=[
            "item_a","item_b","pair_tx","support","lift",
            "confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ]), Counter(), 0

    # Item counts (by transaction)
    item_counts = Counter()
    for items in tx:
        item_counts.update(set(items))

    # Item frequency guard
    allowed_items: Set[str] = {it for it, cnt in item_counts.items() if cnt >= min_item_tx}
    if not allowed_items:
        return pd.DataFrame(columns=[
            "item_a","item_b","pair_tx","support","lift",
            "confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ]), item_counts, n_tx

    # Pair counts (only for allowed items)
    pair_counts = Counter()
    for items in tx:
        u = sorted(set(it for it in items if it in allowed_items))
        for a, b in combinations(u, 2):
            pair_counts[(a, b)] += 1
            if len(pair_counts) > max_pairs:
                break

    if not pair_counts:
        return pd.DataFrame(columns=[
            "item_a","item_b","pair_tx","support","lift",
            "confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b"
        ]), item_counts, n_tx

    # Item supports (probabilities)
    items_df = pd.DataFrame(
        [{"item": k, "tx": v, "support": v / n_tx} for k, v in item_counts.items() if k in allowed_items]
    )
    sup_map = items_df.set_index("item")["support"].to_dict()

    # Meta (category/tab) for labeling
    meta = (df.groupby(["description"])
              .agg(category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
                   tab=("tab",       lambda s: s.mode().iloc[0] if not s.mode().empty else None))
              .reset_index()
              .rename(columns={"description": "item"})
            ).set_index("item")

    # Build rules with thresholds
    rows = []
    for (a, b), cnt in pair_counts.items():
        if cnt < min_pair_tx:
            continue
        sab = cnt / n_tx
        if sab < min_support:
            continue
        sa = sup_map.get(a, 0.0); sb = sup_map.get(b, 0.0)
        if sa <= 0 or sb <= 0:
            continue
        lift = sab / (sa * sb)
        if lift < min_lift:
            continue
        rows.append({
            "item_a": a, "item_b": b, "pair_tx": int(cnt),
            "support": sab, "lift": lift,
            "confidence_a_to_b": sab / sa,
            "confidence_b_to_a": sab / sb,
            "category_a": meta.loc[a, "category"] if a in meta.index else None,
            "tab_a":       meta.loc[a, "tab"]      if a in meta.index else None,
            "category_b": meta.loc[b, "category"] if b in meta.index else None,
            "tab_b":       meta.loc[b, "tab"]      if b in meta.index else None,
        })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules, item_counts, n_tx

    # Convenience fields for frontend
    rules["support_pct"] = (rules["support"] * 100).round(2)
    rules["confidence_ab_pct"] = (rules["confidence_a_to_b"] * 100).round(2)
    rules["confidence_ba_pct"] = (rules["confidence_b_to_a"] * 100).round(2)
    rules["either_confidence_max"] = rules[["confidence_a_to_b","confidence_b_to_a"]].max(axis=1)

    rules = rules.sort_values(["lift","support","pair_tx"], ascending=False).reset_index(drop=True)
    return rules, item_counts, n_tx

def _anchors_view(rules: pd.DataFrame, topn=5) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(columns=["anchor","complement","lift","confidence_pct","support_pct","pair_tx"])
    a_rows = []
    for item in pd.unique(rules[["item_a","item_b"]].values.ravel()):
        sub_a = rules.loc[rules["item_a"] == item, ["item_b","lift","confidence_a_to_b","support","pair_tx"]].rename(
            columns={"item_b":"complement","confidence_a_to_b":"confidence"})
        sub_b = rules.loc[rules["item_b"] == item, ["item_a","lift","confidence_b_to_a","support","pair_tx"]].rename(
            columns={"item_a":"complement","confidence_b_to_a":"confidence"})
        sub = pd.concat([sub_a, sub_b], ignore_index=True)\
                .sort_values(["lift","support","pair_tx"], ascending=False).head(topn)
        sub.insert(0, "anchor", item)
        sub["confidence_pct"] = (sub["confidence"] * 100).round(2)
        sub["support_pct"] = (sub["support"] * 100).round(2)
        a_rows.append(sub[["anchor","complement","lift","confidence_pct","support_pct","pair_tx"]])
    return pd.concat(a_rows, ignore_index=True) if a_rows else pd.DataFrame(
        columns=["anchor","complement","lift","confidence_pct","support_pct","pair_tx"])

def _plot_rules(rules: pd.DataFrame, out_path: Path):
    if rules.empty:
        return

    # -------- LIFT: take top 20 by lift --------
    top_lift = rules.sort_values("lift", ascending=False).head(20).copy()
    labels_lift = (top_lift["item_a"] + " + " + top_lift["item_b"]).str.slice(0, 60)
    fig, ax = _new_fig()
    # reverse to show highest at the TOP
    ax.barh(labels_lift.iloc[::-1], top_lift["lift"].iloc[::-1])
    ax.set_title("Top 20 Association Rules by Lift (↑ stronger than chance)")
    ax.set_xlabel("Lift"); ax.set_ylabel("Item Pair")
    _finalize(fig, out_path / "fig_mba_top20_lift.png")

    # -------- CONFIDENCE A→B: take top 20 by confidence_a_to_b --------
    top_conf = rules.sort_values("confidence_a_to_b", ascending=False).head(20).copy()
    labels_conf = (top_conf["item_a"] + " → " + top_conf["item_b"]).str.slice(0, 60)
    fig, ax = _new_fig()
    ax.barh(labels_conf.iloc[::-1], (top_conf["confidence_a_to_b"] * 100).iloc[::-1])
    ax.set_title("Top 20 Association Rules by Confidence A→B (percent)")
    ax.set_xlabel("Confidence (%)"); ax.set_ylabel("Item Pair")
    _finalize(fig, out_path / "fig_mba_top20_confidence_a_to_b.png")

# -------------------- Public API --------------------
def run_mba(
    df: pd.DataFrame,
    out_dir: str,
    *,
    min_support=None,
    min_lift=None,
    min_item_tx=5,
    min_pair_tx=2,
    topn_anchor=5
) -> Dict[str, Any]:

    out_path = _ensure_dir(Path(out_dir))
    df = _normalize(df)

    # Valid receipts = those with ≥2 distinct items
    tx_items = df.groupby("receipt")["description"].nunique()
    valid_receipts = set(tx_items[tx_items >= 2].index)
    n_tx = int(len(valid_receipts))
    if n_tx == 0:
        empty = pd.DataFrame(columns=[
            "item_a","item_b","pair_tx","support","lift",
            "confidence_a_to_b","confidence_b_to_a",
            "category_a","tab_a","category_b","tab_b",
            "support_pct","confidence_ab_pct","confidence_ba_pct","either_confidence_max"
        ])
        (out_path / "mba_rules.csv").write_text(empty.to_csv(index=False), encoding="utf-8")
        _to_json(empty, out_path / "mba_rules.json")
        (out_path / "mba_anchors.csv").write_text(empty.to_csv(index=False), encoding="utf-8")
        _to_json(empty, out_path / "mba_anchors.json")
        # also write items/filters/manifest for the frontend to mount gracefully
        (out_path / "mba_items.json").write_text("[]", encoding="utf-8")
        (out_path / "mba_filters.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
        (out_path / "mba_manifest.json").write_text(json.dumps({"transactions":0,"items":0,"rules_count":0}, indent=2), encoding="utf-8")
        return {"transactions": 0, "items": 0, "rules_count": 0, "outputs": []}

    if min_support is None:
        min_support = max(0.0005, min(0.01, 5.0 / n_tx))
    if min_lift is None:
        min_lift = 1.1

    # Mine rules
    rules, item_counts, n_tx2 = _pairs_mba(
        df[df["receipt"].isin(valid_receipts)],
        min_support=min_support,
        min_lift=min_lift,
        min_item_tx=min_item_tx,
        min_pair_tx=min_pair_tx
    )
    n_tx = n_tx2  # keep consistent with what _pairs_mba used

    # If empty, relax gradually but keep pair_tx>=2 and item_tx>=3 to avoid one-offs
    if rules.empty:
        for params in [
            dict(min_support=min_support*0.5, min_lift=max(1.05, min_lift*0.95), min_item_tx=max(3, min_item_tx-2), min_pair_tx=2),
            dict(min_support=min_support*0.25, min_lift=1.00,                  min_item_tx=max(3, min_item_tx-2), min_pair_tx=2),
        ]:
            rules, item_counts, _ = _pairs_mba(df[df["receipt"].isin(valid_receipts)], **params)
            if not rules.empty:
                # update thresholds to reflect what produced the rules
                min_support, min_lift = params["min_support"], params["min_lift"]
                min_item_tx, min_pair_tx = params["min_item_tx"], params["min_pair_tx"]
                break

    anchors = _anchors_view(rules, topn=topn_anchor) if not rules.empty else pd.DataFrame()

    # Item list for frontend (anchors dropdown/search)
    items_rows = []
    if n_tx > 0:
        for item, tx_cnt in item_counts.items():
            items_rows.append({"item": item, "item_tx": int(tx_cnt), "support_pct": round(tx_cnt / n_tx * 100, 2)})
    items_df = pd.DataFrame(items_rows)
    if not items_df.empty:
        # Attach category/tab using modal values from df
        meta = (df.groupby(["description"])
                  .agg(category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
                       tab=("tab",       lambda s: s.mode().iloc[0] if not s.mode().empty else None))
                  .reset_index()
                  .rename(columns={"description":"item"}))
        items_df = items_df.merge(meta, on="item", how="left")
        items_df = items_df[["item","category","tab","item_tx","support_pct"]].sort_values(["item_tx","support_pct","item"], ascending=[False, False, True])

    # Distinct categories/tabs seen in rules (for quick dropdown fill)
    cats = sorted(pd.unique(pd.concat([rules["category_a"], rules["category_b"]], ignore_index=True).dropna().astype(str))) if not rules.empty else []
    tabs = sorted(pd.unique(pd.concat([rules["tab_a"], rules["tab_b"]], ignore_index=True).dropna().astype(str))) if not rules.empty else []
    anchors_list = sorted(pd.unique(rules[["item_a","item_b"]].values.ravel())) if not rules.empty else []

    # Save artifacts
    (out_path / "mba_rules.csv").write_text(rules.to_csv(index=False), encoding="utf-8")
    _to_json(rules, out_path / "mba_rules.json")
    (out_path / "mba_anchors.csv").write_text(anchors.to_csv(index=False), encoding="utf-8")
    _to_json(anchors, out_path / "mba_anchors.json")

    # NEW helper JSONs for frontend
    _to_json(items_df if not items_df.empty else pd.DataFrame(columns=["item","category","tab","item_tx","support_pct"]),
             out_path / "mba_items.json")
    Path(out_path / "mba_filters.json").write_text(
        json.dumps(
            {
                "min_support": min_support,
                "min_lift": min_lift,
                "min_item_tx": int(min_item_tx),
                "min_pair_tx": int(min_pair_tx)
            }, indent=2
        ),
        encoding="utf-8"
    )
    Path(out_path / "mba_manifest.json").write_text(
        json.dumps(
            {
                "transactions": n_tx,
                "items": int(df["description"].nunique()),
                "rules_count": int(len(rules)),
                "categories": cats,
                "tabs": tabs,
                "anchors": anchors_list
            }, indent=2
        ),
        encoding="utf-8"
    )

    # Plots (compact)
    _plot_rules(rules, out_path)

    return {
        "transactions": n_tx,
        "items": int(df["description"].nunique()),
        "rules_count": int(len(rules)),
        "outputs": [str(p) for p in out_path.glob("*.csv")]
                + [str(p) for p in out_path.glob("*.json")]
                + [str(p) for p in out_path.glob("*.png")]
    }

if __name__ == "__main__":
    pass
