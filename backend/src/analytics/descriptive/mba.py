# mba.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mlxtend.frequent_patterns import apriori, association_rules

plt.rcParams["figure.dpi"] = 120


@dataclass
class MBAConfig:
    BASE_MIN_SUPPORT_SMALL: float = 0.010   # ~1.0% for <= 50k txns
    BASE_MIN_SUPPORT_LARGE: float = 0.005   # ~0.5% for > 50k txns

    MIN_CONFIDENCE: float = 0.60
    MIN_LIFT: float = 1.0
    MAX_LIFT: float = 20.0

    RULE_COUNT_FRAC: float = 0.001
    RULE_COUNT_MIN: int = 30
    RULE_COUNT_MAX: int = 200

    MIN_RHS_SUPPORT: float = 0.005

    FIGSIZE = (12, 8)
    DPI = 120


def _choose_min_support(n_txns: int, cfg: MBAConfig) -> float:
    base = cfg.BASE_MIN_SUPPORT_LARGE if n_txns > 50_000 else cfg.BASE_MIN_SUPPORT_SMALL
    evidence_floor = max(cfg.MIN_RHS_SUPPORT, np.ceil(cfg.RULE_COUNT_MIN) / max(n_txns, 1))
    return max(base, evidence_floor)


def _rule_count_target(n_txns: int, cfg: MBAConfig) -> int:
    target = int(np.ceil(cfg.RULE_COUNT_FRAC * n_txns))
    return int(np.clip(target, cfg.RULE_COUNT_MIN, cfg.RULE_COUNT_MAX))


def _build_basket(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"receipt", "item_code"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"MBA needs columns {missing}")
    basket = (
        df.groupby(["receipt", "item_code"])["item_code"]
          .count()
          .unstack(fill_value=0)
          .astype(bool)
    )
    return basket


def _filter_rules(rules: pd.DataFrame, n_txns: int, min_support_used: float, cfg: MBAConfig) -> pd.DataFrame:
    if rules.empty:
        return rules
    rules = rules.copy()
    rules["rule_count"] = (rules["support"] * n_txns).round().astype(int)

    if "consequent support" not in rules.columns:
        rules["consequent support"] = np.nan

    rhs_floor = max(cfg.MIN_RHS_SUPPORT, min_support_used)
    min_rule_count = _rule_count_target(n_txns, cfg)

    filt = (
        (rules["support"] >= min_support_used) &
        (rules["confidence"] >= cfg.MIN_CONFIDENCE) &
        (rules["lift"] >= cfg.MIN_LIFT) &
        (rules["lift"] <= cfg.MAX_LIFT) &
        (rules["rule_count"] >= min_rule_count)
    )
    if "consequent support" in rules.columns:
        filt &= (rules["consequent support"].fillna(0) >= rhs_floor)

    return rules.loc[filt].reset_index(drop=True)


def run_mba_pipeline(df: pd.DataFrame, output_dir: str = "analytics_output"):
    os.makedirs(output_dir, exist_ok=True)
    cfg = MBAConfig()

    basket = _build_basket(df)
    n_txns = basket.shape[0]
    min_support = _choose_min_support(n_txns, cfg)
    print(f"✅ Transactions: {n_txns:,} | min_support={min_support:.3%} | min_conf={cfg.MIN_CONFIDENCE}")

    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        print("No frequent itemsets at this support. Lower min_support slightly if needed.")
        return

    rules = association_rules(itemsets, metric="confidence", min_threshold=cfg.MIN_CONFIDENCE)

    if "consequents" in rules.columns and "consequent support" not in rules.columns:
        m = dict(zip(itemsets["itemsets"], itemsets["support"]))
        rules["consequent support"] = rules["consequents"].map(m)

    frules = _filter_rules(rules, n_txns, min_support, cfg)

    itemsets.to_csv(os.path.join(output_dir, "mba_itemsets.csv"), index=False)
    rules.to_csv(os.path.join(output_dir, "mba_rules_all.csv"), index=False)
    frules.to_csv(os.path.join(output_dir, "mba_rules_filtered.csv"), index=False)
    print(f"✅ Rules: {len(rules):,} → {len(frules):,} after filters")

    if not frules.empty:
        plt.figure(figsize=cfg.FIGSIZE)
        sc = plt.scatter(frules["support"], frules["confidence"], c=frules["lift"], alpha=0.85)
        plt.colorbar(sc, label="Lift")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Association Rules: Support vs Confidence (Filtered)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mba_scatter_filtered.png"), dpi=cfg.DPI, bbox_inches="tight")
        plt.close()

        topN = frules.sort_values("lift", ascending=False).head(15)
        labels = (topN["antecedents"].astype(str) + " → " + topN["consequents"].astype(str)).str.wrap(60)
        plt.figure(figsize=(12, 7))
        plt.barh(range(len(topN)), topN["lift"].values)
        plt.yticks(range(len(topN)), labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Lift")
        plt.title("Top 15 Association Rules by Lift (Filtered)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mba_toplift_filtered.png"), dpi=cfg.DPI, bbox_inches="tight")
        plt.close()
    else:
        print("⚠ No rules after filtering. Slightly lower min_support or the min rule count threshold.")

    print("✅ MBA complete → CSV/PNG saved.")
