# mba.py
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

@dataclass
class MBAConfig:
    SMALL_DATASET_THRESHOLD: int = 50_000
    MIN_SUPPORT_SMALL: float = 0.01
    MIN_SUPPORT_LARGE: float = 0.001
    MIN_CONFIDENCE: float = 0.30
    MIN_LIFT: float = 1.20
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 100

def prepare_transactions(df: pd.DataFrame):
    if "description" in df.columns:
        tx = df.groupby("receipt")["description"].apply(list).values
        print("✓ Using item descriptions for MBA")
    else:
        tx = df.groupby("receipt")["item_code"].apply(list).values
        print("⚠ Using item codes (no descriptions)")
    print(f"✓ Prepared {len(tx):,} transactions")
    return tx

def run_mba(transactions, cfg: MBAConfig, dataset_size: int):
    min_support = cfg.MIN_SUPPORT_SMALL if dataset_size < cfg.SMALL_DATASET_THRESHOLD else cfg.MIN_SUPPORT_LARGE
    print(f"FP-Growth: min_support={min_support}, min_conf={cfg.MIN_CONFIDENCE}, min_lift={cfg.MIN_LIFT}")
    te = TransactionEncoder()
    onehot = te.fit(transactions).transform(transactions)
    df_enc = pd.DataFrame(onehot, columns=te.columns_)
    itemsets = fpgrowth(df_enc, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        print("✗ No frequent itemsets—try lowering min_support.")
        return None, None
    rules = association_rules(itemsets, metric="confidence", min_threshold=cfg.MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= cfg.MIN_LIFT].sort_values("lift", ascending=False)
    print(f"✓ {len(itemsets):,} itemsets, {len(rules):,} rules")
    return itemsets, rules

def save_mba_outputs(itemsets, rules, output_dir: str):
    if itemsets is None or rules is None:
        print("⚠ No MBA results to save")
        return
    os.makedirs(output_dir, exist_ok=True)
    is_export = itemsets.copy()
    is_export["items"] = is_export["itemsets"].apply(lambda s: " | ".join(list(s)))
    is_export["itemset_size"] = is_export["itemsets"].apply(len)
    is_export = is_export[["items", "itemset_size", "support"]].sort_values("support", ascending=False)
    is_export.to_csv(os.path.join(output_dir, "mba_frequent_itemsets.csv"), index=False)

    r_export = pd.DataFrame({
        "antecedents": rules["antecedents"].apply(lambda s: " | ".join(list(s))),
        "consequents": rules["consequents"].apply(lambda s: " | ".join(list(s))),
        "support": rules["support"].astype(float).round(4),
        "confidence": rules["confidence"].astype(float).round(4),
        "lift": rules["lift"].astype(float).round(2),
        "leverage": rules["leverage"].astype(float).round(4),
        "conviction": rules["conviction"].astype(float).round(2)
    }).sort_values("lift", ascending=False)
    r_export.to_csv(os.path.join(output_dir, "mba_association_rules.csv"), index=False)
    print("✓ MBA CSVs saved")

def create_mba_visualizations(rules: pd.DataFrame, output_dir: str, cfg: MBAConfig):
    if rules is None or rules.empty:
        print("⚠ No rules to visualize")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Support vs Confidence
    plt.figure(figsize=cfg.FIGURE_SIZE)
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.6, s=100)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support"); plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mba_support_confidence.png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()

    # Top rules by Lift
    plt.figure(figsize=cfg.FIGURE_SIZE)
    top = rules.nlargest(15, "lift").copy()
    top["rule"] = top.apply(lambda r: f"{list(r['antecedents'])[0][:25]}... → {list(r['consequents'])[0][:25]}...", axis=1)
    plt.barh(range(len(top)), top["lift"].values)
    plt.yticks(range(len(top)), top["rule"].values, fontsize=8)
    plt.xlabel("Lift"); plt.title("Top 15 Association Rules by Lift")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mba_top_rules.png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    print("✓ MBA charts saved")

def run_mba_pipeline(df: pd.DataFrame, output_dir: str = "analytics_output"):
    cfg = MBAConfig()
    tx = prepare_transactions(df)
    itemsets, rules = run_mba(tx, cfg, len(df))
    save_mba_outputs(itemsets, rules, output_dir)
    create_mba_visualizations(rules, output_dir, cfg)
    print("✓ MBA complete → CSV/PNG saved.")
