# -*- coding: utf-8 -*-
"""
Enhanced Pharmacy Sales Analysis with Prescriptive Models (IMPROVED)

Improvements:
- Robust error handling and data validation
- Accurate EOQ calculation without overly strict constraints
- Better reorder point forecasting with fallback strategies
- Normalized linear programming objective function
- Adaptive anomaly detection based on dataset size
- Configurable resource planning parameters
- Comprehensive input validation
- Better handling of edge cases and missing data
- COMPREHENSIVE CSV EXPORTS FOR ALL 8 MODELS
"""

import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import linprog, minimize
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

# Define the output directory
output_dir = "prescriptive_output"
os.makedirs(output_dir, exist_ok=True)

# Configuration constants
REQUIRED_COLUMNS = ['date', 'description', 'qty', 'sales', 'cost']
MIN_DATA_POINTS = 30
ORDERING_COST = 150.0
HOLDING_COST_PERCENT = 0.25
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
PLANNING_HORIZON_DAYS = 90


def validate_dataframe(df, required_cols):
    """
    Validate that dataframe has required columns and sufficient data
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {MIN_DATA_POINTS}")
    
    return True

def to_numeric_safe(series, column_name=""):
    """Convert to numeric with comprehensive error handling"""
    try:
        result = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )
        null_count = result.isna().sum()
        if null_count > len(result) * 0.5:
            print(f"Warning: {column_name} has {null_count} null values ({null_count/len(result)*100:.1f}%)")
        return result
    except Exception as e:
        print(f"Error converting {column_name}: {e}")
        return pd.Series([np.nan] * len(series))

def select_csv_file():
    """Automatically select the CSV file in the current directory"""
    csv_files = list(Path(".").glob("*.csv"))
    if not csv_files:
        print("No CSV files found in current directory. Exiting...")
        sys.exit(1)

    csv_path = csv_files[0]
    print(f"Using CSV file: {csv_path}")
    return csv_path

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

print("=" * 80)
print("PHARMACY SALES PRESCRIPTIVE ANALYSIS (IMPROVED)")
print("=" * 80)

# Select and load CSV file
csv_path = select_csv_file()
print(f"\nSelected file: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]


print("\nCleaning and validating data...")

try:
    validate_dataframe(df, REQUIRED_COLUMNS)
except ValueError as e:
    print(f"Validation error: {e}")
    sys.exit(1)

try:
    df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    if df['date'].isna().sum() > 0:
        print(f"Warning: {df['date'].isna().sum()} invalid dates found and removed")
        df = df.dropna(subset=['date'])
except Exception as e:
    print(f"Error processing dates: {e}")
    sys.exit(1)

if 'expiration' in df.columns:
    df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce', infer_datetime_format=True)

numeric_cols = ['qty', 'sales', 'cost', 'profit', 'payment', 'discount', 'receipt', 'so']
for col in numeric_cols:
    if col in df.columns:
        df[col] = to_numeric_safe(df[col], col)
        df[col] = df[col].fillna(0)

if 'profit' not in df.columns or df['profit'].isna().sum() > len(df) * 0.5:
    df['profit'] = df['sales'] - df['cost']
    print("✓ Calculated profit from sales - cost")

if 'description' in df.columns:
    df['description'] = df['description'].astype(str).str.strip()
else:
    print("Error: 'description' column required")
    sys.exit(1)

df['medicine'] = df['description']

df['customer_group'] = np.where(
    (df.get('discount', 0).fillna(0) > 0) | (df.get('payment', '').astype(str).str.contains('Senior|PWD', case=False, na=False)),
    'Senior/PWD',
    'Regular'
)

df['week'] = df['date'].dt.to_period('W')
df['month'] = df['date'].dt.to_period('M')

df = df[(df['qty'] > 0) & (df['sales'] > 0)].copy()

print(f"✓ Data cleaned: {len(df)} valid rows")

medicine_stats = df.groupby('medicine').agg({
    'qty': ['sum', 'count'],
    'sales': 'sum',
    'cost': 'sum',
    'profit': 'sum'
}).reset_index()

medicine_stats.columns = ['medicine', 'total_qty', 'transaction_count', 'total_sales', 'total_cost', 'total_profit']
medicine_stats = medicine_stats[medicine_stats['total_qty'] > 0].copy()
medicine_stats['avg_unit_price'] = medicine_stats['total_sales'] / medicine_stats['total_qty']
medicine_stats['profit_margin'] = (medicine_stats['total_profit'] / medicine_stats['total_sales'] * 100).fillna(0)
medicine_stats['profit_margin'] = medicine_stats['profit_margin'].clip(lower=-100, upper=100)

top_products = medicine_stats.nlargest(20, 'total_qty')['medicine'].tolist()
print(f"✓ Analyzing top {len(top_products)} products")

# ========== MODEL 1: FORECAST-DRIVEN REORDER POINT ==========
print("\n" + "=" * 80)
print("MODEL 1: FORECAST-DRIVEN REORDER POINT WITH SAFETY STOCK")
print("=" * 80)

def calculate_reorder_point(medicine_name, df, lead_time_days=LEAD_TIME_DAYS, service_level=SERVICE_LEVEL):
    """
    Calculate reorder point with safety stock based on demand forecast
    """
    med_data = df[df['medicine'] == medicine_name].copy()
    
    if len(med_data) < 30:
        return None
    
    try:
        daily_demand = med_data.groupby('date')['qty'].sum().reset_index()
        daily_demand = daily_demand.set_index('date').resample('D').sum().fillna(0)
        
        avg_daily_demand = daily_demand['qty'].mean()
        std_daily_demand = daily_demand['qty'].std()
        
        if std_daily_demand < 0.1:
            std_daily_demand = avg_daily_demand * 0.1
        
        z_score = stats.norm.ppf(service_level)
        
        safety_stock = z_score * std_daily_demand * math.sqrt(lead_time_days)
        
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        forecast_demand = avg_daily_demand * 30
        
        try:
            if len(daily_demand) >= 14:
                model = ExponentialSmoothing(
                    daily_demand['qty'],
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add',
                    initialization_method='estimated'
                )
                fitted = model.fit(optimized=True)
                forecast = fitted.forecast(steps=30)
                forecast_demand = max(forecast.sum(), avg_daily_demand * 30 * 0.5)
        except Exception as e:
            try:
                model = ExponentialSmoothing(daily_demand['qty'], trend='add')
                fitted = model.fit(optimized=True)
                forecast = fitted.forecast(steps=30)
                forecast_demand = forecast.sum()
            except:
                pass
        
        return {
            'medicine': medicine_name,
            'avg_daily_demand': avg_daily_demand,
            'std_daily_demand': std_daily_demand,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'forecast_30day': forecast_demand,
            'service_level': service_level * 100,
            'lead_time_days': lead_time_days
        }
    except Exception as e:
        print(f"Error calculating ROP for {medicine_name}: {e}")
        return None

reorder_points = []
for med in top_products:
    result = calculate_reorder_point(med, df)
    if result:
        reorder_points.append(result)

rop_df = pd.DataFrame(reorder_points)

if not rop_df.empty:
    print("\nReorder Point Analysis (Top 10):")
    print(rop_df.head(10)[['medicine', 'avg_daily_demand', 'safety_stock', 'reorder_point', 'forecast_30day']].to_string(index=False))

    rop_df.to_csv(os.path.join(output_dir, 'model_1_reorder_point.csv'), index=False)
    print(f"✓ Saved: model_1_reorder_point.csv ({len(rop_df)} records)")

    # Visualization
    plt.figure(figsize=(14, 8))
    top_10_rop = rop_df.nlargest(10, 'reorder_point')
    x = np.arange(len(top_10_rop))
    width = 0.35

    plt.bar(x - width/2, top_10_rop['reorder_point'] - top_10_rop['safety_stock'], width, label='Base Stock', color='steelblue')
    plt.bar(x + width/2, top_10_rop['safety_stock'], width, label='Safety Stock', color='coral')

    plt.xlabel('Medicine')
    plt.ylabel('Quantity')
    plt.title('Reorder Point with Safety Stock (Top 10 Products)', fontsize=16, pad=20)
    plt.xticks(x, top_10_rop['medicine'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_1_reorder_point_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========== MODEL 2: ECONOMIC ORDER QUANTITY ==========
print("\n" + "=" * 80)
print("MODEL 2: ECONOMIC ORDER QUANTITY (EOQ)")
print("=" * 80)

def calculate_eoq(annual_demand, ordering_cost, holding_cost_percent, unit_cost):
    """
    Calculate Economic Order Quantity with improved validation
    """
    try:
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_percent <= 0 or unit_cost <= 0:
            return np.nan
        
        holding_cost = holding_cost_percent * float(unit_cost)
        
        if holding_cost <= 0:
            return np.nan
        
        eoq = math.sqrt((2 * float(annual_demand) * float(ordering_cost)) / holding_cost)
        
        if eoq > annual_demand * 5:
            print(f"Warning: EOQ ({eoq:.0f}) is very high relative to annual demand ({annual_demand:.0f})")
        
        return eoq
    
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        print(f"Error calculating EOQ: {e}")
        return np.nan

eoq_results = []
for _, row in medicine_stats[medicine_stats['medicine'].isin(top_products)].iterrows():
    annual_demand = row['total_qty']
    unit_cost = row['avg_unit_price']
    
    if pd.notna(unit_cost) and unit_cost > 0:
        eoq = calculate_eoq(annual_demand, ORDERING_COST, HOLDING_COST_PERCENT, unit_cost)
        
        if pd.notna(eoq) and eoq > 0:
            orders_per_year = annual_demand / eoq
            days_between_orders = 365 / orders_per_year if orders_per_year > 0 else 365
            
            annual_ordering_cost = orders_per_year * ORDERING_COST
            annual_holding_cost = (eoq / 2) * (HOLDING_COST_PERCENT * unit_cost)
            annual_purchase_cost = annual_demand * unit_cost
            total_annual_cost = annual_ordering_cost + annual_holding_cost + annual_purchase_cost
            
            eoq_results.append({
                'medicine': row['medicine'],
                'annual_demand': annual_demand,
                'unit_cost': unit_cost,
                'eoq': eoq,
                'orders_per_year': orders_per_year,
                'days_between_orders': days_between_orders,
                'annual_ordering_cost': annual_ordering_cost,
                'annual_holding_cost': annual_holding_cost,
                'annual_purchase_cost': annual_purchase_cost,
                'total_annual_cost': total_annual_cost
            })

eoq_df = pd.DataFrame(eoq_results).sort_values('annual_demand', ascending=False)

if not eoq_df.empty:
    print("\nEOQ Analysis (Top 10):")
    print(eoq_df.head(10)[['medicine', 'annual_demand', 'eoq', 'orders_per_year', 'days_between_orders']].to_string(index=False))

    eoq_df.to_csv(os.path.join(output_dir, 'model_2_eoq.csv'), index=False)
    print(f"✓ Saved: model_2_eoq.csv ({len(eoq_df)} records)")

    # Visualization
    plt.figure(figsize=(14, 8))
    top_10_eoq = eoq_df.head(10)
    sns.barplot(data=top_10_eoq, x='eoq', y='medicine', palette='viridis')
    plt.title('Economic Order Quantity (EOQ) for Top 10 Products', fontsize=16, pad=20)
    plt.xlabel('EOQ (Units)')
    plt.ylabel('Medicine')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_2_eoq_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========== MODEL 3: LINEAR PROGRAMMING FOR INVENTORY ALLOCATION ==========
print("\n" + "=" * 80)
print("MODEL 3: LINEAR PROGRAMMING FOR INVENTORY ALLOCATION")
print("=" * 80)

def optimize_inventory_allocation(products_df, budget_constraint, storage_constraint):
    """
    Optimize inventory allocation using linear programming
    """
    try:
        products_df = products_df.copy()
        n = len(products_df)
        
        if n == 0:
            return None
        
        profit_margin_normalized = (products_df['profit_margin'] - products_df['profit_margin'].min()) / (products_df['profit_margin'].max() - products_df['profit_margin'].min() + 1e-6)
        
        c = -profit_margin_normalized.values
        
        A_ub = np.array([
            products_df['avg_unit_price'].values,
            np.ones(n)
        ])
        b_ub = np.array([budget_constraint, storage_constraint])
        
        bounds = [(0, None) for _ in range(n)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            allocation = result.x
            products_df['optimal_allocation'] = allocation
            products_df['allocated_value'] = allocation * products_df['avg_unit_price']
            products_df['expected_profit'] = allocation * (products_df['profit_margin'] / 100) * products_df['avg_unit_price']
            
            return products_df[products_df['optimal_allocation'] > 0.1].sort_values('optimal_allocation', ascending=False)
        else:
            print(f"Optimization failed: {result.message}")
            return None
    except Exception as e:
        print(f"Error in inventory allocation: {e}")
        return None

total_budget = medicine_stats['total_sales'].sum() * 0.6
storage_capacity = medicine_stats['total_qty'].sum() * 0.8

allocation_df = optimize_inventory_allocation(
    medicine_stats[medicine_stats['medicine'].isin(top_products)].copy(),
    total_budget,
    storage_capacity
)

if allocation_df is not None and not allocation_df.empty:
    print(f"\nOptimal Inventory Allocation (Budget: ₱{total_budget:,.2f}, Storage: {storage_capacity:,.0f} units):")
    print(allocation_df.head(10)[['medicine', 'optimal_allocation', 'allocated_value', 'expected_profit']].to_string(index=False))

    allocation_export = allocation_df[['medicine', 'total_qty', 'avg_unit_price', 'profit_margin', 'optimal_allocation', 'allocated_value', 'expected_profit']].copy()
    allocation_export.to_csv(os.path.join(output_dir, 'model_3_inventory_allocation.csv'), index=False)
    print(f"✓ Saved: model_3_inventory_allocation.csv ({len(allocation_export)} records)")

    # Visualization
    plt.figure(figsize=(14, 8))
    top_10_alloc = allocation_df.head(10)
    sns.barplot(data=top_10_alloc, x='optimal_allocation', y='medicine', palette='rocket')
    plt.title('Optimal Inventory Allocation (Linear Programming)', fontsize=16, pad=20)
    plt.xlabel('Allocated Quantity (Units)')
    plt.ylabel('Medicine')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_3_inventory_allocation_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("\nInventory allocation optimization could not be completed.")
    allocation_df = pd.DataFrame()

# ========== MODEL 4: WHAT-IF ANALYSIS ==========
print("\n" + "=" * 80)
print("MODEL 4: WHAT-IF ANALYSIS FOR INVENTORY PLANNING")
print("=" * 80)

def what_if_analysis(medicine_name, df, scenarios):
    """
    Perform what-if analysis for different demand scenarios
    """
    med_data = df[df['medicine'] == medicine_name]
    
    if len(med_data) == 0:
        return None
    
    current_demand = med_data['qty'].sum()
    current_sales = med_data['sales'].sum()
    current_profit = med_data['profit'].sum()
    
    if current_demand <= 0 or current_sales <= 0:
        return None
    
    avg_price = current_sales / current_demand
    
    cost_per_unit = (current_sales - current_profit) / current_demand
    cost_per_unit = max(cost_per_unit, 0)
    
    results = []
    for scenario_name, demand_change, price_change in scenarios:
        new_demand = current_demand * (1 + demand_change)
        new_price = avg_price * (1 + price_change)
        new_sales = new_demand * new_price
        new_profit = new_sales - (new_demand * cost_per_unit)
        
        profit_change = ((new_profit - current_profit) / abs(current_profit)) * 100 if current_profit != 0 else 0
        
        results.append({
            'medicine': medicine_name,
            'scenario': scenario_name,
            'demand_change_pct': demand_change * 100,
            'price_change_pct': price_change * 100,
            'projected_demand': new_demand,
            'projected_sales': new_sales,
            'projected_profit': new_profit,
            'projected_cost': new_sales - new_profit,
            'profit_change_pct': profit_change
        })
    
    return pd.DataFrame(results)

scenarios = [
    ("Pessimistic", -0.20, -0.10),
    ("Conservative", -0.10, 0.00),
    ("Current", 0.00, 0.00),
    ("Optimistic", 0.15, 0.05),
    ("Aggressive", 0.30, 0.10),
]

whatif_all_results = []
for top_medicine in top_products[:10]:
    whatif_df = what_if_analysis(top_medicine, df, scenarios)
    if whatif_df is not None:
        whatif_all_results.append(whatif_df)

if whatif_all_results:
    whatif_combined = pd.concat(whatif_all_results, ignore_index=True)
    whatif_combined.to_csv(os.path.join(output_dir, 'model_4_whatif_analysis.csv'), index=False)
    print(f"✓ Saved: model_4_whatif_analysis.csv ({len(whatif_combined)} records)")
    
    # Show sample for top product
    if whatif_all_results:
        print(f"\nWhat-If Analysis for: {top_products[0]}")
        print(whatif_all_results[0][['scenario', 'demand_change_pct', 'price_change_pct', 'projected_sales', 'projected_profit', 'profit_change_pct']].to_string(index=False))
    
    # Visualization for top product
    if whatif_all_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sample_whatif = whatif_all_results[0]
        ax1.bar(sample_whatif['scenario'], sample_whatif['projected_sales'], color=['red', 'orange', 'gray', 'lightgreen', 'green'])
        ax1.set_title(f'Sales Projections - {top_products[0]}', fontsize=14)
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Projected Sales (₱)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(sample_whatif['scenario'], sample_whatif['projected_profit'], color=['red', 'orange', 'gray', 'lightgreen', 'green'])
        ax2.set_title(f'Profit Projections - {top_products[0]}', fontsize=14)
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Projected Profit (₱)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_4_whatif_analysis_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ========== MODEL 5: DISCOUNT OPTIMIZATION ==========
print("\n" + "=" * 80)
print("MODEL 5: DISCOUNT OPTIMIZATION ENGINE")
print("=" * 80)

def optimize_discount_strategy(df):
    """
    Analyze discount effectiveness and optimize discount strategy
    """
    try:
        discount_analysis = df.groupby('customer_group').agg({
            'qty': 'sum',
            'sales': 'sum',
            'profit': 'sum',
            'discount': 'mean'
        }).reset_index()
        
        discount_analysis['avg_discount_pct'] = (discount_analysis['discount'] / discount_analysis['sales'] * 100).fillna(0)
        discount_analysis['profit_margin'] = (discount_analysis['profit'] / discount_analysis['sales'] * 100).fillna(0)
        
        product_discount = df[df['discount'] > 0].groupby('medicine').agg({
            'qty': 'sum',
            'sales': 'sum',
            'profit': 'sum',
            'discount': 'sum'
        }).reset_index()
        
        if len(product_discount) > 0:
            product_discount['discount_pct'] = (product_discount['discount'] / (product_discount['sales'] + product_discount['discount']) * 100).fillna(0)
            product_discount['profit_margin'] = (product_discount['profit'] / product_discount['sales'] * 100).fillna(0)
            product_discount['discount_efficiency'] = (product_discount['qty'] / (product_discount['discount'] + 1e-6)).fillna(0)
        
        return discount_analysis, product_discount.sort_values('discount_efficiency', ascending=False)
    except Exception as e:
        print(f"Error in discount optimization: {e}")
        return None, None

group_discount, product_discount = optimize_discount_strategy(df)

if group_discount is not None:
    print("\nDiscount Analysis by Customer Group:")
    print(group_discount[['customer_group', 'qty', 'sales', 'profit', 'avg_discount_pct', 'profit_margin']].to_string(index=False))
    
    group_discount.to_csv(os.path.join(output_dir, 'model_5_discount_by_group.csv'), index=False)
    print(f"✓ Saved: model_5_discount_by_group.csv ({len(group_discount)} records)")

if product_discount is not None and not product_discount.empty:
    print("\nTop 10 Products by Discount Efficiency:")
    print(product_discount.head(10)[['medicine', 'qty', 'discount_pct', 'profit_margin', 'discount_efficiency']].to_string(index=False))
    
    product_discount.to_csv(os.path.join(output_dir, 'model_5_discount_by_product.csv'), index=False)
    print(f"✓ Saved: model_5_discount_by_product.csv ({len(product_discount)} records)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(product_discount['discount_pct'], product_discount['profit_margin'], alpha=0.6, s=100)
    ax1.set_xlabel('Discount %')
    ax1.set_ylabel('Profit Margin %')
    ax1.set_title('Discount vs Profit Margin', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    top_10_disc = product_discount.head(10)
    ax2.barh(top_10_disc['medicine'], top_10_disc['discount_efficiency'], color='teal')
    ax2.set_xlabel('Discount Efficiency (Qty per ₱ Discount)')
    ax2.set_ylabel('Medicine')
    ax2.set_title('Top 10 Discount Efficient Products', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_5_discount_optimization_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========== MODEL 6: RESOURCE PLANNING ==========
print("\n" + "=" * 80)
print("MODEL 6: RESOURCE PLANNING FOR SUPPLIES")
print("=" * 80)

def resource_planning(df, planning_horizon_days=PLANNING_HORIZON_DAYS, unit_volume=1.0):
    """
    Plan resources needed for next planning period
    """
    try:
        date_range = (df['date'].max() - df['date'].min()).days
        if date_range <= 0:
            date_range = 1
        
        resource_plan = df.groupby('medicine').agg({
            'qty': 'sum',
            'sales': 'sum'
        }).reset_index()
        
        resource_plan['daily_demand'] = resource_plan['qty'] / date_range
        resource_plan['projected_demand'] = resource_plan['daily_demand'] * planning_horizon_days
        resource_plan['projected_revenue'] = (resource_plan['sales'] / resource_plan['qty']) * resource_plan['projected_demand']
        
        resource_plan['storage_needed'] = resource_plan['projected_demand'] * unit_volume * 1.2
        resource_plan['capital_needed'] = resource_plan['projected_revenue'] * 0.6
        
        return resource_plan.sort_values('projected_demand', ascending=False)
    except Exception as e:
        print(f"Error in resource planning: {e}")
        return None

resource_df = resource_planning(df, PLANNING_HORIZON_DAYS, unit_volume=1.0)

total_storage = 0
total_capital = 0

if resource_df is not None and not resource_df.empty:
    print(f"\nResource Planning for Next {PLANNING_HORIZON_DAYS} Days (Top 10):")
    print(resource_df.head(10)[['medicine', 'daily_demand', 'projected_demand', 'storage_needed', 'capital_needed']].to_string(index=False))

    total_storage = resource_df['storage_needed'].sum()
    total_capital = resource_df['capital_needed'].sum()

    print(f"\nTotal Storage Required: {total_storage:,.0f} cubic feet")
    print(f"Total Capital Required: ₱{total_capital:,.2f}")

    resource_df.to_csv(os.path.join(output_dir, 'model_6_resource_planning.csv'), index=False)
    print(f"✓ Saved: model_6_resource_planning.csv ({len(resource_df)} records)")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    top_10_resource = resource_df.head(10)

    ax1.barh(top_10_resource['medicine'], top_10_resource['storage_needed'], color='steelblue')
    ax1.set_xlabel('Storage Needed (cubic feet)')
    ax1.set_ylabel('Medicine')
    ax1.set_title('Storage Requirements (Top 10 Products)', fontsize=14)

    ax2.barh(top_10_resource['medicine'], top_10_resource['capital_needed'], color='coral')
    ax2.set_xlabel('Capital Needed (₱)')
    ax2.set_ylabel('Medicine')
    ax2.set_title('Capital Requirements (Top 10 Products)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_6_resource_planning_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========== MODEL 7: ANOMALY DETECTION ==========
print("\n" + "=" * 80)
print("MODEL 7: ANOMALY DETECTION MODULE")
print("=" * 80)

def detect_anomalies(df):
    """
    Detect anomalies in sales patterns using Isolation Forest
    """
    try:
        daily_sales = df.groupby('date').agg({
            'sales': 'sum',
            'qty': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['day_of_month'] = daily_sales['date'].dt.day
        
        contamination = max(0.05, min(0.15, 100 / len(daily_sales)))
        
        features = daily_sales[['sales', 'qty', 'profit', 'day_of_week']].values
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        daily_sales['anomaly'] = iso_forest.fit_predict(features)
        daily_sales['anomaly_score'] = iso_forest.score_samples(features)
        
        anomalies = daily_sales[daily_sales['anomaly'] == -1].sort_values('anomaly_score')
        
        return daily_sales, anomalies
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return None, None

daily_sales, anomalies = detect_anomalies(df)

if daily_sales is not None and anomalies is not None:
    print(f"\nDetected {len(anomalies)} anomalous days:")
    if not anomalies.empty:
        print(anomalies[['date', 'sales', 'qty', 'profit', 'anomaly_score']].head(10).to_string(index=False))
    
    daily_sales.to_csv(os.path.join(output_dir, 'model_7_anomaly_detection.csv'), index=False)
    print(f"✓ Saved: model_7_anomaly_detection.csv ({len(daily_sales)} records)")
    
    if not anomalies.empty:
        anomalies.to_csv(os.path.join(output_dir, 'model_7_anomalies_only.csv'), index=False)
        print(f"✓ Saved: model_7_anomalies_only.csv ({len(anomalies)} records)")
    
    # Visualization
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(daily_sales[daily_sales['anomaly'] == 1]['date'],
               daily_sales[daily_sales['anomaly'] == 1]['sales'],
               c='blue', label='Normal', alpha=0.6, s=50)
    plt.scatter(daily_sales[daily_sales['anomaly'] == -1]['date'],
               daily_sales[daily_sales['anomaly'] == -1]['sales'],
               c='red', label='Anomaly', alpha=0.8, s=100, marker='x')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales (₱)')
    plt.title('Sales Anomaly Detection', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(daily_sales[daily_sales['anomaly'] == 1]['date'],
               daily_sales[daily_sales['anomaly'] == 1]['qty'],
               c='blue', label='Normal', alpha=0.6, s=50)
    plt.scatter(daily_sales[daily_sales['anomaly'] == -1]['date'],
               daily_sales[daily_sales['anomaly'] == -1]['qty'],
               c='red', label='Anomaly', alpha=0.8, s=100, marker='x')
    plt.xlabel('Date')
    plt.ylabel('Daily Quantity Sold')
    plt.title('Quantity Anomaly Detection', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_7_anomaly_detection_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ========== MODEL 8: PRESCRIPTIVE RECOMMENDATIONS ==========
print("\n" + "=" * 80)
print("MODEL 8: PRESCRIPTIVE RECOMMENDATIONS")
print("=" * 80)

def generate_recommendations(df, rop_df, eoq_df, allocation_df, product_discount, anomalies, resource_df):
    """
    Generate actionable recommendations based on all analyses
    """
    recommendations = []
    
    recommendations.append("\n📦 INVENTORY MANAGEMENT RECOMMENDATIONS:")
    
    if not rop_df.empty:
        high_demand = rop_df.nlargest(5, 'avg_daily_demand')
        recommendations.append(f"\n  ✓ HIGH PRIORITY REORDERS (Top 5 by demand):")
        for _, row in high_demand.iterrows():
            recommendations.append(f"    • {row['medicine']}: Reorder at {row['reorder_point']:.0f} units")
            recommendations.append(f"      - Safety stock: {row['safety_stock']:.0f} units | 30-day forecast: {row['forecast_30day']:.0f} units")
    
    recommendations.append("\n📋 ORDERING STRATEGY RECOMMENDATIONS:")
    
    if not eoq_df.empty:
        top_eoq = eoq_df.head(5)
        recommendations.append(f"\n  ✓ OPTIMAL ORDER QUANTITIES (Top 5):")
        for _, row in top_eoq.iterrows():
            recommendations.append(f"    • {row['medicine']}: Order {row['eoq']:.0f} units every {row['days_between_orders']:.0f} days")
    
    recommendations.append("\n💰 BUDGET ALLOCATION RECOMMENDATIONS:")
    
    if allocation_df is not None and not allocation_df.empty:
        top_alloc = allocation_df.head(5)
        recommendations.append(f"\n  ✓ PRIORITIZE INVESTMENT IN:")
        for _, row in top_alloc.iterrows():
            recommendations.append(f"    • {row['medicine']}: {row['optimal_allocation']:.0f} units (₱{row['allocated_value']:,.2f})")
    
    recommendations.append("\n💵 PRICING & DISCOUNT RECOMMENDATIONS:")
    
    if product_discount is not None and not product_discount.empty:
        efficient_discounts = product_discount.head(3)
        recommendations.append(f"\n  ✓ MAINTAIN/INCREASE DISCOUNTS FOR:")
        for _, row in efficient_discounts.iterrows():
            recommendations.append(f"    • {row['medicine']}: {row['discount_pct']:.1f}% discount (efficiency: {row['discount_efficiency']:.1f})")
    
    recommendations.append("\n📈 SALES GROWTH OPPORTUNITIES:")
    
    high_margin = medicine_stats.nlargest(5, 'profit_margin')
    recommendations.append(f"\n  ✓ PROMOTE HIGH-MARGIN PRODUCTS:")
    for _, row in high_margin.iterrows():
        recommendations.append(f"    • {row['medicine']}: {row['profit_margin']:.1f}% margin")
    
    recommendations.append("\n⚙️ OPERATIONAL EFFICIENCY:")
    
    if anomalies is not None and not anomalies.empty:
        recommendations.append(f"\n  ✓ INVESTIGATE {len(anomalies)} ANOMALOUS SALES DAYS")
        for _, row in anomalies.head(3).iterrows():
            recommendations.append(f"    • {row['date'].strftime('%Y-%m-%d')}: ₱{row['sales']:,.2f} (Score: {row['anomaly_score']:.2f})")
    
    recommendations.append("\n🏗️ RESOURCE PLANNING (Next 90 Days):")
    
    if resource_df is not None and not resource_df.empty:
        recommendations.append(f"\n  ✓ PREPARE FOR NEXT QUARTER:")
        recommendations.append(f"    • Storage needed: {resource_df['storage_needed'].sum():,.0f} cubic feet")
        recommendations.append(f"    • Capital required: ₱{resource_df['capital_needed'].sum():,.2f}")
    
    return "\n".join(recommendations)

recommendations = generate_recommendations(
    df, rop_df, eoq_df, allocation_df,
    product_discount, anomalies, resource_df
)

print(recommendations)

recommendations_file = os.path.join(output_dir, 'model_8_prescriptive_recommendations.txt')
with open(recommendations_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PHARMACY SALES PRESCRIPTIVE RECOMMENDATIONS (IMPROVED)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")
    f.write(recommendations)

print(f"\n✓ Saved: model_8_prescriptive_recommendations.txt")

summary_data = {
    'Model': [
        'Model 1: Reorder Point',
        'Model 2: EOQ',
        'Model 3: Inventory Allocation',
        'Model 4: What-If Analysis',
        'Model 5: Discount Optimization',
        'Model 6: Resource Planning',
        'Model 7: Anomaly Detection',
        'Model 8: Recommendations'
    ],
    'Records': [
        len(rop_df) if not rop_df.empty else 0,
        len(eoq_df) if not eoq_df.empty else 0,
        len(allocation_df) if not allocation_df.empty else 0,
        len(whatif_combined) if 'whatif_combined' in locals() else 0,
        len(product_discount) if product_discount is not None and not product_discount.empty else 0,
        len(resource_df) if resource_df is not None and not resource_df.empty else 0,
        len(daily_sales) if daily_sales is not None else 0,
        1
    ],
    'CSV_File': [
        'model_1_reorder_point.csv',
        'model_2_eoq.csv',
        'model_3_inventory_allocation.csv',
        'model_4_whatif_analysis.csv',
        'model_5_discount_by_product.csv',
        'model_6_resource_planning.csv',
        'model_7_anomaly_detection.csv',
        'model_8_prescriptive_recommendations.txt'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_dir, 'SUMMARY_all_models.csv'), index=False)
print(f"✓ Saved: SUMMARY_all_models.csv")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("\n📊 Generated CSV Files for Graphing:")
print("  1. model_1_reorder_point.csv - Reorder points with safety stock")
print("  2. model_2_eoq.csv - Economic order quantities and costs")
print("  3. model_3_inventory_allocation.csv - Optimal inventory allocation")
print("  4. model_4_whatif_analysis.csv - Scenario analysis for all products")
print("  5. model_5_discount_by_group.csv - Discount analysis by customer group")
print("  6. model_5_discount_by_product.csv - Discount efficiency by product")
print("  7. model_6_resource_planning.csv - Resource requirements for 90 days")
print("  8. model_7_anomaly_detection.csv - Daily sales with anomaly flags")
print("  9. model_7_anomalies_only.csv - Flagged anomalous days only")
print("  10. SUMMARY_all_models.csv - Index of all generated files")
print("\n📈 Generated Chart Images:")
print("  • model_1_reorder_point_chart.png")
print("  • model_2_eoq_chart.png")
print("  • model_3_inventory_allocation_chart.png")
print("  • model_4_whatif_analysis_chart.png")
print("  • model_5_discount_optimization_chart.png")
print("  • model_6_resource_planning_chart.png")
print("  • model_7_anomaly_detection_chart.png")
print("\n📄 Generated Report:")
print("  • model_8_prescriptive_recommendations.txt")
print("\n" + "=" * 80)
