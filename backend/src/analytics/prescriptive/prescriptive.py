# -*- coding: utf-8 -*-
"""
Enhanced Pharmacy Sales Analysis with Prescriptive Models

Includes:
- File selection dialog
- Forecast-driven Reorder Point with Safety Stock
- Economic Order Quantity (EOQ)
- Linear Programming for Inventory Allocation
- What-If Analysis for Inventory Planning
- Discount Optimization Engine
- Resource Planning for Supplies
- Anomaly Detection Module
- Prescriptive Recommendations
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
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import linprog, minimize
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings('ignore')

# Define the output directory
output_dir = "prescriptive_output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# FILE SELECTION DIALOG
# -----------------------------
def select_csv_file():
    """Open file dialog to select CSV file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Pharmacy Sales CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected. Exiting...")
        sys.exit(0)
    
    return Path(file_path)

# -----------------------------
# CONFIG
# -----------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

print("=" * 80)
print("PHARMACY SALES PRESCRIPTIVE ANALYSIS")
print("=" * 80)

# Select CSV file
csv_path = select_csv_file()
print(f"\n✓ Selected file: {csv_path}")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(csv_path)
print(f"\n✓ Loaded {len(df)} rows × {len(df.columns)} columns")

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# -----------------------------
# CLEAN / CAST TYPES
# -----------------------------
def to_numeric_safe(series):
    """Convert to numeric even if there are commas or spaces."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

# Dates
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
else:
    raise KeyError("Required column 'date' not found in CSV.")

if 'expiration' in df.columns:
    df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce', infer_datetime_format=True)

# Numerics
for col in ['qty', 'sales', 'cost', 'profit', 'payment', 'discount', 'receipt', 'so']:
    if col in df.columns:
        df[col] = to_numeric_safe(df[col])

# Text
if 'description' in df.columns:
    df['description'] = df['description'].astype(str).str.strip()
else:
    raise KeyError("Required column 'description' not found in CSV.")

df['medicine'] = df['description']

# Customer group
df['customer_group'] = np.where((df.get('discount', 0).fillna(0) > 0), 'Senior/PWD', 'Regular')

# Add week and month
df['week'] = df['date'].dt.to_period('W')
df['month'] = df['date'].dt.to_period('M')

print("✓ Data cleaned and processed")

# -----------------------------
# BASIC AGGREGATIONS
# -----------------------------
medicine_stats = df.groupby('medicine').agg({
    'qty': 'sum',
    'sales': 'sum',
    'cost': 'sum',
    'profit': 'sum'
}).reset_index()

medicine_stats.columns = ['medicine', 'total_qty', 'total_sales', 'total_cost', 'total_profit']
medicine_stats['avg_unit_price'] = medicine_stats['total_sales'] / medicine_stats['total_qty']
medicine_stats['profit_margin'] = (medicine_stats['total_profit'] / medicine_stats['total_sales'] * 100).fillna(0)

# Top 20 products for detailed analysis
top_products = medicine_stats.nlargest(20, 'total_qty')['medicine'].tolist()

print(f"✓ Analyzing top {len(top_products)} products")

# -----------------------------
# MODEL 1: FORECAST-DRIVEN REORDER POINT WITH SAFETY STOCK
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 1: FORECAST-DRIVEN REORDER POINT WITH SAFETY STOCK")
print("=" * 80)

def calculate_reorder_point(medicine_name, df, lead_time_days=7, service_level=0.95):
    """
    Calculate reorder point with safety stock based on demand forecast
    
    ROP = (Average Daily Demand × Lead Time) + Safety Stock
    Safety Stock = Z-score × Std Dev of Daily Demand × √Lead Time
    """
    med_data = df[df['medicine'] == medicine_name].copy()
    
    if len(med_data) < 30:  # Need sufficient data
        return None
    
    # Daily demand
    daily_demand = med_data.groupby('date')['qty'].sum().reset_index()
    daily_demand = daily_demand.set_index('date').resample('D').sum().fillna(0)
    
    avg_daily_demand = daily_demand['qty'].mean()
    std_daily_demand = daily_demand['qty'].std()
    
    # Z-score for service level
    z_score = stats.norm.ppf(service_level)
    
    # Safety stock
    safety_stock = z_score * std_daily_demand * math.sqrt(lead_time_days)
    
    # Reorder point
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
    
    # Forecast next 30 days using exponential smoothing
    try:
        if len(daily_demand) >= 14:
            model = ExponentialSmoothing(daily_demand['qty'], seasonal_periods=7, trend='add', seasonal='add')
            fitted = model.fit()
            forecast = fitted.forecast(steps=30)
            forecast_demand = forecast.sum()
        else:
            forecast_demand = avg_daily_demand * 30
    except:
        forecast_demand = avg_daily_demand * 30
    
    return {
        'medicine': medicine_name,
        'avg_daily_demand': avg_daily_demand,
        'std_daily_demand': std_daily_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'forecast_30day': forecast_demand,
        'service_level': service_level * 100
    }

reorder_points = []
for med in top_products:
    result = calculate_reorder_point(med, df)
    if result:
        reorder_points.append(result)

rop_df = pd.DataFrame(reorder_points)

print("\nReorder Point Analysis (Top 10):")
print(rop_df.head(10)[['medicine', 'avg_daily_demand', 'safety_stock', 'reorder_point', 'forecast_30day']].to_string(index=False))

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
plt.savefig(os.path.join(output_dir, 'reorder_point_analysis.png'), dpi=300, bbox_inches='tight')   
plt.show()

# -----------------------------
# MODEL 2: ENHANCED ECONOMIC ORDER QUANTITY (EOQ)
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 2: ECONOMIC ORDER QUANTITY (EOQ)")
print("=" * 80)

def calculate_eoq(annual_demand, ordering_cost, holding_cost_percent, unit_cost):
    """
    Calculate Economic Order Quantity with enhanced error handling

    EOQ = sqrt(2 * Annual Demand * Ordering Cost / Holding Cost)

    Includes validation and error handling for edge cases
    """
    try:
        # Input validation
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_percent <= 0 or unit_cost <= 0:
            return np.nan

        # Calculate holding cost per unit per year
        holding_cost = holding_cost_percent * float(unit_cost)

        if holding_cost <= 0:
            return np.nan

        # Calculate EOQ
        eoq = math.sqrt((2 * float(annual_demand) * float(ordering_cost)) / holding_cost)

        # Sanity check - EOQ shouldn't be unreasonably large
        if eoq > annual_demand * 2:  # EOQ > 2x annual demand is suspicious
            print(f"Warning: EOQ ({eoq:.0f}) seems unusually high for annual demand ({annual_demand:.0f})")
            return np.nan

        return eoq

    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        print(f"Error calculating EOQ: {e}")
        return np.nan

# Assumptions
ordering_cost = 150.0  # Cost per order
holding_cost_percent = 0.25  # 25% of unit cost annually

eoq_results = []
for _, row in medicine_stats[medicine_stats['medicine'].isin(top_products)].iterrows():
    annual_demand = row['total_qty']
    unit_cost = row['avg_unit_price']
    
    if pd.notna(unit_cost) and unit_cost > 0:
        eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_percent, unit_cost)
        
        if pd.notna(eoq) and eoq > 0:
            orders_per_year = annual_demand / eoq
            days_between_orders = 365 / orders_per_year
            
            # Total annual cost
            annual_ordering_cost = orders_per_year * ordering_cost
            annual_holding_cost = (eoq / 2) * (holding_cost_percent * unit_cost)
            annual_purchase_cost = annual_demand * unit_cost
            total_annual_cost = annual_ordering_cost + annual_holding_cost + annual_purchase_cost
            
            eoq_results.append({
                'medicine': row['medicine'],
                'annual_demand': annual_demand,
                'unit_cost': unit_cost,
                'eoq': eoq,
                'orders_per_year': orders_per_year,
                'days_between_orders': days_between_orders,
                'total_annual_cost': total_annual_cost,
                'ordering_cost': annual_ordering_cost,
                'holding_cost': annual_holding_cost
            })

eoq_df = pd.DataFrame(eoq_results).sort_values('annual_demand', ascending=False)

print("\nEOQ Analysis (Top 10):")
print(eoq_df.head(10)[['medicine', 'annual_demand', 'eoq', 'orders_per_year', 'days_between_orders']].to_string(index=False))

# Visualization
plt.figure(figsize=(14, 8))
top_10_eoq = eoq_df.head(10)
sns.barplot(data=top_10_eoq, x='eoq', y='medicine', palette='viridis')
plt.title('Economic Order Quantity (EOQ) for Top 10 Products', fontsize=16, pad=20)
plt.xlabel('EOQ (Units)')
plt.ylabel('Medicine')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eoq_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# MODEL 3: LINEAR PROGRAMMING FOR INVENTORY ALLOCATION
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 3: LINEAR PROGRAMMING FOR INVENTORY ALLOCATION")
print("=" * 80)

def optimize_inventory_allocation(products_df, budget_constraint, storage_constraint):
    """
    Optimize inventory allocation using linear programming
    Maximize: Total Profit
    Subject to: Budget and Storage constraints
    """
    n = len(products_df)
    
    # Objective: Maximize profit (negative for minimization)
    c = -products_df['profit_margin'].values
    
    # Inequality constraints: Ax <= b
    # Budget constraint: sum(unit_cost * qty) <= budget
    # Storage constraint: sum(qty) <= storage_capacity
    A_ub = np.array([
        products_df['avg_unit_price'].values,  # Budget
        np.ones(n)  # Storage
    ])
    b_ub = np.array([budget_constraint, storage_constraint])
    
    # Bounds: each product quantity >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        allocation = result.x
        products_df['optimal_allocation'] = allocation
        products_df['allocated_value'] = allocation * products_df['avg_unit_price']
        products_df['expected_profit'] = allocation * products_df['profit_margin'] * products_df['avg_unit_price'] / 100
        
        return products_df[products_df['optimal_allocation'] > 0].sort_values('optimal_allocation', ascending=False)
    else:
        return None

# Set constraints (adjust based on your business)
total_budget = medicine_stats['total_sales'].sum() * 0.3  # 30% of total sales as budget
storage_capacity = medicine_stats['total_qty'].sum() * 0.5  # 50% of annual volume

allocation_df = optimize_inventory_allocation(
    medicine_stats[medicine_stats['medicine'].isin(top_products)].copy(),
    total_budget,
    storage_capacity
)

if allocation_df is not None:
    print(f"\nOptimal Inventory Allocation (Budget: ${total_budget:,.2f}, Storage: {storage_capacity:,.0f} units):")
    print(allocation_df.head(10)[['medicine', 'optimal_allocation', 'allocated_value', 'expected_profit']].to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(14, 8))
    top_10_alloc = allocation_df.head(10)
    sns.barplot(data=top_10_alloc, x='optimal_allocation', y='medicine', palette='rocket')
    plt.title('Optimal Inventory Allocation (Linear Programming)', fontsize=16, pad=20)
    plt.xlabel('Allocated Quantity (Units)')
    plt.ylabel('Medicine')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inventory_allocation.png'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\nOptimization failed. Adjust constraints.")

# -----------------------------
# MODEL 4: WHAT-IF ANALYSIS FOR INVENTORY PLANNING
# -----------------------------
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
    avg_price = current_sales / current_demand if current_demand > 0 else 0
    
    results = []
    for scenario_name, demand_change, price_change in scenarios:
        new_demand = current_demand * (1 + demand_change)
        new_price = avg_price * (1 + price_change)
        new_sales = new_demand * new_price
        
        # Estimate profit (assuming cost remains constant)
        cost_per_unit = (current_sales - current_profit) / current_demand if current_demand > 0 else 0
        new_profit = new_sales - (new_demand * cost_per_unit)
        
        results.append({
            'scenario': scenario_name,
            'demand_change': f"{demand_change*100:+.0f}%",
            'price_change': f"{price_change*100:+.0f}%",
            'projected_demand': new_demand,
            'projected_sales': new_sales,
            'projected_profit': new_profit,
            'profit_change': ((new_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        })
    
    return pd.DataFrame(results)

# Define scenarios
scenarios = [
    ("Pessimistic", -0.20, -0.10),  # 20% demand drop, 10% price drop
    ("Conservative", -0.10, 0.00),   # 10% demand drop, no price change
    ("Current", 0.00, 0.00),         # No change
    ("Optimistic", 0.15, 0.05),      # 15% demand increase, 5% price increase
    ("Aggressive", 0.30, 0.10),      # 30% demand increase, 10% price increase
]

# Analyze top product
top_medicine = top_products[0]
whatif_df = what_if_analysis(top_medicine, df, scenarios)

if whatif_df is not None:
    print(f"\nWhat-If Analysis for: {top_medicine}")
    print(whatif_df[['scenario', 'demand_change', 'price_change', 'projected_sales', 'projected_profit', 'profit_change']].to_string(index=False))
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sales projection
    ax1.bar(whatif_df['scenario'], whatif_df['projected_sales'], color=['red', 'orange', 'gray', 'lightgreen', 'green'])
    ax1.set_title(f'Sales Projections - {top_medicine}', fontsize=14)
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Projected Sales ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Profit projection
    ax2.bar(whatif_df['scenario'], whatif_df['projected_profit'], color=['red', 'orange', 'gray', 'lightgreen', 'green'])
    ax2.set_title(f'Profit Projections - {top_medicine}', fontsize=14)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Projected Profit ($)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ewhatif_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# MODEL 5: DISCOUNT OPTIMIZATION ENGINE
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 5: DISCOUNT OPTIMIZATION ENGINE")
print("=" * 80)

def optimize_discount_strategy(df):
    """
    Analyze discount effectiveness and optimize discount strategy
    """
    # Analyze discount impact
    discount_analysis = df.groupby('customer_group').agg({
        'qty': 'sum',
        'sales': 'sum',
        'profit': 'sum',
        'discount': 'mean'
    }).reset_index()
    
    discount_analysis['avg_discount_pct'] = discount_analysis['discount'] / discount_analysis['sales'] * 100
    discount_analysis['profit_margin'] = discount_analysis['profit'] / discount_analysis['sales'] * 100
    
    # Product-level discount analysis
    product_discount = df[df['discount'] > 0].groupby('medicine').agg({
        'qty': 'sum',
        'sales': 'sum',
        'profit': 'sum',
        'discount': 'sum'
    }).reset_index()
    
    product_discount['discount_pct'] = product_discount['discount'] / (product_discount['sales'] + product_discount['discount']) * 100
    product_discount['profit_margin'] = product_discount['profit'] / product_discount['sales'] * 100
    
    # Find optimal discount range
    product_discount['discount_efficiency'] = product_discount['qty'] / product_discount['discount']
    
    return discount_analysis, product_discount.sort_values('discount_efficiency', ascending=False)

group_discount, product_discount = optimize_discount_strategy(df)

print("\nDiscount Analysis by Customer Group:")
print(group_discount[['customer_group', 'qty', 'sales', 'profit', 'avg_discount_pct', 'profit_margin']].to_string(index=False))

print("\nTop 10 Products by Discount Efficiency:")
print(product_discount.head(10)[['medicine', 'qty', 'discount_pct', 'profit_margin', 'discount_efficiency']].to_string(index=False))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Discount vs Profit Margin
ax1.scatter(product_discount['discount_pct'], product_discount['profit_margin'], alpha=0.6, s=100)
ax1.set_xlabel('Discount %')
ax1.set_ylabel('Profit Margin %')
ax1.set_title('Discount vs Profit Margin', fontsize=14)
ax1.grid(True, alpha=0.3)

# Top discount efficient products
top_10_disc = product_discount.head(10)
ax2.barh(top_10_disc['medicine'], top_10_disc['discount_efficiency'], color='teal')
ax2.set_xlabel('Discount Efficiency (Qty per $ Discount)')
ax2.set_ylabel('Medicine')
ax2.set_title('Top 10 Discount Efficient Products', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'discount_optimization.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# MODEL 6: RESOURCE PLANNING FOR SUPPLIES
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 6: RESOURCE PLANNING FOR SUPPLIES")
print("=" * 80)

def resource_planning(df, planning_horizon_days=90):
    """
    Plan resources needed for next planning period
    """
    # Calculate daily demand rate
    date_range = (df['date'].max() - df['date'].min()).days
    
    resource_plan = df.groupby('medicine').agg({
        'qty': 'sum',
        'sales': 'sum'
    }).reset_index()
    
    resource_plan['daily_demand'] = resource_plan['qty'] / date_range
    resource_plan['projected_demand'] = resource_plan['daily_demand'] * planning_horizon_days
    resource_plan['projected_revenue'] = (resource_plan['sales'] / resource_plan['qty']) * resource_plan['projected_demand']
    
    # Calculate required storage space (assuming 1 unit = 1 cubic foot)
    resource_plan['storage_needed'] = resource_plan['projected_demand'] * 1.2  # 20% buffer
    
    # Calculate required capital
    resource_plan['capital_needed'] = resource_plan['projected_revenue'] * 0.6  # Assuming 60% COGS
    
    return resource_plan.sort_values('projected_demand', ascending=False)

resource_df = resource_planning(df, planning_horizon_days=90)

print(f"\nResource Planning for Next 90 Days (Top 10):")
print(resource_df.head(10)[['medicine', 'daily_demand', 'projected_demand', 'storage_needed', 'capital_needed']].to_string(index=False))

total_storage = resource_df['storage_needed'].sum()
total_capital = resource_df['capital_needed'].sum()

print(f"\nTotal Storage Required: {total_storage:,.0f} cubic feet")
print(f"Total Capital Required: ${total_capital:,.2f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top_10_resource = resource_df.head(10)

# Storage requirements
ax1.barh(top_10_resource['medicine'], top_10_resource['storage_needed'], color='steelblue')
ax1.set_xlabel('Storage Needed (cubic feet)')
ax1.set_ylabel('Medicine')
ax1.set_title('Storage Requirements (Top 10 Products)', fontsize=14)

# Capital requirements
ax2.barh(top_10_resource['medicine'], top_10_resource['capital_needed'], color='coral')
ax2.set_xlabel('Capital Needed ($)')
ax2.set_ylabel('Medicine')
ax2.set_title('Capital Requirements (Top 10 Products)', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'resource_planning.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# MODEL 7: ANOMALY DETECTION MODULE
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 7: ANOMALY DETECTION MODULE")
print("=" * 80)

def detect_anomalies(df):
    """
    Detect anomalies in sales patterns using Isolation Forest
    """
    # Daily sales aggregation
    daily_sales = df.groupby('date').agg({
        'sales': 'sum',
        'qty': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    # Add day of week and other features
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['day_of_month'] = daily_sales['date'].dt.day
    
    # Prepare features for anomaly detection
    features = daily_sales[['sales', 'qty', 'profit', 'day_of_week']].values
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    daily_sales['anomaly'] = iso_forest.fit_predict(features)
    daily_sales['anomaly_score'] = iso_forest.score_samples(features)
    
    # -1 indicates anomaly, 1 indicates normal
    anomalies = daily_sales[daily_sales['anomaly'] == -1].sort_values('anomaly_score')
    
    return daily_sales, anomalies

daily_sales, anomalies = detect_anomalies(df)

print(f"\nDetected {len(anomalies)} anomalous days:")
print(anomalies[['date', 'sales', 'qty', 'profit', 'anomaly_score']].head(10).to_string(index=False))

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
plt.ylabel('Daily Sales ($)')
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
plt.savefig(os.path.join(output_dir, 'anomaly_detection.png'), dpi=300, bbox_inches='tight')

plt.show()

# -----------------------------
# MODEL 8: PRESCRIPTIVE RECOMMENDATIONS ENGINE
# -----------------------------
print("\n" + "=" * 80)
print("MODEL 8: PRESCRIPTIVE RECOMMENDATIONS TO IMPROVE SALES")
print("=" * 80)

def generate_recommendations(df, rop_df, eoq_df, allocation_df, product_discount, anomalies, resource_df):
    """
    Generate actionable recommendations based on all analyses
    """
    recommendations = []
    
    # 1. Inventory Management Recommendations
    recommendations.append("\n📦 INVENTORY MANAGEMENT RECOMMENDATIONS:")
    
    # Products needing immediate reorder
    if not rop_df.empty:
        high_demand = rop_df.nlargest(5, 'avg_daily_demand')
        recommendations.append(f"\n  ✓ HIGH PRIORITY REORDERS (Top 5 by demand):")
        for _, row in high_demand.iterrows():
            recommendations.append(f"    • {row['medicine']}: Reorder when stock falls below {row['reorder_point']:.0f} units")
            recommendations.append(f"      - Maintain safety stock of {row['safety_stock']:.0f} units")
            recommendations.append(f"      - Expected demand (30 days): {row['forecast_30day']:.0f} units")
    
    # 2. Ordering Strategy Recommendations
    recommendations.append("\n📋 ORDERING STRATEGY RECOMMENDATIONS:")
    
    if not eoq_df.empty:
        top_eoq = eoq_df.head(5)
        recommendations.append(f"\n  ✓ OPTIMAL ORDER QUANTITIES (Top 5 products):")
        for _, row in top_eoq.iterrows():
            recommendations.append(f"    • {row['medicine']}: Order {row['eoq']:.0f} units every {row['days_between_orders']:.0f} days")
            recommendations.append(f"      - This minimizes total cost to ${row['total_annual_cost']:,.2f}/year")
    
    # 3. Budget Allocation Recommendations
    recommendations.append("\n💰 BUDGET ALLOCATION RECOMMENDATIONS:")
    
    if allocation_df is not None and not allocation_df.empty:
        top_alloc = allocation_df.head(5)
        recommendations.append(f"\n  ✓ PRIORITIZE INVESTMENT IN:")
        for _, row in top_alloc.iterrows():
            recommendations.append(f"    • {row['medicine']}: Allocate {row['optimal_allocation']:.0f} units (${row['allocated_value']:,.2f})")
            recommendations.append(f"      - Expected profit: ${row['expected_profit']:,.2f}")
    
    # 4. Pricing & Discount Recommendations
    recommendations.append("\n💵 PRICING & DISCOUNT RECOMMENDATIONS:")
    
    if not product_discount.empty:
        # High discount efficiency products
        efficient_discounts = product_discount.head(3)
        recommendations.append(f"\n  ✓ MAINTAIN/INCREASE DISCOUNTS FOR:")
        for _, row in efficient_discounts.iterrows():
            recommendations.append(f"    • {row['medicine']}: Current discount {row['discount_pct']:.1f}% is highly efficient")
            recommendations.append(f"      - Generates {row['discount_efficiency']:.1f} units per dollar of discount")
        
        # Low efficiency products
        inefficient_discounts = product_discount[product_discount['discount_efficiency'] < product_discount['discount_efficiency'].median()].head(3)
        if not inefficient_discounts.empty:
            recommendations.append(f"\n  ✓ REDUCE/ELIMINATE DISCOUNTS FOR:")
            for _, row in inefficient_discounts.iterrows():
                recommendations.append(f"    • {row['medicine']}: Low efficiency ({row['discount_efficiency']:.1f} units/$)")
                recommendations.append(f"      - Consider reducing discount from {row['discount_pct']:.1f}%")
    
    # 5. Sales Growth Opportunities
    recommendations.append("\n📈 SALES GROWTH OPPORTUNITIES:")
    
    # High margin products
    high_margin = medicine_stats.nlargest(5, 'profit_margin')
    recommendations.append(f"\n  ✓ PROMOTE HIGH-MARGIN PRODUCTS:")
    for _, row in high_margin.iterrows():
        recommendations.append(f"    • {row['medicine']}: {row['profit_margin']:.1f}% margin")
        recommendations.append(f"      - Increase visibility and marketing for this product")
    
    # Underperforming products with potential
    if not rop_df.empty:
        low_volume_high_demand = rop_df[rop_df['avg_daily_demand'] > rop_df['avg_daily_demand'].median()]
        if not low_volume_high_demand.empty:
            recommendations.append(f"\n  ✓ PRODUCTS WITH GROWTH POTENTIAL:")
            for _, row in low_volume_high_demand.head(3).iterrows():
                recommendations.append(f"    • {row['medicine']}: Steady demand ({row['avg_daily_demand']:.1f} units/day)")
                recommendations.append(f"      - Consider promotional campaigns to boost sales")
    
    # 6. Operational Efficiency
    recommendations.append("\n⚙️ OPERATIONAL EFFICIENCY RECOMMENDATIONS:")
    
    if not anomalies.empty:
        recommendations.append(f"\n  ✓ INVESTIGATE ANOMALOUS SALES DAYS:")
        recommendations.append(f"    • {len(anomalies)} days showed unusual patterns")
        recommendations.append(f"    • Review these dates for:")
        recommendations.append(f"      - Stock-outs or supply issues")
        recommendations.append(f"      - Unexpected demand spikes")
        recommendations.append(f"      - Data entry errors")
        
        # Show top 3 anomalies
        top_anomalies = anomalies.head(3)
        for _, row in top_anomalies.iterrows():
            recommendations.append(f"    • {row['date'].strftime('%Y-%m-%d')}: Sales ${row['sales']:,.2f} (Score: {row['anomaly_score']:.2f})")
    
    # 7. Resource Planning
    recommendations.append("\n🏗️ RESOURCE PLANNING RECOMMENDATIONS:")
    
    if not resource_df.empty:
        recommendations.append(f"\n  ✓ PREPARE FOR NEXT 90 DAYS:")
        recommendations.append(f"    • Total storage space needed: {resource_df['storage_needed'].sum():,.0f} cubic feet")
        recommendations.append(f"    • Total capital required: ${resource_df['capital_needed'].sum():,.2f}")
        recommendations.append(f"    • Ensure adequate:")
        recommendations.append(f"      - Warehouse capacity")
        recommendations.append(f"      - Working capital/credit lines")
        recommendations.append(f"      - Staff for inventory management")
    
    # 8. Customer Segment Strategy
    recommendations.append("\n👥 CUSTOMER SEGMENT STRATEGY:")
    
    group_stats = df.groupby('customer_group').agg({
        'sales': 'sum',
        'profit': 'sum',
        'qty': 'sum'
    }).reset_index()
    
    recommendations.append(f"\n  ✓ CUSTOMER GROUP INSIGHTS:")
    for _, row in group_stats.iterrows():
        profit_margin = (row['profit'] / row['sales'] * 100) if row['sales'] > 0 else 0
        recommendations.append(f"    • {row['customer_group']}:")
        recommendations.append(f"      - Sales: ${row['sales']:,.2f}")
        recommendations.append(f"      - Profit Margin: {profit_margin:.1f}%")
        recommendations.append(f"      - Units Sold: {row['qty']:,.0f}")
    
    # 9. Quick Wins
    recommendations.append("\n🎯 QUICK WINS (Immediate Actions):")
    recommendations.append(f"\n  ✓ THIS WEEK:")
    recommendations.append(f"    1. Review and adjust reorder points for top 10 products")
    recommendations.append(f"    2. Implement EOQ-based ordering for high-volume items")
    recommendations.append(f"    3. Analyze anomalous sales days and document findings")
    
    recommendations.append(f"\n  ✓ THIS MONTH:")
    recommendations.append(f"    1. Optimize discount strategy based on efficiency analysis")
    recommendations.append(f"    2. Reallocate budget to high-margin products")
    recommendations.append(f"    3. Set up automated reorder alerts at calculated ROP levels")
    recommendations.append(f"    4. Train staff on new inventory management procedures")
    
    recommendations.append(f"\n  ✓ THIS QUARTER:")
    recommendations.append(f"    1. Implement full resource planning system")
    recommendations.append(f"    2. Develop customer segment-specific marketing campaigns")
    recommendations.append(f"    3. Review and optimize supplier relationships")
    recommendations.append(f"    4. Establish KPIs and monitoring dashboards")
    
    # 10. Risk Mitigation
    recommendations.append("\n⚠️ RISK MITIGATION:")
    recommendations.append(f"\n  ✓ MONITOR CLOSELY:")
    
    # Products with high variability
    if not rop_df.empty:
        high_variability = rop_df.nlargest(3, 'std_daily_demand')
        recommendations.append(f"    • High demand variability products:")
        for _, row in high_variability.iterrows():
            cv = (row['std_daily_demand'] / row['avg_daily_demand'] * 100) if row['avg_daily_demand'] > 0 else 0
            recommendations.append(f"      - {row['medicine']}: CV = {cv:.1f}%")
            recommendations.append(f"        → Maintain higher safety stock")
    
    return "\n".join(recommendations)

# Generate comprehensive recommendations
recommendations = generate_recommendations(
    df, rop_df, eoq_df, allocation_df, 
    product_discount, anomalies, resource_df
)

print(recommendations)

# Save recommendations to file in the prescriptive_output folder
recommendations_file_path = os.path.join(output_dir, 'prescriptive_recommendations.txt')

with open(recommendations_file_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PHARMACY SALES PRESCRIPTIVE RECOMMENDATIONS\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")
    f.write(recommendations)

print(f"\n✓ Recommendations saved to: {recommendations_file_path}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  • reorder_point_analysis.png")
print("  • eoq_analysis.png")
print("  • inventory_allocation.png")
print("  • whatif_analysis.png")
print("  • discount_optimization.png")
print("  • resource_planning.png")
print("  • anomaly_detection.png")
print("  • prescriptive_recommendations.txt")
print("\n" + "=" * 80)
