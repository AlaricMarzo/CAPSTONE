"use client"

import { useState, useEffect } from "react"
import { BarChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { AlertCircle, ShoppingCart, TrendingUp, Package, DollarSign, Zap, Loader2, CheckCircle } from "lucide-react"

interface ReorderPoint {
  medicine: string
  avg_daily_demand: number
  safety_stock: number
  reorder_point: number
  forecast_30day: number
}

interface EOQData {
  medicine: string
  annual_demand: number
  eoq: number
  orders_per_year: number
  days_between_orders: number
  total_annual_cost: number
}

interface AllocationData {
  medicine: string
  optimal_allocation: number
  allocated_value: number
  profit_margin: number
  expected_profit: number
}

interface DiscountData {
  customer_group: string
  qty: number
  sales: number
  profit: number
  avg_discount_pct: number
  profit_margin: number
}

interface ResourceData {
  medicine: string
  daily_demand: number
  projected_demand: number
  storage_needed: number
  capital_needed: number
}

interface AnomalyData {
  date: string
  sales: number
  qty: number
  profit: number
  anomaly_score: number
}

interface InventoryData {
  reorder_points: ReorderPoint[]
  eoq_data: EOQData[]
  allocations: AllocationData[]
  discount_groups: DiscountData[]
  resource_planning: ResourceData[]
  anomalies: AnomalyData[]
  financial_summary?: {
    total_sales: number
    total_cost: number
    total_profit: number
    overall_profit_margin_pct: number
    total_quantity_sold: number
  }
  key_metrics?: {
    total_products_optimized: number
    total_models: number
    total_cost_savings: number
  }
}

export default function InventoryRecommendations() {
  const [data, setData] = useState<InventoryData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
  const fetchData = async () => {
    try {
      setLoading(true)
      const response = await fetch("/api/analytics/prescriptive")
      if (!response.ok) throw new Error("Failed to fetch inventory recommendations")
      const result = await response.json()
      const processedData = {
        ...result.data,
        key_metrics: result.data?.key_metrics || { total_products_optimized: 0, total_models: 0, total_cost_savings: 0 },
        financial_summary: result.data?.financial_summary || { total_sales: 0, total_cost: 0, total_profit: 0, overall_profit_margin_pct: 0, total_quantity_sold: 0 }
      }
      setData(processedData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      console.error("[v0] Error fetching inventory data:", err)
    } finally {
      setLoading(false)
    }
  }

    fetchData()
    const handleUpdate = () => fetchData()
    window.addEventListener("analyticsUpdated", handleUpdate)
    return () => window.removeEventListener("analyticsUpdated", handleUpdate)
  }, [])

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading inventory recommendations...
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="text-destructive flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          Error: {error || "No data available"}
        </div>
      </div>
    )
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-PH", {
      style: "currency",
      currency: "PHP",
      minimumFractionDigits: 0,
    }).format(value)
  }

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

  return (
    <div className="flex-1 space-y-8 p-4 md:p-8 pt-6 bg-background overflow-auto">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl md:text-4xl font-bold text-foreground">Inventory Recommendations</h1>
        <p className="text-muted-foreground text-lg">
          Comprehensive optimization recommendations for inventory management
        </p>
      </div>

      {/* Top KPI Cards */}
      <div className="grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Products Optimized"
          value={(data.key_metrics?.total_products_optimized || 0).toString()}
          change="analyzed"
          changeType="positive"
          icon={Package}
          color="info"
        />
        <MetricCard
          title="Total Revenue"
          value={formatCurrency(data.financial_summary?.total_sales || 0)}
          change="this period"
          changeType="positive"
          icon={DollarSign}
          color="success"
        />
        <MetricCard
          title="Total Profit"
          value={formatCurrency(data.financial_summary?.total_profit || 0)}
          change={`${(data.financial_summary?.overall_profit_margin_pct || 0).toFixed(1)}% margin`}
          changeType="positive"
          icon={TrendingUp}
          color="warning"
        />
        <MetricCard
          title="Models Executed"
          value={(data.key_metrics?.total_models || 0).toString()}
          change="active"
          changeType="positive"
          icon={Zap}
          color="success"
        />
      </div>

      {/* Tabbed Recommendations Interface */}
      <Tabs defaultValue="reorder" className="space-y-6 w-full">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
          <TabsTrigger value="reorder">Reorder Points</TabsTrigger>
          <TabsTrigger value="eoq">EOQ Strategy</TabsTrigger>
          <TabsTrigger value="allocation">Budget Allocation</TabsTrigger>
          <TabsTrigger value="discount">Pricing</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
          <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
        </TabsList>

        {/* Reorder Point Recommendations */}
        <TabsContent value="reorder" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShoppingCart className="h-5 w-5 text-blue-500" />
                Reorder Point Analysis
              </CardTitle>
              <CardDescription>
                Recommended inventory thresholds and safety stock levels for all products
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Chart */}
              <div className="h-80 md:h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.reorder_points} margin={{ top: 20, right: 30, left: 0, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="medicine"
                      angle={-45}
                      textAnchor="end"
                      height={120}
                      interval={0}
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar dataKey="reorder_point" fill="#3b82f6" name="Reorder Point" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="safety_stock" fill="#10b981" name="Safety Stock" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Detailed Table */}
              <div className="space-y-3">
                <h3 className="font-semibold text-foreground">Detailed Recommendations</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left p-2 font-semibold">Product</th>
                        <th className="text-right p-2 font-semibold">Daily Demand</th>
                        <th className="text-right p-2 font-semibold">Reorder Point</th>
                        <th className="text-right p-2 font-semibold">Safety Stock</th>
                        <th className="text-right p-2 font-semibold">30-Day Forecast</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.reorder_points.map((item, idx) => (
                        <tr key={idx} className="border-b border-border hover:bg-muted/50 transition-colors">
                          <td className="p-2 font-medium">{item.medicine}</td>
                          <td className="text-right p-2">{item.avg_daily_demand.toFixed(2)}</td>
                          <td className="text-right p-2 text-blue-600 font-semibold">
                            {Math.ceil(item.reorder_point)}
                          </td>
                          <td className="text-right p-2 text-green-600 font-semibold">
                            {Math.ceil(item.safety_stock)}
                          </td>
                          <td className="text-right p-2">{Math.ceil(item.forecast_30day)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* EOQ Strategy Recommendations */}
        <TabsContent value="eoq" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-amber-500" />
                Economic Order Quantity (EOQ) Strategy
              </CardTitle>
              <CardDescription>Optimal order quantities and ordering frequency to minimize total costs</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Dual Axis Chart */}
              <div className="h-80 md:h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.eoq_data} margin={{ top: 20, right: 30, left: 0, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="medicine"
                      angle={-45}
                      textAnchor="end"
                      height={120}
                      interval={0}
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis yAxisId="left" label={{ value: "EOQ Units", angle: -90, position: "insideLeft" }} />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      label={{ value: "Orders/Year", angle: 90, position: "insideRight" }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="eoq" fill="#f59e0b" name="EOQ Units" radius={[4, 4, 0, 0]} />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="orders_per_year"
                      stroke="#ef4444"
                      name="Orders/Year"
                      strokeWidth={2}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Detailed Table */}
              <div className="space-y-3">
                <h3 className="font-semibold text-foreground">Ordering Strategy Details</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left p-2 font-semibold">Product</th>
                        <th className="text-right p-2 font-semibold">Annual Demand</th>
                        <th className="text-right p-2 font-semibold">EOQ</th>
                        <th className="text-right p-2 font-semibold">Orders/Year</th>
                        <th className="text-right p-2 font-semibold">Days Between</th>
                        <th className="text-right p-2 font-semibold">Annual Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.eoq_data.map((item, idx) => (
                        <tr key={idx} className="border-b border-border hover:bg-muted/50 transition-colors">
                          <td className="p-2 font-medium text-sm">{item.medicine}</td>
                          <td className="text-right p-2">{Math.round(item.annual_demand).toLocaleString()}</td>
                          <td className="text-right p-2 font-semibold text-amber-600">{Math.ceil(item.eoq)}</td>
                          <td className="text-right p-2">{item.orders_per_year.toFixed(2)}</td>
                          <td className="text-right p-2">{Math.ceil(item.days_between_orders)}</td>
                          <td className="text-right p-2">{formatCurrency(item.total_annual_cost)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Budget Allocation Recommendations */}
        <TabsContent value="allocation" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="h-5 w-5 text-green-500" />
                Budget Allocation Strategy
              </CardTitle>
              <CardDescription>Optimal budget distribution to maximize profit and meet constraints</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Key Metrics */}
              <div className="grid gap-4 md:gap-6 grid-cols-1 md:grid-cols-3">
                {data.allocations.map((alloc, idx) => (
                  <div key={idx} className="p-4 bg-muted/50 rounded-lg border border-border">
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-muted-foreground line-clamp-2">{alloc.medicine}</div>
                      <div className="space-y-1">
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-muted-foreground">Units</span>
                          <span className="font-semibold">{Math.round(alloc.optimal_allocation).toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-muted-foreground">Allocated</span>
                          <span className="font-semibold text-green-600">{formatCurrency(alloc.allocated_value)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-muted-foreground">Expected Profit</span>
                          <span className="font-semibold text-blue-600">{formatCurrency(alloc.expected_profit)}</span>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t border-border">
                          <span className="text-xs text-muted-foreground">Margin</span>
                          <span className="font-semibold text-amber-600">{alloc.profit_margin.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pricing & Discount Recommendations */}
        <TabsContent value="discount" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-purple-500" />
                Pricing & Discount Strategy
              </CardTitle>
              <CardDescription>Recommendations by customer group and profit margin optimization</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Customer Group Analysis */}
              <div className="space-y-4">
                <h3 className="font-semibold text-foreground">Customer Group Analysis</h3>
                <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
                  {data.discount_groups.map((group, idx) => (
                    <div key={idx} className="p-4 bg-muted/50 rounded-lg border border-border">
                      <h4 className="font-semibold mb-3">{group.customer_group}</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Quantity Sold</span>
                          <span className="font-medium">{Math.round(group.qty).toLocaleString()} units</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Sales Revenue</span>
                          <span className="font-medium">{formatCurrency(group.sales)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Profit</span>
                          <span className="font-medium text-green-600">{formatCurrency(group.profit)}</span>
                        </div>
                        <div className="flex justify-between pt-2 border-t border-border">
                          <span className="text-muted-foreground">Avg Discount</span>
                          <span className="font-medium">{(group.avg_discount_pct * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Profit Margin</span>
                          <span className="font-medium text-blue-600">{group.profit_margin.toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Resource Planning */}
        <TabsContent value="resources" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Package className="h-5 w-5 text-indigo-500" />
                Resource Planning (Next 90 Days)
              </CardTitle>
              <CardDescription>Storage and capital requirements for upcoming quarter</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Chart */}
              {(() => {
                const sortedData = [...data.resource_planning].sort((a, b) => b.daily_demand - a.daily_demand);
                const chartData = sortedData.slice(0, 20);
                const barSize = Math.min(30, 600 / (chartData.length || 1));
                return (
                  <div className="w-full">
                    <div className="h-80 md:h-96 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 80 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis
                            dataKey="medicine"
                            angle={-45}
                            textAnchor="end"
                            height={120}
                            interval={0}
                            tick={{ fontSize: 12, fill: "var(--foreground)" }}
                          />
                          <YAxis yAxisId="left" label={{ value: "Daily Demand", angle: -90, position: "insideLeft" }} />
                          <YAxis
                            yAxisId="right"
                            orientation="right"
                            label={{ value: "Storage (cu ft)", angle: 90, position: "insideRight" }}
                          />
                          <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", opacity: 0.95 }} />
                          <Legend />
                          <Bar
                            yAxisId="left"
                            dataKey="daily_demand"
                            fill="#6366f1"
                            name="Daily Demand"
                            radius={[4, 4, 0, 0]}
                            barSize={barSize}
                          />
                          <Bar
                            yAxisId="right"
                            dataKey="storage_needed"
                            fill="#ec4899"
                            name="Storage Needed"
                            radius={[4, 4, 0, 0]}
                            barSize={barSize}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })()}

              {/* Detailed Table */}
              <div className="space-y-3">
                <h3 className="font-semibold text-foreground">Resource Requirements by Product</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left p-2 font-semibold">Product</th>
                        <th className="text-right p-2 font-semibold">Daily Demand</th>
                        <th className="text-right p-2 font-semibold">90-Day Projection</th>
                        <th className="text-right p-2 font-semibold">Storage (cu ft)</th>
                        <th className="text-right p-2 font-semibold">Capital Required</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.resource_planning.slice(0, 10).map((item, idx) => (
                        <tr key={idx} className="border-b border-border hover:bg-muted/50 transition-colors">
                          <td className="p-2 font-medium text-sm">{item.medicine}</td>
                          <td className="text-right p-2">{item.daily_demand.toFixed(2)}</td>
                          <td className="text-right p-2">{Math.round(item.projected_demand).toLocaleString()}</td>
                          <td className="text-right p-2 font-semibold text-indigo-600">
                            {Math.round(item.storage_needed)}
                          </td>
                          <td className="text-right p-2 font-semibold">{formatCurrency(item.capital_needed)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Anomaly Detection */}
        <TabsContent value="anomalies" className="space-y-6">
          <Card className="border shadow-sm hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-red-500" />
                Critical Anomalies & Outliers
              </CardTitle>
              <CardDescription>Unusual sales patterns requiring investigation and action</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Anomalies List */}
              <div className="space-y-3">
                <h3 className="font-semibold text-foreground">Critical Events Detected</h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {data.anomalies.map((anomaly, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        anomaly.anomaly_score < -0.7
                          ? "border-red-500 bg-red-50 dark:bg-red-950"
                          : "border-yellow-500 bg-yellow-50 dark:bg-yellow-950"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-2 flex-1">
                          <div className="flex items-center gap-2">
                            {anomaly.anomaly_score < -0.7 ? (
                              <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
                            ) : (
                              <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0" />
                            )}
                            <span className="font-semibold text-foreground">{anomaly.date}</span>
                          </div>
                          <div className="grid gap-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Sales</span>
                              <span className="font-medium">{formatCurrency(anomaly.sales)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Quantity</span>
                              <span className="font-medium">{Math.round(anomaly.qty).toLocaleString()} units</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Profit</span>
                              <span className="font-medium text-green-600">{formatCurrency(anomaly.profit)}</span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div
                            className={`text-lg font-bold ${anomaly.anomaly_score < -0.7 ? "text-red-600" : "text-yellow-600"}`}
                          >
                            {anomaly.anomaly_score.toFixed(2)}
                          </div>
                          <span className="text-xs text-muted-foreground">Anomaly Score</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Summary Section */}
      <Card className="border shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            Financial Summary & Key Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-5">
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Total Revenue</div>
              <div className="text-2xl font-bold text-foreground">
                {formatCurrency(data.financial_summary?.total_sales || 0)}
              </div>
            </div>
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Total Cost</div>
              <div className="text-2xl font-bold text-foreground">
                {formatCurrency(data.financial_summary?.total_cost || 0)}
              </div>
            </div>
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Net Profit</div>
              <div className="text-2xl font-bold text-green-600">
                {formatCurrency(data.financial_summary?.total_profit || 0)}
              </div>
            </div>
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Profit Margin</div>
              <div className="text-2xl font-bold text-blue-600">
                {(data.financial_summary?.overall_profit_margin_pct || 0).toFixed(1)}%
              </div>
            </div>
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Total Quantity</div>
              <div className="text-2xl font-bold text-foreground">
                {((data.financial_summary?.total_quantity_sold || 0) / 1000).toFixed(1)}K units
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
