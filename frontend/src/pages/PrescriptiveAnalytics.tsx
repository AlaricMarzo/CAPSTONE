"use client"

import { useState, useEffect } from "react"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { Package, Zap, TrendingUp, Target, DollarSign, Loader2 } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("en-PH", {
    style: "currency",
    currency: "PHP",
    minimumFractionDigits: 0,
  }).format(value)
}

interface PrescriptiveData {
  reorder_points: Array<{
    medicine: string
    reorder_point: number
    safety_stock: number
    avg_daily_demand: number
  }>
  eoq_data: Array<{
    medicine: string
    eoq: number
    annual_demand: number
    orders_per_year: number
  }>
  allocations: Array<{
    medicine: string
    optimal_allocation: number
    allocated_value: number
    profit_margin: number
  }>
  discount_groups: Array<{
    customer_group: string
    qty: number
    sales: number
    profit: number
    avg_discount_pct: number
    profit_margin: number
  }>
  resource_planning: Array<{
    medicine: string
    daily_demand: number
    projected_demand: number
    storage_needed: number
    capital_needed: number
  }>
  anomalies: Array<{
    date: string
    sales: number
    qty: number
    anomaly_score: number
  }>
  financial_summary: {
    total_sales: number
    total_cost: number
    total_profit: number
    overall_profit_margin_pct: number
    total_quantity_sold: number
  }
  key_metrics: {
    total_products_optimized: number
    total_models: number
  }
  model_images?: {
    model_1_reorder_point?: string
    model_2_eoq?: string
    model_3_inventory_allocation?: string
    model_4_whatif_analysis?: string
    model_5_discount_optimization?: string
    model_6_resource_planning?: string
    model_7_anomaly_detection?: string
  }
}

export default function PrescriptiveAnalytics() {
  const [data, setData] = useState<PrescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch("/api/analytics/prescriptive")
        if (!response.ok) throw new Error("Failed to fetch prescriptive analytics")
        const result = await response.json()
        setData(result.data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred")
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
          Loading prescriptive analytics...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="text-destructive">Error: {error}</div>
      </div>
    )
  }

  if (!data || !data.reorder_points || data.reorder_points.length === 0) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="text-muted-foreground">Prescriptive analytics data not available. Please upload data and wait for processing to complete.</div>
      </div>
    )
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h1>
        <p className="text-muted-foreground">Optimization models and actionable recommendations</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Items to Reorder"
          value={data.reorder_points.length.toString()}
          change="recommendations"
          changeType="warning"
          icon={Package}
          color="warning"
          className="shadow-soft"
        />
        <MetricCard
          title="Cost Savings"
          value={formatCurrency(data.financial_summary.total_profit)}
          change="optimized"
          changeType="positive"
          icon={DollarSign}
          color="success"
          className="shadow-soft"
        />
        <MetricCard
          title="Optimized Products"
          value={data.key_metrics.total_products_optimized.toString()}
          change="improved"
          changeType="positive"
          icon={TrendingUp}
          color="info"
          className="shadow-soft"
        />
        <MetricCard
          title="Models Applied"
          value={data.key_metrics.total_models.toString()}
          change="active"
          changeType="positive"
          icon={Zap}
          color="success"
          className="shadow-soft"
        />
      </div>

      {/* Tabbed Interface for Different Models */}
      <Tabs defaultValue="reorder" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="reorder">Reorder Points</TabsTrigger>
          <TabsTrigger value="eoq">EOQ</TabsTrigger>
          <TabsTrigger value="allocation">Allocation</TabsTrigger>
          <TabsTrigger value="resource">Resource</TabsTrigger>
          <TabsTrigger value="discount">Discount</TabsTrigger>
        </TabsList>

        <TabsContent value="reorder" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Reorder Point Analysis</CardTitle>
              <CardDescription>Inventory thresholds and safety stock levels</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.reorder_points}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="medicine" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar dataKey="reorder_point" fill="#3b82f6" name="Reorder Point" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="safety_stock" fill="#f59e0b" name="Safety Stock" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {data.model_images?.model_1_reorder_point && (
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Generated Model Chart</CardTitle>
                <CardDescription>Chart generated by the prescriptive analytics model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <img
                    src={data.model_images.model_1_reorder_point}
                    alt="Generated Reorder Point Chart"
                    className="w-full h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="eoq" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Economic Order Quantity (EOQ)</CardTitle>
              <CardDescription>Optimal order sizes and ordering frequency</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.eoq_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="medicine" angle={-45} textAnchor="end" height={100} />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="eoq" fill="#10b981" name="EOQ Units" radius={[4, 4, 0, 0]} />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="orders_per_year"
                      stroke="#ef4444"
                      name="Orders/Year"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {data.model_images?.model_2_eoq && (
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Generated Model Chart</CardTitle>
                <CardDescription>Chart generated by the prescriptive analytics model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <img
                    src={data.model_images.model_2_eoq}
                    alt="Generated EOQ Chart"
                    className="w-full h-full object-contain"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="allocation" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Inventory Allocation</CardTitle>
              <CardDescription>Optimal inventory distribution by product</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.allocations}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="medicine" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Bar dataKey="optimal_allocation" fill="#8b5cf6" name="Optimal Allocation" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="resource" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Resource Planning</CardTitle>
              <CardDescription>Storage and capital requirements</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.resource_planning.slice(0, 20)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="medicine" angle={-45} textAnchor="end" height={100} />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="storage_needed" fill="#f59e0b" name="Storage (cubic ft)" radius={[4, 4, 0, 0]} />
                    <Bar yAxisId="right" dataKey="capital_needed" fill="#ef4444" name="Capital (PHP)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="discount" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Discount Analysis</CardTitle>
              <CardDescription>Discount effectiveness by customer group</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.discount_groups}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="customer_group" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales (PHP)" radius={[4, 4, 0, 0]} />
                    <Bar yAxisId="right" dataKey="profit_margin" fill="#10b981" name="Profit Margin %" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Anomaly Detection */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Anomaly Detection</CardTitle>
          <CardDescription>Unusual sales patterns and outliers</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {data.anomalies.slice(0, 10).map((anomaly, idx) => (
              <div key={idx} className="flex justify-between items-center p-3 bg-card border border-border rounded">
                <div>
                  <div className="font-semibold text-foreground">{anomaly.date}</div>
                  <div className="text-sm text-muted-foreground">
                    Sales: ₱{anomaly.sales.toFixed(2)} | Qty: {anomaly.qty}
                  </div>
                </div>
                <div className={`font-bold ${anomaly.anomaly_score < -0.5 ? "text-red-500" : "text-yellow-500"}`}>
                  Score: {anomaly.anomaly_score.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
