  "use client"

import { useState, useEffect } from "react"
import {
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { DollarSign, TrendingUp, Package, AlertCircle, Loader2 } from "lucide-react"

interface DescriptiveData {
  kpi_summary: {
    total_sales: number
    total_quantity: number
    active_skus: number
    growth_rate: number
  }
  monthly_sales: Array<{ month: string; sales: number; quantity: number }>
  category_distribution: Array<{ name: string; value: number }>
  top_products: Array<{ name: string; sales: number; quantity: number }>
  clustering_summary: Array<{ cluster: number; count: number; avg_value: number }>
  clustering_images: Array<{
    name: string
    image: string
    summary: Array<{ cluster: number; n: number; total_sales: number; total_qty: number; avg_price_med: number; months_active_med: number; mean_monthly_qty_med: number; cv_monthly_qty_med: number; trend_qty_slope_med: number; persona: string }>
  }>
  mba_rules: Array<{
    item_a: string
    item_b: string
    lift: number
    confidence_ab_pct: number
    support_pct: number
  }>
}

export default function DescriptiveAnalytics() {
  const [data, setData] = useState<DescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch("/api/analytics/descriptive")
        if (!response.ok) throw new Error("Failed to fetch descriptive analytics")
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
          Loading descriptive analytics...
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="text-destructive">Error: {error || "No data available"}</div>
      </div>
    )
  }

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

  const kpiCards = [
    {
      title: "Total Sales",
      value: `$${(data.kpi_summary.total_sales / 1000000).toFixed(2)}M`,
      change: `+${data.kpi_summary.growth_rate.toFixed(1)}%`,
      icon: DollarSign,
      color: "success",
    },
    {
      title: "Total Quantity Sold",
      value: (data.kpi_summary.total_quantity / 1000).toFixed(0) + "K",
      change: "units",
      icon: Package,
      color: "info",
    },
    {
      title: "Active SKUs",
      value: data.kpi_summary.active_skus.toString(),
      change: "products",
      icon: TrendingUp,
      color: "warning",
    },
    {
      title: "Growth Rate",
      value: `${data.kpi_summary.growth_rate.toFixed(1)}%`,
      change: "YoY",
      icon: AlertCircle,
      color: "success" as const,
    },
  ]

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Descriptive Analytics</h1>
        <p className="text-muted-foreground">Sales trends, product distribution, and customer insights</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {kpiCards.map((card, idx) => (
          <MetricCard
            key={idx}
            title={card.title}
            value={card.value}
            change={card.change}
            changeType="positive"
            icon={card.icon}
            color={card.color}
            className="shadow-soft"
          />
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Monthly Sales Trend */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Monthly Sales & Quantity Trend</CardTitle>
            <CardDescription>Sales and quantity sold by month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data.monthly_sales}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales ($)" radius={[4, 4, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="quantity" stroke="#10b981" name="Quantity" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Category Distribution */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Sales by Category</CardTitle>
            <CardDescription>Product distribution across categories</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.category_distribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {data.category_distribution.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Products */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Top 10 Products by Sales</CardTitle>
          <CardDescription>Highest performing products</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.top_products} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                <Legend />
                <Bar dataKey="sales" fill="#3b82f6" name="Sales ($)" />
                <Bar dataKey="quantity" fill="#10b981" name="Qty" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Customer Segmentation Clustering */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Customer Segments (Clustering)</CardTitle>
            <CardDescription>Distribution across identified clusters</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.clustering_summary}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="cluster" />
                  <YAxis />
                  <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                  <Legend />
                  <Bar dataKey="count" fill="#8b5cf6" name="Customer Count" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Clustering Images */}
        {data.clustering_images && data.clustering_images.length > 0 && (
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Product Clustering Visualizations</CardTitle>
              <CardDescription>Interactive cluster analysis by category and tab</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {data.clustering_images.map((cluster, idx) => (
                  <div key={idx} className="border border-border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-foreground">{cluster.name}</h4>
                      <img
                        src={cluster.image}
                        alt={cluster.name}
                        className="w-20 h-20 object-cover rounded border"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {cluster.summary && cluster.summary.length > 0 && (
                        <div className="grid grid-cols-2 gap-2 text-xs mb-2">
                          <span>Clusters: {cluster.summary.length}</span>
                          <span>Total Products: {cluster.summary.reduce((sum, s) => sum + (s.n || 0), 0)}</span>
                        </div>
                      )}
                      {cluster.summary && cluster.summary.length > 0 && (
                        <div className="space-y-1">
                          {cluster.summary.map((s, idx) => (
                            <div key={idx} className="text-xs bg-muted/50 p-2 rounded">
                              <div className="font-medium">Cluster {s.cluster}: {s.persona}</div>
                              <div className="text-muted-foreground">
                                {s.n} products • ₱{s.total_sales?.toLocaleString()} sales • {s.total_qty?.toLocaleString()} qty
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Market Basket Analysis */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Top Product Associations</CardTitle>
            <CardDescription>Items frequently bought together (top 5)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {data.mba_rules.slice(0, 5).map((rule, idx) => (
                <div key={idx} className="border-l-4 border-blue-500 pl-4 py-2">
                  <div className="text-sm font-semibold text-foreground">{rule.item_a}</div>
                  <div className="text-xs text-muted-foreground">→ {rule.item_b}</div>
                  <div className="flex gap-4 mt-1 text-xs">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">Lift: {rule.lift.toFixed(1)}x</span>
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded">
                      Conf: {rule.confidence_ab_pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
