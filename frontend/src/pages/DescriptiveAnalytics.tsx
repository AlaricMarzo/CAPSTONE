  "use client"

import { useState, useEffect } from "react"
import {
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  LineChart,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { DollarSign, TrendingUp, Package, AlertCircle, Loader2, Search, Filter } from "lucide-react"

interface DescriptiveData {
  kpi_summary: {
    total_sales: number
    total_quantity: number
    active_skus: number
    growth_rate: number
  }
  monthly_sales: Array<{ month: string; sales: number; quantity: number }>
  monthly_growth: Array<{ month: string; growth_rate: number }>
  active_skus_trend: Array<{ month: string; active_skus: number }>
  yearly_sales: Array<{ year: string; sales: number; quantity: number }>
  category_distribution: Array<{ name: string; value: number }>
  tab_distribution: Array<{ name: string; value: number }>
  seasonal_index: Array<{ category: string; season_index: number }>
  top_products_sales: Array<{ name: string; sales: number; quantity: number }>
  top_products_qty: Array<{ name: string; sales: number; quantity: number }>
  clustering_summary: Array<{ cluster: number; count: number; avg_value: number }>
  clustering_images: Array<{
    name: string
    image: string
    summary: Array<{
      cluster: number
      n: number
      total_sales: number
      total_qty: number
      avg_price_med: number
      months_active_med: number
      mean_monthly_qty_med: number
      cv_monthly_qty_med: number
      trend_qty_slope_med: number
      persona: string
    }>
  }>
  mba_rules: Array<{
    item_a: string
    item_b: string
    lift: number
    confidence_ab_pct: number
    support_pct: number
  }>
  mba_images: {
    top20_lift: string | null
    top20_confidence_a_to_b: string | null
  }
  kpi_images: {
    monthly_sales_growth_rate: string | null
    sales_month_vs_year: string | null
    qty_month_vs_year: string | null
  }
}

export default function DescriptiveAnalytics() {
  const [data, setData] = useState<DescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTimeRange, setSelectedTimeRange] = useState("all")
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [isMobile, setIsMobile] = useState(false)
  const [expandedProductSales, setExpandedProductSales] = useState(false)
  const [expandedProductQty, setExpandedProductQty] = useState(false)
  const [clusteringFilter, setClusteringFilter] = useState("all")
  const [clusteringSearch, setClusteringSearch] = useState("")

  useEffect(() => {
    const handleResize = () => {  
      setIsMobile(window.innerWidth < 768)
    }

    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

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
      <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading descriptive analytics...
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
        <div className="text-destructive">Error: {error || "No data available"}</div>
      </div>
    )
  }

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

  // Filter data based on selections
  const filteredMonthlySales = data.monthly_sales.filter((item) => {
    if (selectedTimeRange === "last6") return true
    return true
  })

  const filteredCategoryDistribution = data.category_distribution.filter((item) => {
    if (selectedCategory === "all") return true
    return item.name === selectedCategory
  })

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-PH", {
      style: "currency",
      currency: "PHP",
      minimumFractionDigits: 0,
    }).format(value)
  }

  const kpiCards = [
    {
      title: "Total Sales",
      value: `₱${(data.kpi_summary.total_sales / 1000000).toFixed(2)}M`,
      change: `+${data.kpi_summary.growth_rate.toFixed(1)}%`,
      icon: DollarSign,
      color: "success" as const,
    },
    {
      title: "Total Quantity Sold",
      value: (data.kpi_summary.total_quantity / 1000).toFixed(0) + "K",
      change: "units",
      icon: Package,
      color: "info" as const,
    },
    {
      title: "Active SKUs",
      value: data.kpi_summary.active_skus.toString(),
      change: "products",
      icon: TrendingUp,
      color: "warning" as const,
    },
    {
      title: "Growth Rate",
      value: `${data.kpi_summary.growth_rate.toFixed(1)}%`,
      change: "YoY",
      icon: AlertCircle,
      color: "success" as const,
    },
  ]

  const getChartHeight = () => (isMobile ? 350 : 450)

  const calculateLeftMargin = (products: Array<{ name: string }>) => {
    if (!products || products.length === 0) return isMobile ? 80 : 120
    const maxLength = Math.max(...products.map((p) => p.name.length))
    const charWidth = isMobile ? 6 : 8
    return Math.min(Math.max(maxLength * charWidth, isMobile ? 80 : 120), 200)
  }

  return (
    <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-foreground">Descriptive Analytics</h1>
          <p className="text-sm md:text-base text-muted-foreground mt-1">
            Comprehensive sales trends, product distribution, and customer insights
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:gap-2">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-full sm:w-32">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="last6">Last 6 Months</SelectItem>
            </SelectContent>
          </Select>
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-full sm:w-40">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              {data.category_distribution.map((cat) => (
                <SelectItem key={cat.name} value={cat.name}>
                  {cat.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* KPI Cards - Responsive Grid */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
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

      {/* Tabs for different chart views */}
      <Tabs defaultValue="trends" className="w-full">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 gap-2">
          <TabsTrigger value="trends">Sales Trends (KPI)</TabsTrigger>
          <TabsTrigger value="products">MBA</TabsTrigger>
          <TabsTrigger value="clustering">Clustering</TabsTrigger>
          <TabsTrigger value="visualizations">Visuals</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-6">
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {/* Monthly Sales Trend */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Monthly Sales & Quantity</CardTitle>
                <CardDescription className="text-xs md:text-sm">Sales and quantity sold by month</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={filteredMonthlySales}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 0 : 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="month" tick={{ fontSize: isMobile ? 11 : 12 }} />
                      <YAxis yAxisId="left" tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: isMobile ? 11 : 12 }}
                        width={isMobile ? 35 : 40}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                      />
                      <Legend wrapperStyle={{ paddingTop: "10px", fontSize: isMobile ? "11px" : "12px" }} />
                      <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales ($)" radius={[4, 4, 0, 0]} />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="quantity"
                        stroke="#10b981"
                        name="Quantity"
                        strokeWidth={2}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Monthly Growth Rate */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Sales Growth Rate</CardTitle>
                <CardDescription className="text-xs md:text-sm">Percentage growth in sales over time</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={data.monthly_growth}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 0 : 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="month" tick={{ fontSize: isMobile ? 11 : 12 }} />
                      <YAxis tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="growth_rate"
                        stroke="#f59e0b"
                        name="Growth Rate (%)"
                        strokeWidth={2}
                        dot={{ fill: "#f59e0b", r: 4 }}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Additional Trends */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {/* Active SKUs Trend */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Active SKUs Over Time</CardTitle>
                <CardDescription className="text-xs md:text-sm">Number of active product SKUs</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.active_skus_trend}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 0 : 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="month" tick={{ fontSize: isMobile ? 11 : 12 }} />
                      <YAxis tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                      />
                      <Bar dataKey="active_skus" fill="#8b5cf6" name="Active SKUs" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Yearly Sales Trend */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Yearly Sales & Quantity</CardTitle>
                <CardDescription className="text-xs md:text-sm">Sales and quantity sold by year</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={data.yearly_sales}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 0 : 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="year" tick={{ fontSize: isMobile ? 11 : 12 }} />
                      <YAxis yAxisId="left" tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        tick={{ fontSize: isMobile ? 11 : 12 }}
                        width={isMobile ? 35 : 40}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                      />
                      <Legend wrapperStyle={{ paddingTop: "10px", fontSize: isMobile ? "11px" : "12px" }} />
                      <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales ($)" radius={[4, 4, 0, 0]} />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="quantity"
                        stroke="#10b981"
                        name="Quantity"
                        strokeWidth={2}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Products */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {/* Top Products by Sales */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Top Products by Sales</CardTitle>
                <CardDescription className="text-xs md:text-sm">Top 10 products by total sales value</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                {data.top_products_sales && data.top_products_sales.length > 0 ? (
                  <div className="space-y-4">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-2 px-2 font-medium text-foreground">Rank</th>
                            <th className="text-left py-2 px-2 font-medium text-foreground">Product</th>
                            <th className="text-right py-2 px-2 font-medium text-foreground">Sales</th>
                            <th className="text-right py-2 px-2 font-medium text-foreground">Quantity</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(expandedProductSales
                            ? data.top_products_sales.slice(0, 10)
                            : data.top_products_sales.slice(0, 5)
                          ).map((product, index) => (
                            <tr key={index} className="border-b border-border/50 hover:bg-muted/50">
                              <td className="py-2 px-2 text-muted-foreground">#{index + 1}</td>
                              <td className="py-2 px-2 text-foreground max-w-xs truncate" title={product.name}>
                                {product.name}
                              </td>
                              <td className="py-2 px-2 text-right font-medium text-foreground">
                                ₱{product.sales.toLocaleString()}
                              </td>
                              <td className="py-2 px-2 text-right text-muted-foreground">
                                {product.quantity.toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <button
                      onClick={() => setExpandedProductSales(!expandedProductSales)}
                      className="w-full px-4 py-2 text-sm font-medium text-foreground bg-muted hover:bg-muted/80 rounded transition-colors"
                    >
                      {expandedProductSales ? "Show Top 5" : "Show All 10"}
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-80 text-muted-foreground">
                    <AlertCircle className="h-8 w-8 mr-2" />
                    <span className="text-sm">No top products data available</span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Top Products by Quantity */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Top Products by Quantity</CardTitle>
                <CardDescription className="text-xs md:text-sm">Top 10 products by total quantity sold</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                {data.top_products_qty && data.top_products_qty.length > 0 ? (
                  <div className="space-y-4">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-2 px-2 font-medium text-foreground">Rank</th>
                            <th className="text-left py-2 px-2 font-medium text-foreground">Product</th>
                            <th className="text-right py-2 px-2 font-medium text-foreground">Quantity</th>
                            <th className="text-right py-2 px-2 font-medium text-foreground">Sales</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(expandedProductQty
                            ? data.top_products_qty.slice(0, 10)
                            : data.top_products_qty.slice(0, 5)
                          ).map((product, index) => (
                            <tr key={index} className="border-b border-border/50 hover:bg-muted/50">
                              <td className="py-2 px-2 text-muted-foreground">#{index + 1}</td>
                              <td className="py-2 px-2 text-foreground max-w-xs truncate" title={product.name}>
                                {product.name}
                              </td>
                              <td className="py-2 px-2 text-right font-medium text-foreground">
                                {product.quantity.toLocaleString()}
                              </td>
                              <td className="py-2 px-2 text-right text-muted-foreground">
                                ₱{product.sales.toLocaleString()}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <button
                      onClick={() => setExpandedProductQty(!expandedProductQty)}
                      className="w-full px-4 py-2 text-sm font-medium text-foreground bg-muted hover:bg-muted/80 rounded transition-colors"
                    >
                      {expandedProductQty ? "Show Top 5" : "Show All 10"}
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-80 text-muted-foreground">
                    <AlertCircle className="h-8 w-8 mr-2" />
                    <span className="text-sm">No top products data available</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="products" className="space-y-6">
          {/* MBA Rules Table */}
          <Card className="shadow-soft overflow-hidden">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg md:text-base">Market Basket Analysis Rules</CardTitle>
              <CardDescription className="text-xs md:text-sm">
                Top association rules showing product relationships
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0 md:p-6">
              {data.mba_rules && data.mba_rules.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 px-2 font-medium text-foreground">Item A</th>
                        <th className="text-left py-2 px-2 font-medium text-foreground">Item B</th>
                        <th className="text-right py-2 px-2 font-medium text-foreground">Lift</th>
                        <th className="text-right py-2 px-2 font-medium text-foreground">Confidence (%)</th>
                        <th className="text-right py-2 px-2 font-medium text-foreground">Support (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.mba_rules.map((rule, index) => (
                        <tr key={index} className="border-b border-border/50 hover:bg-muted/50">
                          <td className="py-2 px-2 text-foreground max-w-xs truncate" title={rule.item_a}>
                            {rule.item_a}
                          </td>
                          <td className="py-2 px-2 text-foreground max-w-xs truncate" title={rule.item_b}>
                            {rule.item_b}
                          </td>
                          <td className="py-2 px-2 text-right font-medium text-foreground">
                            {rule.lift.toFixed(2)}
                          </td>
                          <td className="py-2 px-2 text-right text-foreground">
                            {rule.confidence_ab_pct.toFixed(1)}%
                          </td>
                          <td className="py-2 px-2 text-right text-muted-foreground">
                            {rule.support_pct.toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex items-center justify-center h-80 text-muted-foreground">
                  <AlertCircle className="h-8 w-8 mr-2" />
                  <span className="text-sm">No MBA rules data available</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Interactive MBA Charts */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {/* Top Rules by Lift */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Top Association Rules by Lift</CardTitle>
                <CardDescription className="text-xs md:text-sm">
                  Interactive chart showing strongest product associations
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.mba_rules
                        .sort((a, b) => b.lift - a.lift)
                        .slice(0, 20)
                        .map((rule, index) => ({
                          name: `${rule.item_a} → ${rule.item_b}`,
                          lift: rule.lift,
                          confidence: rule.confidence_ab_pct,
                          support: rule.support_pct,
                          rule: rule
                        }))}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 100 : 150 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: isMobile ? 8 : 10 }}
                        angle={-45}
                        textAnchor="end"
                        height={100}
                      />
                      <YAxis tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        formatter={(value, name, props) => [
                          `${value.toFixed(2)}`,
                          name === 'lift' ? 'Lift' : name
                        ]}
                        labelFormatter={(label) => `Rule: ${label}`}
                      />
                      <Bar dataKey="lift" fill="#3b82f6" name="Lift" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Top Rules by Confidence */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Top Association Rules by Confidence</CardTitle>
                <CardDescription className="text-xs md:text-sm">
                  Interactive chart showing highest confidence in product associations
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.mba_rules
                        .sort((a, b) => b.confidence_ab_pct - a.confidence_ab_pct)
                        .slice(0, 20)
                        .map((rule, index) => ({
                          name: `${rule.item_a} → ${rule.item_b}`,
                          confidence: rule.confidence_ab_pct,
                          lift: rule.lift,
                          support: rule.support_pct,
                          rule: rule
                        }))}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 100 : 150 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: isMobile ? 8 : 10 }}
                        angle={-45}
                        textAnchor="end"
                        height={100}
                      />
                      <YAxis tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        formatter={(value, name) => [
                          `${value.toFixed(1)}%`,
                          name === 'confidence' ? 'Confidence' : name
                        ]}
                        labelFormatter={(label) => `Rule: ${label}`}
                      />
                      <Bar dataKey="confidence" fill="#10b981" name="Confidence (%)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="visualizations" className="space-y-6">
          {/* KPI Visualizations */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {data.kpi_images.monthly_sales_growth_rate && (
              <Card className="shadow-soft overflow-hidden">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg md:text-base">Monthly Sales Growth</CardTitle>
                  <CardDescription className="text-xs md:text-sm">
                    Visual representation of sales growth
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0 md:p-6">
                  <div className="w-full overflow-x-auto">
                    <img
                      src={data.kpi_images.monthly_sales_growth_rate || "/placeholder.svg"}
                      alt="Monthly Sales Growth Rate"
                      className="w-full h-auto max-w-full object-contain rounded border"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {data.kpi_images.sales_month_vs_year && (
              <Card className="shadow-soft overflow-hidden">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg md:text-base">Sales Month vs Year</CardTitle>
                  <CardDescription className="text-xs md:text-sm">Comparative analysis of sales</CardDescription>
                </CardHeader>
                <CardContent className="p-0 md:p-6">
                  <div className="w-full overflow-x-auto">
                    <img
                      src={data.kpi_images.sales_month_vs_year || "/placeholder.svg"}
                      alt="Sales Month vs Year"
                      className="w-full h-auto max-w-full object-contain rounded border"
                    />
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
            {data.kpi_images.qty_month_vs_year && (
              <Card className="shadow-soft overflow-hidden">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg md:text-base">Quantity Month vs Year</CardTitle>
                  <CardDescription className="text-xs md:text-sm">Comparative analysis of quantity</CardDescription>
                </CardHeader>
                <CardContent className="p-0 md:p-6">
                  <div className="w-full overflow-x-auto">
                    <img
                      src={data.kpi_images.qty_month_vs_year || "/placeholder.svg"}
                      alt="Quantity Month vs Year"
                      className="w-full h-auto max-w-full object-contain rounded border"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Seasonal Index */}
            <Card className="shadow-soft overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg md:text-base">Seasonal Index</CardTitle>
                <CardDescription className="text-xs md:text-sm">Seasonal patterns by category</CardDescription>
              </CardHeader>
              <CardContent className="p-0 md:p-6">
                <div className="w-full" style={{ height: `${getChartHeight()}px` }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data.seasonal_index}
                      margin={{ top: 5, right: isMobile ? 5 : 30, bottom: 5, left: isMobile ? 60 : 80 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="category" tick={{ fontSize: isMobile ? 10 : 11 }} />
                      <YAxis tick={{ fontSize: isMobile ? 11 : 12 }} width={isMobile ? 35 : 40} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                        cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
                      />
                      <Bar dataKey="season_index" fill="#ef4444" name="Seasonal Index" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="clustering" className="space-y-6">
          {/* Clustering Filters */}
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-col gap-2 sm:flex-row sm:gap-2">
              <Select value={clusteringFilter} onValueChange={setClusteringFilter}>
                <SelectTrigger className="w-full sm:w-40">
                  <SelectValue placeholder="Filter by Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Clustering</SelectItem>
                  <SelectItem value="global">Global</SelectItem>
                  <SelectItem value="category">By Category</SelectItem>
                  <SelectItem value="tab">By Tab</SelectItem>
                </SelectContent>
              </Select>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search clustering..."
                  value={clusteringSearch}
                  onChange={(e) => setClusteringSearch(e.target.value)}
                  className="pl-10 w-full sm:w-64"
                />
              </div>
            </div>
          </div>

          {/* Clustering Images Grid */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-2 xl:grid-cols-3">
            {data.clustering_images
              .filter((item) => {
                const matchesFilter =
                  clusteringFilter === "all" ||
                  (clusteringFilter === "global" && item.name === "Global Clustering") ||
                  (clusteringFilter === "category" && item.name.includes("Clustering by Category")) ||
                  (clusteringFilter === "tab" && item.name.includes("Clustering by Tab"))

                const matchesSearch = clusteringSearch === "" ||
                  item.name.toLowerCase().includes(clusteringSearch.toLowerCase())

                return matchesFilter && matchesSearch
              })
              .map((clusteringItem, index) => (
                <Card key={index} className="shadow-soft overflow-hidden">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg md:text-base">{clusteringItem.name}</CardTitle>
                    <CardDescription className="text-xs md:text-sm">
                      Product clustering visualization with detailed metrics
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-0 md:p-6 space-y-4">
                    <div className="w-full overflow-x-auto">
                      <img
                        src={clusteringItem.image}
                        alt={clusteringItem.name}
                        className="w-full h-auto max-w-full object-contain rounded border"
                      />
                    </div>

                    {/* Clustering Summary Table */}
                    {clusteringItem.summary && clusteringItem.summary.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-foreground">Cluster Summary</h4>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-border">
                                <th className="text-left py-1 px-1 font-medium text-foreground">Cluster</th>
                                <th className="text-right py-1 px-1 font-medium text-foreground">Count</th>
                                <th className="text-right py-1 px-1 font-medium text-foreground">Avg Sales</th>
                                <th className="text-right py-1 px-1 font-medium text-foreground">Avg Qty</th>
                                <th className="text-left py-1 px-1 font-medium text-foreground">Persona</th>
                              </tr>
                            </thead>
                            <tbody>
                              {clusteringItem.summary.slice(0, 5).map((summary, idx) => (
                                <tr key={idx} className="border-b border-border/50">
                                  <td className="py-1 px-1 text-muted-foreground">{summary.cluster}</td>
                                  <td className="py-1 px-1 text-right text-foreground">{summary.n}</td>
                                  <td className="py-1 px-1 text-right text-foreground">
                                    ₱{summary.total_sales?.toLocaleString() || "0"}
                                  </td>
                                  <td className="py-1 px-1 text-right text-foreground">
                                    {summary.total_qty?.toLocaleString() || "0"}
                                  </td>
                                  <td className="py-1 px-1 text-foreground max-w-20 truncate" title={summary.persona}>
                                    {summary.persona || "N/A"}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
          </div>

          {data.clustering_images.filter((item) => {
            const matchesFilter =
              clusteringFilter === "all" ||
              (clusteringFilter === "global" && item.name === "Global Clustering") ||
              (clusteringFilter === "category" && item.name.includes("Clustering by Category")) ||
              (clusteringFilter === "tab" && item.name.includes("Clustering by Tab"))

            const matchesSearch = clusteringSearch === "" ||
              item.name.toLowerCase().includes(clusteringSearch.toLowerCase())

            return matchesFilter && matchesSearch
          }).length === 0 && (
            <div className="flex items-center justify-center h-80 text-muted-foreground">
              <Filter className="h-8 w-8 mr-2" />
              <span className="text-sm">No clustering data matches the current filters</span>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
