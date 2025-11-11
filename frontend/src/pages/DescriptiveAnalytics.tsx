
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
  LineChart,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DollarSign, TrendingUp, Package, AlertCircle, Loader2, ZoomIn, ZoomOut, Filter } from "lucide-react"

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
    summary: Array<{ cluster: number; n: number; total_sales: number; total_qty: number; avg_price_med: number; months_active_med: number; mean_monthly_qty_med: number; cv_monthly_qty_med: number; trend_qty_slope_med: number; persona: string }>
  }>
  mba_rules: Array<{
    item_a: string
    item_b: string
    lift: number
    confidence_ab_pct: number
    support_pct: number
  }>
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
  const [zoomLevel, setZoomLevel] = useState(1)

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

  // Filter data based on selections
  const filteredMonthlySales = data.monthly_sales.filter(item => {
    if (selectedTimeRange === "last6") return true // Implement time filtering if needed
    return true
  })

  const filteredCategoryDistribution = data.category_distribution.filter(item => {
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

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Descriptive Analytics</h1>
          <p className="text-muted-foreground">Comprehensive sales trends, product distribution, and customer insights</p>
        </div>
        <div className="flex gap-2">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="last6">Last 6 Months</SelectItem>
            </SelectContent>
          </Select>
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Categories</SelectItem>
              {data.category_distribution.map(cat => (
                <SelectItem key={cat.name} value={cat.name}>{cat.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex gap-1">
            <Button variant="outline" size="sm" onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.1))}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={() => setZoomLevel(Math.min(2, zoomLevel + 0.1))}>
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
        </div>
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

      {/* Tabs for different chart views */}
      <Tabs defaultValue="trends" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="trends">Sales Trends</TabsTrigger>
          <TabsTrigger value="products">Products</TabsTrigger>
          <TabsTrigger value="clustering">Clustering</TabsTrigger>
          <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-6">
          {/* Charts Grid */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Monthly Sales Trend */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Monthly Sales & Quantity Trend</CardTitle>
                <CardDescription>Sales and quantity sold by month (filtered by selections)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80" style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={filteredMonthlySales}>
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

            {/* Monthly Growth Rate */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Monthly Sales Growth Rate</CardTitle>
                <CardDescription>Percentage growth in sales over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80" style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data.monthly_growth}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Line type="monotone" dataKey="growth_rate" stroke="#f59e0b" name="Growth Rate (%)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Additional Trends */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Active SKUs Trend */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Active SKUs Over Time</CardTitle>
                <CardDescription>Number of active product SKUs by month</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80" style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.active_skus_trend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar dataKey="active_skus" fill="#8b5cf6" name="Active SKUs" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Yearly Sales Trend */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Yearly Sales & Quantity Trend</CardTitle>
                <CardDescription>Sales and quantity sold by year</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80" style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data.yearly_sales}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="year" />
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
          </div>
        </TabsContent>

        <TabsContent value="products" className="space-y-6">
          {/* Product Analysis */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Category Distribution */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Category Distribution</CardTitle>
                <CardDescription>Sales distribution across product categories</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={filteredCategoryDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {filteredCategoryDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Top Products by Sales */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Top Products by Sales</CardTitle>
                <CardDescription>Top 10 products by total sales value</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.top_products_sales.slice(0, 10)} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={100} />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar dataKey="sales" fill="#3b82f6" name="Sales ($)" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Top Products by Quantity */}
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle>Top Products by Quantity Sold</CardTitle>
              <CardDescription>Top 10 products by total quantity sold</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.top_products_qty.slice(0, 10)} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={100} />
                    <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                    <Bar dataKey="quantity" fill="#10b981" name="Quantity" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clustering" className="space-y-6">
          {/* Clustering Analysis */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Global Clustering Summary */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Global Product Clusters</CardTitle>
                <CardDescription>Distribution of products across identified clusters</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.clustering_summary}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="cluster" />
                      <YAxis />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar dataKey="count" fill="#8b5cf6" name="Product Count" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Clustering Images */}
            {data.clustering_images && data.clustering_images.length > 0 && (
              <Card className="shadow-soft">
                <CardHeader>
                  <CardTitle>Cluster Visualizations</CardTitle>
                  <CardDescription>Interactive cluster analysis by category and tab</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-96 overflow-y-auto">
                    {data.clustering_images.map((cluster, idx) => (
                      <div key={idx} className="border border-border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold text-foreground">{cluster.name}</h4>
                          {cluster.image && (
                            <img
                              src={cluster.image}
                              alt={cluster.name}
                              className="w-32 h-32 object-cover rounded border"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none'
                              }}
                            />
                          )}
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
          </div>

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
                    <div className="text-xs text-muted-foreground">to {rule.item_b}</div>
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
        </TabsContent>

        <TabsContent value="visualizations" className="space-y-6">
          {/* KPI Visualizations */}
          <div className="grid gap-6 lg:grid-cols-2">
            {data.kpi_images.monthly_sales_growth_rate && (
              <Card className="shadow-soft">
                <CardHeader>
                  <CardTitle>Monthly Sales Growth Rate Chart</CardTitle>
                  <CardDescription>Visual representation of sales growth over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={data.kpi_images.monthly_sales_growth_rate}
                    alt="Monthly Sales Growth Rate"
                    className="w-full h-80 object-contain rounded border"
                  />
                </CardContent>
              </Card>
            )}

            {data.kpi_images.sales_month_vs_year && (
              <Card className="shadow-soft">
                <CardHeader>
                  <CardTitle>Sales Month vs Year Comparison</CardTitle>
                  <CardDescription>Comparative analysis of sales by month and year</CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={data.kpi_images.sales_month_vs_year}
                    alt="Sales Month vs Year"
                    className="w-full h-80 object-contain rounded border"
                  />
                </CardContent>
              </Card>
            )}
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {data.kpi_images.qty_month_vs_year && (
              <Card className="shadow-soft">
                <CardHeader>
                  <CardTitle>Quantity Month vs Year Comparison</CardTitle>
                  <CardDescription>Comparative analysis of quantity sold by month and year</CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={data.kpi_images.qty_month_vs_year}
                    alt="Quantity Month vs Year"
                    className="w-full h-80 object-contain rounded border"
                  />
                </CardContent>
              </Card>
            )}

            {/* Seasonal Index */}
            <Card className="shadow-soft">
              <CardHeader>
                <CardTitle>Seasonal Index by Category</CardTitle>
                <CardDescription>Seasonal patterns in sales across categories</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.seasonal_index}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="category" />
                      <YAxis />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar dataKey="season_index" fill="#ef4444" name="Seasonal Index" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>



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
                      {cluster.image && (
                        <img
                          src={cluster.image}
                          alt={cluster.name}
                          className="w-32 h-32 object-cover rounded border"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none'
                          }}
                        />
                      )}
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
                  <div className="text-xs text-muted-foreground">to {rule.item_b}</div>
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
