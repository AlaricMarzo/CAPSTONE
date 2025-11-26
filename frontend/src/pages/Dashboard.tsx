  import { useState, useEffect } from "react"
import { Sidebar } from "@/components/Layout/Sidebar"
import { MetricCard } from "@/components/Dashboard/MetricCard"

import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard"
import { TrendingUp, DollarSign, Package, ShoppingCart, User, FileText, Loader2, AlertCircle, Search } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

import { getUserMetrics } from "@/data/mockData"
import DataUploadPage from "./upload"
import ProfilePage from "./Profile"
import DescriptiveAnalytics from "./DescriptiveAnalytics"
import PredictiveAnalytics from "./PredictiveAnalytics"
import PrescriptiveAnalytics from "./PrescriptiveAnalytics"
import InventoryRecommendations from "./InventoryRecommendations"
import ReportsPage from "./Reports"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ComposedChart,
  Tooltip,
  Legend,
} from "recharts"

interface PrescriptiveData {
  reorder_points: any[]
  eoq_data: any[]
  allocations: any[]
  discount_groups: any[]
  resource_planning: any[]
  anomalies: any[]
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
}

interface DescriptiveData {
  kpi_summary: {
    total_sales: number
    total_quantity: number
    active_skus: number
    growth_rate: number
  }
  monthly_sales: any[]
  monthly_growth: { month: string; growth_rate: number }[]
  category_distribution: { name: string; value: number }[]
  top_products_sales: any[]
  clustering_summary: any[]
  mba_rules: any[]
}

interface PredictiveData {
  models_summary: {
    total_models: number
    avg_accuracy: number
    total_forecasts: number
  }
  forecast_data: any[]
  model_performance: any
}

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }): JSX.Element {
  const [descriptiveData, setDescriptiveData] = useState<DescriptiveData | null>(null)
  const [predictiveData, setPredictiveData] = useState<PredictiveData | null>(null)
  const [prescriptiveData, setPrescriptiveData] = useState<PrescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [showChart, setShowChart] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")


  const fetchAllData = async () => {
    try {
      setLoading(true)
      const [descRes, predRes, prescRes] = await Promise.all([
        fetch("http://localhost:5050/api/analytics/descriptive"),
        fetch("http://localhost:5050/api/analytics/predictive"),
        fetch("http://localhost:5050/api/analytics/prescriptive")
      ])

      if (!descRes.ok) throw new Error("Failed to fetch descriptive data")
      if (!predRes.ok) throw new Error("Failed to fetch predictive data")
      if (!prescRes.ok) throw new Error("Failed to fetch prescriptive data")

      const descResult = await descRes.json()
      const predResult = await predRes.json()
      const prescResult = await prescRes.json()

      setDescriptiveData(descResult.data)
      setPredictiveData(predResult.data)
      setPrescriptiveData(prescResult.data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      console.error("[v0] Error fetching analytics data:", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAllData()

    // Listen for analytics update events
    const handleAnalyticsUpdate = () => {
      fetchAllData()
    }

    window.addEventListener("analyticsUpdated", handleAnalyticsUpdate)

    return () => {
      window.removeEventListener("analyticsUpdated", handleAnalyticsUpdate)
    }
  }, [])

  const confirmLogout = () => {
    onLogout()
    setConfirmOpen(false)
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-PH", {
      style: "currency",
      currency: "PHP",
      minimumFractionDigits: 0,
    }).format(value)
  }

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading dashboard data...
        </div>
      </div>
    )
  }

  if (error) {
    const fallbackMetrics = getUserMetrics()
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
            <p className="text-muted-foreground">Real-time business insights and analytics</p>
          </div>
          <div className="flex items-center gap-4">

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <User className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={onProfileClick}>
                  <User className="mr-2 h-4 w-4" />
                  Profile
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive cursor-pointer"
                  onSelect={(e) => {
                    e.preventDefault()
                    setConfirmOpen(true)
                  }}
                >
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                  <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>



        <div className="text-destructive flex items-center gap-2 mb-4">
          Error loading analytics data: {error}. Showing fallback data.
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Revenue"
            value={`₱${fallbackMetrics.totalRevenue.toFixed(2)}`}
            change="+12.5% from last month"
            changeType="positive"
            icon={DollarSign}
            color="success"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Total Sales"
            value={fallbackMetrics.totalSales.toString()}
            change="+8.2% from last month"
            changeType="positive"
            icon={ShoppingCart}
            color="info"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Low Stock Items"
            value={fallbackMetrics.lowStockItems.toString()}
            change={`${fallbackMetrics.outOfStockItems} out of stock`}
            changeType="warning"
            icon={Package}
            color="warning"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Avg Order Value"
            value={`₱${fallbackMetrics.averageOrderValue.toFixed(2)}`}
            change="+5.1% from last month"
            changeType="positive"
            icon={TrendingUp}
            color="default"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
        </div>

        <div className="grid gap-6 lg:grid-cols-1">
          <SalesTrendsChart />
        </div>
      </div>
    )
  }

  if (!prescriptiveData || !prescriptiveData.financial_summary || !prescriptiveData.reorder_points ||
      !descriptiveData || !predictiveData) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
            <p className="text-muted-foreground">Real-time business insights and analytics</p>
          </div>
        <div className="flex items-center gap-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onProfileClick}>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive cursor-pointer"
                onSelect={(e) => {
                  e.preventDefault()
                  setConfirmOpen(true)
                }}
              >
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
        </div>



        <div className="text-muted-foreground flex items-center gap-2 mb-4">
          Analytics data not available. Please upload data and wait for processing to complete.
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Revenue"
            value="₱0"
            change="No data"
            changeType="positive"
            icon={DollarSign}
            color="success"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Total Sales"
            value="0"
            change="No data"
            changeType="positive"
            icon={ShoppingCart}
            color="info"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Active SKUs"
            value="0"
            change="No data"
            changeType="positive"
            icon={Package}
            color="warning"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
          <MetricCard
            title="Forecast Accuracy"
            value="0%"
            change="No data"
            changeType="positive"
            icon={TrendingUp}
            color="default"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
        </div>

        <div className="grid gap-6 lg:grid-cols-1">
          <SalesTrendsChart />
        </div>
      </div>
    )
  }

  // Extract key metrics from all analytics
  const { financial_summary, reorder_points, key_metrics, resource_planning } = prescriptiveData
  const descKpi = descriptiveData.kpi_summary
  const predSummary = predictiveData.models_summary

  const totalRevenue = financial_summary.total_sales // Optimized revenue from prescriptive
  const totalSalesAmount = descKpi.total_sales // Historical sales from descriptive
  const totalQuantitySold = descKpi.total_quantity // Use descriptive total quantity
  const lowStockItems = reorder_points.length
  const avgOrderValue = totalQuantitySold > 0 ? totalRevenue / totalQuantitySold : 0
  const forecastAccuracy = predSummary.avg_accuracy * 100
  const activeSkus = descKpi.active_skus
  const growthRate = descKpi.growth_rate

  const filteredResourceData = prescriptiveData?.resource_planning || []

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Comprehensive business insights from descriptive, predictive, and prescriptive analytics</p>
        </div>
        <div className="flex items-center gap-4">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onProfileClick}>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive cursor-pointer"
                onSelect={(e) => {
                  e.preventDefault()
                  setConfirmOpen(true)
                }}
              >
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>



      {/* Key Performance Indicators */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Revenue"
          value={formatCurrency(totalRevenue)}
          change={`${financial_summary.overall_profit_margin_pct.toFixed(1)}% margin`}
          changeType="positive"
          icon={DollarSign}
          color="success"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Total Sales"
          value={formatCurrency(totalSalesAmount)}
          change={`${growthRate.toFixed(1)}% growth`}
          changeType="positive"
          icon={ShoppingCart}
          color="info"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Active SKUs"
          value={activeSkus.toString()}
          change={`${growthRate.toFixed(1)}% growth`}
          changeType="positive"
          icon={Package}
          color="warning"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Forecast Accuracy"
          value={`${forecastAccuracy.toFixed(1)}%`}
          change={`${predSummary.total_models} models`}
          changeType="positive"
          icon={TrendingUp}
          color="default"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
      </div>

      {/* Analytics Summary Cards */}
      <div className="grid gap-6 md:grid-cols-3">
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-blue-500" />
              Descriptive Analytics
            </CardTitle>
            <CardDescription>Historical data insights and patterns</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Total Sales:</span>
                <span className="font-medium">{formatCurrency(descKpi.total_sales)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Total Quantity:</span>
                <span className="font-medium">{descKpi.total_quantity.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Customer Segments:</span>
                <span className="font-medium">{descriptiveData.clustering_summary.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">MBA Rules:</span>
                <span className="font-medium">{descriptiveData.mba_rules.length}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              Predictive Analytics
            </CardTitle>
            <CardDescription>Forecasting and trend predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Models Trained:</span>
                <span className="font-medium">{predSummary.total_models}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Forecast Points:</span>
                <span className="font-medium">{predSummary.total_forecasts}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Avg Accuracy:</span>
                <span className="font-medium">{(predSummary.avg_accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Best Model:</span>
                <span className="font-medium">Gradient</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5 text-orange-500" />
              Prescriptive Analytics
            </CardTitle>
            <CardDescription>Actionable recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Items to Reorder:</span>
                <span className="font-medium">{lowStockItems}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Cost Savings:</span>
                <span className="font-medium">{formatCurrency(financial_summary.total_profit)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Optimized Products:</span>
                <span className="font-medium">{key_metrics.total_products_optimized}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Models Applied:</span>
                <span className="font-medium">{key_metrics.total_models}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts and Trends */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Monthly Sales Trend */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Monthly Sales Trend</CardTitle>
            <CardDescription>Sales performance over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={descriptiveData.monthly_sales.slice(-12)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any, name: string) => [
                      name === 'sales' ? formatCurrency(value) : value.toLocaleString(),
                      name === 'sales' ? 'Sales' : 'Quantity'
                    ]}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales" radius={[4, 4, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="quantity" stroke="#10b981" name="Quantity" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Performing Products */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Top Performing Products</CardTitle>
            <CardDescription>Highest revenue generating products</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {descriptiveData.top_products_sales.slice(0, 5).map((product: any, idx: number) => (
                <div key={idx} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center text-xs font-medium">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{product.name}</p>
                      <p className="text-xs text-muted-foreground">{product.quantity} units</p>
                    </div>
                  </div>
                  <span className="font-medium">{formatCurrency(product.sales)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Analytics Charts */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Category Distribution */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Category Distribution</CardTitle>
            <CardDescription>Sales distribution by product category</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={descriptiveData.category_distribution.slice(0, 8)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={80} fontSize={12} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any) => [value.toLocaleString(), 'Sales']}
                  />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Sales Growth Trend */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Sales Growth Trend</CardTitle>
            <CardDescription>Monthly sales and quantity performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={descriptiveData.monthly_sales.slice(-12)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any, name: string) => [
                      name === 'sales' ? formatCurrency(value) : value.toLocaleString(),
                      name === 'sales' ? 'Sales' : 'Quantity'
                    ]}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales" radius={[4, 4, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="quantity" stroke="#10b981" name="Quantity" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Performing Products */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Top Performing Products</CardTitle>
            <CardDescription>Highest revenue generating products</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {descriptiveData.top_products_sales.slice(0, 5).map((product: any, idx: number) => (
                <div key={idx} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center text-xs font-medium">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{product.name}</p>
                      <p className="text-xs text-muted-foreground">{product.quantity} units</p>
                    </div>
                  </div>
                  <span className="font-medium">{formatCurrency(product.sales)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Resource Planning Section */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Resource Planning
            <div className="flex items-center gap-2">

              <Button variant="outline" size="sm" onClick={() => setShowChart(!showChart)}>
                {showChart ? 'View List' : 'View Chart'}
              </Button>
            </div>
          </CardTitle>
          <CardDescription>Resource requirements and planning for inventory management</CardDescription>
        </CardHeader>
        <CardContent>
          {showChart ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={filteredResourceData.slice(0, 10)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis type="number" />
                  <YAxis dataKey="medicine" type="category" width={150} fontSize={12} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any, name: string) => [
                      name === 'storage_needed' ? value.toLocaleString() : formatCurrency(value),
                      name === 'storage_needed' ? 'Storage Needed' : 'Capital Needed'
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="storage_needed" fill="#ef4444" name="Storage Needed" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="capital_needed" fill="#eab308" name="Capital Needed" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Medicine</th>
                      <th className="text-right p-2">Daily Demand</th>
                      <th className="text-right p-2">Projected Demand</th>
                      <th className="text-right p-2">Storage Needed</th>
                      <th className="text-right p-2">Capital Needed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredResourceData.slice(0, 15).map((item: any, idx: number) => (
                      <tr key={idx} className="border-b hover:bg-muted/50">
                        <td className="p-2 font-medium">{item.medicine}</td>
                        <td className="p-2 text-right">{item.daily_demand.toLocaleString()}</td>
                        <td className="p-2 text-right">{item.projected_demand.toLocaleString()}</td>
                        <td className="p-2 text-right">{item.storage_needed.toLocaleString()}</td>
                        <td className="p-2 text-right">{formatCurrency(item.capital_needed)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filteredResourceData.length > 15 && (
                <div className="text-center text-sm text-muted-foreground">
                  Showing first 15 of {filteredResourceData.length} items.
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>


    </div>
  )
}

function SalesPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchSalesData = async () => {
      try {
        setLoading(true)
          const response = await fetch("http://localhost:5050/api/analytics/descriptive")
        if (!response.ok) throw new Error("Failed to fetch sales data")
        const result = await response.json()
        setData(result.data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred")
        console.error("[v0] Error fetching sales data:", err)
      } finally {
        setLoading(false)
      }
    }

    fetchSalesData()
  }, [])

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading sales data...
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="text-destructive">Error loading sales data: {error || 'No data available'}</div>
      </div>
    )
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "PHP",
      minimumFractionDigits: 0,
    }).format(value)
  }

  const kpiCards = [
    {
      title: "Total Sales",
      value: formatCurrency(data.kpi_summary.total_sales),
      change: `+${data.kpi_summary.growth_rate.toFixed(1)}% growth`,
      changeType: "positive" as const,
      icon: DollarSign,
      color: "success" as const,
    },
    {
      title: "Total Quantity Sold",
      value: data.kpi_summary.total_quantity.toLocaleString(),
      change: "units sold",
      changeType: "positive" as const,
      icon: Package,
      color: "info" as const,
    },
    {
      title: "Active SKUs",
      value: data.kpi_summary.active_skus.toString(),
      change: "products",
      changeType: "positive" as const,
      icon: TrendingUp,
      color: "warning" as const,
    },
    {
      title: "Monthly Growth Rate",
      value: `${data.kpi_summary.growth_rate.toFixed(1)}%`,
      change: "average",
      changeType: "positive" as const,
      icon: AlertCircle,
      color: "success" as const,
    },
  ]

  // Calculate yearly totals for comparison
  const yearlyData = data.monthly_sales.reduce((acc: any, item: any) => {
    const year = item.month.substring(0, 4)
    if (!acc[year]) {
      acc[year] = { year, total_sales: 0, total_quantity: 0, months: 0 }
    }
    acc[year].total_sales += item.sales
    acc[year].total_quantity += item.quantity
    acc[year].months += 1
    return acc
  }, {})

  const yearlyComparison = Object.values(yearlyData).sort((a: any, b: any) => b.year - a.year)

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Sales Dashboard</h2>
        <p className="text-muted-foreground">Comprehensive sales analytics and performance insights</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {kpiCards.map((card, idx) => (
          <MetricCard
            key={idx}
            title={card.title}
            value={card.value}
            change={card.change}
            changeType={card.changeType}
            icon={card.icon}
            color={card.color}
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Monthly Sales Trend */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Monthly Sales & Quantity Trend</CardTitle>
            <CardDescription>Sales performance and quantity sold by month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data.monthly_sales}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any, name: string) => [
                      name === 'sales' ? formatCurrency(value) : value.toLocaleString(),
                      name === 'sales' ? 'Sales' : 'Quantity'
                    ]}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="sales" fill="#3b82f6" name="Sales" radius={[4, 4, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="quantity" stroke="#10b981" name="Quantity" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Yearly Comparison */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Year-over-Year Performance</CardTitle>
            <CardDescription>Annual sales and quantity comparison</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={yearlyComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="year" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                    formatter={(value: any, name: string) => [
                      name === 'total_sales' ? formatCurrency(value) : value.toLocaleString(),
                      name === 'total_sales' ? 'Sales' : 'Quantity'
                    ]}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="total_sales" fill="#3b82f6" name="Sales" radius={[4, 4, 0, 0]} />
                  <Bar yAxisId="right" dataKey="total_quantity" fill="#10b981" name="Quantity" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Products */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Top 10 Products by Sales</CardTitle>
          <CardDescription>Highest performing products by revenue</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.top_products_sales.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" tickFormatter={(value) => formatCurrency(value)} />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip
                  contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }}
                  formatter={(value: any, name: string) => [
                    name === 'sales' ? formatCurrency(value) : value.toLocaleString(),
                    name === 'sales' ? 'Sales' : 'Quantity'
                  ]}
                />
                <Legend />
                <Bar dataKey="sales" fill="#3b82f6" name="Sales" radius={[0, 4, 4, 0]} />
                <Bar dataKey="quantity" fill="#10b981" name="Quantity" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Sales Summary Table */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Monthly Sales Summary</CardTitle>
          <CardDescription>Detailed breakdown of sales performance by month</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Month</th>
                  <th className="text-right p-2">Sales</th>
                  <th className="text-right p-2">Quantity</th>
                  <th className="text-right p-2">Avg Price</th>
                </tr>
              </thead>
              <tbody>
                {data.monthly_sales.map((item: any, idx: number) => (
                  <tr key={idx} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-medium">{item.month}</td>
                    <td className="p-2 text-right">{formatCurrency(item.sales)}</td>
                    <td className="p-2 text-right">{item.quantity.toLocaleString()}</td>
                    <td className="p-2 text-right">{formatCurrency(item.sales / item.quantity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function InventoryPage() {
  return <InventoryRecommendations />
}

// Create a Dashboard component that combines all the pages
interface DashboardProps {
  onLogout: () => void
  userEmail: string
}

export default function Dashboard({ onLogout, userEmail }: DashboardProps) {
  const [activeTab, setActiveTab] = useState("home")
  const [showProfile, setShowProfile] = useState(false)

  const handleProfileClick = () => {
    setShowProfile(true)
    setActiveTab("profile")
  }

  const handleBackToDashboard = () => {
    setShowProfile(false)
    setActiveTab("home")
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          {activeTab === "home" && <HomePage onProfileClick={handleProfileClick} onLogout={onLogout} />}
          {activeTab === "sales" && <SalesPage />}
          {activeTab === "inventory" && <InventoryPage />}
          {activeTab === "upload" && <DataUploadPage />}
          {activeTab === "descriptive" && <DescriptiveAnalytics />}
          {activeTab === "predictive" && <PredictiveAnalytics />}
          {activeTab === "prescriptive" && <PrescriptiveAnalytics />}
          {activeTab === "reports" && <ReportsPage />}
          {activeTab === "profile" && (
            <ProfilePage userEmail={userEmail} onBack={handleBackToDashboard} onLogout={onLogout} />
          )}
        </main>
      </div>
    </div>
  )
}
