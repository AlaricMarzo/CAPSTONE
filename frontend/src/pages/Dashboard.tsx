  import { useState, useEffect } from "react"
import { Sidebar } from "@/components/Layout/Sidebar"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { RecentAlertsCard } from "@/components/Dashboard/RecentAlertCard"
import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard"
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard"
import { TrendingUp, DollarSign, Package, ShoppingCart, User, Search, Calendar, FileText, Loader2, AlertCircle } from "lucide-react"
import { Input } from "@/components/ui/input"
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
    total_cost_savings: number
  }
}

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }) {
  const [prescriptiveData, setPrescriptiveData] = useState<PrescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)

  useEffect(() => {
    const fetchPrescriptiveData = async () => {
      try {
        setLoading(true)
        const response = await fetch("http://localhost:5050/api/analytics/prescriptive")
        if (!response.ok) throw new Error("Failed to fetch prescriptive data")
        const result = await response.json()
        setPrescriptiveData(result.data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred")
        console.error("[v0] Error fetching prescriptive data:", err)
      } finally {
        setLoading(false)
      }
    }

    fetchPrescriptiveData()
  }, [])

  const confirmLogout = () => {
    onLogout()
    setConfirmOpen(false)
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
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

  if (error || !prescriptiveData || !prescriptiveData.financial_summary || !prescriptiveData.reorder_points) {
    const fallbackMetrics = getUserMetrics()
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
            <p className="text-muted-foreground">Real-time business insights and analytics</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Last 30 Days</span>
            </div>
            <Button variant="outline" size="sm" className="shadow-soft bg-transparent">
              <FileText className="h-4 w-4 mr-2" />
              Export Report
            </Button>

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

        <div className="flex items-center gap-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input placeholder="Search products, customers, orders..." className="pl-10 shadow-soft" />
          </div>
        </div>

        <div className="text-destructive flex items-center gap-2 mb-4">
          Error loading prescriptive data: {error || 'Invalid data structure'}. Showing fallback data.
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Revenue"
            value={`$${fallbackMetrics.totalRevenue.toFixed(2)}`}
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
            value={`$${fallbackMetrics.averageOrderValue.toFixed(2)}`}
            change="+5.1% from last month"
            changeType="positive"
            icon={TrendingUp}
            color="default"
            className="shadow-soft hover:shadow-elegant transition-shadow"
          />
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <SalesTrendsChart />
          </div>
          <ProductTrafficCard />
        </div>

        <RecentAlertsCard />
      </div>
    )
  }

  const { financial_summary, reorder_points } = prescriptiveData
  const totalRevenue = financial_summary.total_sales
  const totalSales = financial_summary.total_quantity_sold
  const lowStockItems = reorder_points.length
  const avgOrderValue = totalSales > 0 ? totalRevenue / totalSales : 0

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time business insights and analytics</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Last 30 Days</span>
          </div>
          <Button variant="outline" size="sm" className="shadow-soft bg-transparent">
            <FileText className="h-4 w-4 mr-2" />
            Export Report
          </Button>

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

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search products, customers, orders..." className="pl-10 shadow-soft" />
        </div>
      </div>

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
          value={totalSales.toLocaleString()}
          change="units sold"
          changeType="positive"
          icon={ShoppingCart}
          color="info"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Items to Reorder"
          value={lowStockItems.toString()}
          change="need attention"
          changeType="warning"
          icon={Package}
          color="warning"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Avg Order Value"
          value={formatCurrency(avgOrderValue)}
          change="per unit"
          changeType="positive"
          icon={TrendingUp}
          color="default"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SalesTrendsChart />
        </div>
        <ProductTrafficCard />
      </div>

      <RecentAlertsCard />
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
              <BarChart data={data.top_products} layout="vertical">
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
