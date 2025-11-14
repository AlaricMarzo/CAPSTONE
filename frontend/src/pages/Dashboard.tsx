import { useState, useEffect } from "react"
import { Sidebar } from "@/components/Layout/Sidebar"
import { MetricCard } from "@/components/Dashboard/MetricCard"

import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard"
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard"
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
import { LineChart,
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
  }
  forecast_data: any[]
  accuracy_metrics: any[]
  feature_importance: any[]
  model_performance: any[]
}

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }) {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState<{
    descriptive: DescriptiveData | null
    predictive: PredictiveData | null
    prescriptive: PrescriptiveData | null
  }>({
    descriptive: null,
    predictive: null,
    prescriptive: null,
  })

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [descriptiveRes, predictiveRes, prescriptiveRes] = await Promise.all([
          fetch("/api/analytics/descriptive"),
          fetch("/api/analytics/predictive"),
          fetch("/api/analytics/prescriptive")
        ])

        if (!descriptiveRes.ok) throw new Error("Failed to fetch sales data")
        if (!predictiveRes.ok) throw new Error("Failed to fetch predictive data")
        if (!prescriptiveRes.ok) throw new Error("Failed to fetch prescriptive data")

        const [descriptiveData, predictiveData, prescriptiveData] = await Promise.all([
          descriptiveRes.json(),
          predictiveRes.json(),
          prescriptiveRes.json(),
        ])

        setData({
          descriptive: descriptiveData,
          predictive: predictiveData,
          prescriptive: prescriptiveData,
        })
      } catch (error) {
        console.error("Error fetching dashboard data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()

    // Listen for analytics updates
    const handleAnalyticsUpdate = () => {
      fetchData()
    }

    window.addEventListener('analyticsUpdated', handleAnalyticsUpdate)

    return () => {
      window.removeEventListener('analyticsUpdated', handleAnalyticsUpdate)
    }
  }, [])

  if (loading) {
    return (
      <div className="flex-1 p-8 pt-6 bg-background">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="ml-2 text-muted-foreground">Loading dashboard...</span>
        </div>
      </div>
    )
  }

  if (!data.descriptive) {
    return (
      <div className="flex-1 p-8 pt-6 bg-background">
        <div className="flex items-center justify-center h-64">
          <AlertCircle className="h-8 w-8 text-destructive" />
          <span className="ml-2 text-muted-foreground">No data available. Please upload sales data first.</span>
        </div>
      </div>
    )
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'PHP',
    }).format(value)
  }

  return (
    <div className="flex-1 p-8 pt-6 bg-background">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
        <p className="text-muted-foreground">Overview of your sales performance and analytics</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Total Sales"
          value={formatCurrency(data.descriptive?.kpi_summary?.total_sales ?? 0)}
          icon={DollarSign}
          trend="+12.5%"
        />
        <MetricCard
          title="Total Quantity"
          value={(data.descriptive?.kpi_summary?.total_quantity ?? 0).toLocaleString()}
          icon={Package}
          trend="+8.2%"
        />
        <MetricCard
          title="Active SKUs"
          value={(data.descriptive?.kpi_summary?.active_skus ?? 0).toString()}
          icon={ShoppingCart}
          trend="+5.1%"
        />
        <MetricCard
          title="Growth Rate"
          value={`${(data.descriptive?.kpi_summary?.growth_rate ?? 0).toFixed(1)}%`}
          icon={TrendingUp}
          trend="+2.3%"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Monthly Sales Trend</CardTitle>
            <CardDescription>Sales performance over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.descriptive?.monthly_sales ?? []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value) => formatCurrency(value as number)} />
                <Line type="monotone" dataKey="sales" stroke="#8884d8" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Category Distribution</CardTitle>
            <CardDescription>Sales by product category</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.descriptive?.category_distribution ?? []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Monthly Sales Table */}
      <Card>
        <CardHeader>
          <CardTitle>Monthly Sales Summary</CardTitle>
          <CardDescription>Detailed breakdown of monthly performance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Month</th>
                  <th className="text-right p-2">Sales</th>
                  <th className="text-right p-2">Quantity</th>
                  <th className="text-right p-2">Avg Price</th>
                </tr>
              </thead>
              <tbody>
                {data.descriptive?.monthly_sales ?? [].map((item: any, idx: number) => (
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

function SalesPage() {
  return <HomePage onProfileClick={() => {}} onLogout={() => {}} />
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
