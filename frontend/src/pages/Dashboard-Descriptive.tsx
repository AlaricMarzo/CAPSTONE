"use client"

import { useState, useEffect } from "react"
import {
  LineChart,
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
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { TrendingUp, Package, Activity, Zap } from "lucide-react"

interface KPIData {
  totalSales: number
  totalQuantity: number
  activeSkus: number
  growthRate: number
  clusterCount: number
  mbaRuleCount: number
}

interface ClusteringImage {
  name: string
  image: string
  summary?: any[]
}

interface DescriptiveData {
  kpi: KPIData
  monthlySales: any[]
  categoryDistribution: any[]
  clusteringSummary: any[]
  mbaTopRules: any[]
  clustering_images?: ClusteringImage[]
}

export default function DescriptiveDashboard() {
  const [data, setData] = useState<DescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/analytics/descriptive")
        const result = await response.json()
        setData(result.data)
      } catch (error) {
        console.error("Failed to fetch descriptive data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

  if (loading) {
    return <div className="p-8">Loading descriptive analytics...</div>
  }

  if (!data) {
    return <div className="p-8 text-destructive">Failed to load descriptive analytics</div>
  }

  return (
    <div className="space-y-6 p-8">
      <div>
        <h1 className="text-3xl font-bold">Descriptive Analytics</h1>
        <p className="text-muted-foreground">Overview of historical data, patterns, and trends</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          title="Total Sales"
          value={`$${data.kpi.totalSales.toLocaleString()}`}
          change={`+${data.kpi.growthRate.toFixed(1)}% MoM`}
          changeType="positive"
          icon={TrendingUp}
          color="success"
        />
        <MetricCard
          title="Total Quantity"
          value={data.kpi.totalQuantity.toLocaleString()}
          change="units sold"
          changeType="positive"
          icon={Package}
          color="info"
        />
        <MetricCard
          title="Active SKUs"
          value={data.kpi.activeSkus.toString()}
          change="products"
          changeType="positive"
          icon={Activity}
          color="warning"
        />
        <MetricCard
          title="Clusters"
          value={data.kpi.clusterCount.toString()}
          change={`${data.kpi.mbaRuleCount} association rules`}
          changeType="positive"
          icon={Zap}
          color="default"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Monthly Sales Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Monthly Sales Trend</CardTitle>
            <CardDescription>Sales and quantity over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.monthlySales}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="sales" stroke="#3b82f6" name="Sales ($)" />
                  <Line type="monotone" dataKey="quantity" stroke="#10b981" name="Quantity" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Category Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Category Distribution</CardTitle>
            <CardDescription>Sales by category</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.categoryDistribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {data.categoryDistribution.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Clustering Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Customer Clustering Analysis</CardTitle>
          <CardDescription>Behavioral segments and groupings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.clusteringSummary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cluster" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#3b82f6" name="Customer Count" />
                <Bar dataKey="avgValue" fill="#10b981" name="Avg Order Value" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Market Basket Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Top Product Associations</CardTitle>
          <CardDescription>Frequently bought together products (top 10)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.mbaTopRules} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={150} />
                <Tooltip />
                <Legend />
                <Bar dataKey="lift" fill="#f59e0b" name="Lift" />
                <Bar dataKey="confidence" fill="#ef4444" name="Confidence" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
