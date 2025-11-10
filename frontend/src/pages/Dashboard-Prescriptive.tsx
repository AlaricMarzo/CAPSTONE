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
import { Zap, TrendingUp, DollarSign, Target } from "lucide-react"

interface PrescriptiveKPI {
  reorderPointsCount: number
  eoqCalculations: number
  discountOptimizations: number
  avgCostSavings: number
}

interface PrescriptiveData {
  kpi: PrescriptiveKPI
  reorderPoints: any[]
  eoqData: any[]
  whatIfAnalysis: any[]
  discountOptimization: any[]
  inventoryAllocation: any[]
}

export default function PrescriptiveDashboard() {
  const [data, setData] = useState<PrescriptiveData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/analytics/prescriptive")
        const result = await response.json()
        setData(result.data)
      } catch (error) {
        console.error("Failed to fetch prescriptive data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return <div className="p-8">Loading prescriptive analytics...</div>
  }

  if (!data) {
    return <div className="p-8 text-destructive">Failed to load prescriptive analytics</div>
  }

  return (
    <div className="space-y-6 p-8">
      <div>
        <h1 className="text-3xl font-bold">Prescriptive Analytics</h1>
        <p className="text-muted-foreground">Optimization recommendations and decision support</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          title="Reorder Points"
          value={data.kpi.reorderPointsCount.toString()}
          change="calculated"
          changeType="positive"
          icon={Target}
          color="success"
        />
        <MetricCard
          title="EOQ Models"
          value={data.kpi.eoqCalculations.toString()}
          change="optimized"
          changeType="positive"
          icon={Zap}
          color="info"
        />
        <MetricCard
          title="Discount Options"
          value={data.kpi.discountOptimizations.toString()}
          change="scenarios"
          changeType="positive"
          icon={DollarSign}
          color="warning"
        />
        <MetricCard
          title="Potential Savings"
          value={`$${data.kpi.avgCostSavings.toLocaleString()}`}
          change="identified"
          changeType="positive"
          icon={TrendingUp}
          color="default"
        />
      </div>

      {/* Reorder Point Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Reorder Point Analysis</CardTitle>
          <CardDescription>Safety stock and optimal reorder levels by product</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.reorderPoints}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="product" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="reorderPoint" fill="#3b82f6" name="Reorder Point" />
                <Bar dataKey="safetyStock" fill="#10b981" name="Safety Stock" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* EOQ & Inventory Allocation */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Economic Order Quantity</CardTitle>
            <CardDescription>Optimal order quantities for cost minimization</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.eoqData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="product" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="eoq" fill="#f59e0b" name="EOQ" />
                  <Bar dataKey="currentOrder" fill="#ef4444" name="Current Order" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Inventory Allocation</CardTitle>
            <CardDescription>Recommended stock distribution across locations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.inventoryAllocation}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="location" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="allocated" fill="#8b5cf6" name="Recommended" />
                  <Bar dataKey="current" fill="#ec4899" name="Current" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* What-If Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>What-If Analysis</CardTitle>
          <CardDescription>Impact of different scenarios on profit and costs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.whatIfAnalysis}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="profit" stroke="#10b981" name="Projected Profit" />
                <Line yAxisId="right" type="monotone" dataKey="cost" stroke="#ef4444" name="Projected Cost" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Discount Optimization */}
      <Card>
        <CardHeader>
          <CardTitle>Discount Optimization</CardTitle>
          <CardDescription>Profit margin impact across discount levels</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data.discountOptimization}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="discountPct" name="Discount %" />
                <YAxis dataKey="profitMargin" name="Profit Margin %" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Products" data={data.discountOptimization} fill="#3b82f6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
