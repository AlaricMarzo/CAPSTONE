"use client"

import { useState, useEffect } from "react"
import {
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
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
import { TrendingUp, Target, BarChart3, Zap } from "lucide-react"

interface PredictiveKPI {
  totalModels: number
  totalForecasts: number
  avgAccuracy: number
  bestModel: string
}

interface PredictiveData {
  kpi: PredictiveKPI
  combinedForecasts: any[]
  featureImportance: any[]
  modelComparison: any[]
  confidenceIntervals: any[]
}

export default function PredictiveDashboard() {
  const [data, setData] = useState<PredictiveData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/analytics/predictive")
        const result = await response.json()
        setData(result.data)
      } catch (error) {
        console.error("Failed to fetch predictive data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return <div className="p-8">Loading predictive analytics...</div>
  }

  if (!data) {
    return <div className="p-8 text-destructive">Failed to load predictive analytics</div>
  }

  return (
    <div className="space-y-6 p-8">
      <div>
        <h1 className="text-3xl font-bold">Predictive Analytics</h1>
        <p className="text-muted-foreground">Forecasting, ML models, and predictive insights</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          title="Active Models"
          value={data.kpi.totalModels.toString()}
          change="ML models"
          changeType="positive"
          icon={BarChart3}
          color="success"
        />
        <MetricCard
          title="Forecasts"
          value={data.kpi.totalForecasts.toString()}
          change="generated"
          changeType="positive"
          icon={Target}
          color="info"
        />
        <MetricCard
          title="Accuracy"
          value={`${data.kpi.avgAccuracy.toFixed(1)}%`}
          change="average"
          changeType="positive"
          icon={TrendingUp}
          color="warning"
        />
        <MetricCard
          title="Best Model"
          value={data.kpi.bestModel}
          change="highest performing"
          changeType="positive"
          icon={Zap}
          color="default"
        />
      </div>

      {/* Combined Forecast Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Combined Model Forecasts</CardTitle>
          <CardDescription>Actual vs predicted values with confidence intervals</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.combinedForecasts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#3b82f6" name="Actual" isAnimationActive={true} />
                <Line type="monotone" dataKey="predicted" stroke="#10b981" name="Predicted" strokeDasharray="5 5" />
                <Line
                  type="monotone"
                  dataKey="upper"
                  stroke="#ef4444"
                  name="Upper Bound"
                  strokeDasharray="3 3"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="lower"
                  stroke="#f59e0b"
                  name="Lower Bound"
                  strokeDasharray="3 3"
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance & Model Comparison */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Feature Importance</CardTitle>
            <CardDescription>Top predictive features</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#3b82f6" name="Importance Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Comparison</CardTitle>
            <CardDescription>Performance metrics across models</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.modelComparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mae" fill="#3b82f6" name="MAE" />
                  <Bar dataKey="rmse" fill="#10b981" name="RMSE" />
                  <Bar dataKey="r2" fill="#f59e0b" name="R² Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Confidence Intervals */}
      <Card>
        <CardHeader>
          <CardTitle>Forecast Confidence Analysis</CardTitle>
          <CardDescription>Prediction uncertainty by time period</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data.confidenceIntervals}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Scatter name="Forecast Range" dataKey="confidence" fill="#8b5cf6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
