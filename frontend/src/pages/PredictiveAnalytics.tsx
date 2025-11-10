"use client"

import { useState, useEffect } from "react"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { TrendingUp, Target, Zap, AlertCircle, Loader2 } from "lucide-react"

interface ModelMetrics {
  mae: number
  rmse: number
  r_squared: number
}

interface PredictiveData {
  models_summary: {
    total_models: number
    avg_accuracy: number
    total_forecasts: number
  }
  forecast_data: Array<{
    date: string
    actual: number
    rf_predicted: number
    xgb_predicted: number
    sarima_predicted: number
    confidence_lower: number
    confidence_upper: number
  }>
  feature_importance: Array<{ feature: string; importance: number }>
  model_performance: {
    random_forest: ModelMetrics
    xgboost: ModelMetrics
    sarima: ModelMetrics
  }
}

export default function PredictiveAnalytics() {
  const [data, setData] = useState<PredictiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch("http://localhost:5050/api/analytics/predictive")
        if (!response.ok) throw new Error("Failed to fetch predictive analytics")
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
          Loading predictive analytics...
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

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Predictive Analytics</h1>
        <p className="text-muted-foreground">ML forecasts and model performance metrics</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Active Models"
          value={data.models_summary.total_models.toString()}
          change="ensemble"
          changeType="positive"
          icon={Zap}
          color="success"
          className="shadow-soft"
        />
        <MetricCard
          title="Avg Accuracy (R²)"
          value={`${(data.models_summary.avg_accuracy * 100).toFixed(1)}%`}
          change="across models"
          changeType="positive"
          icon={Target}
          color="info"
          className="shadow-soft"
        />
        <MetricCard
          title="Total Forecasts"
          value={data.models_summary.total_forecasts.toString()}
          change="generated"
          changeType="positive"
          icon={TrendingUp}
          color="warning"
          className="shadow-soft"
        />
        <MetricCard
          title="Confidence"
          value="95%"
          change="intervals"
          changeType="positive"
          icon={AlertCircle}
          color="success"
          className="shadow-soft"
        />
      </div>

      {/* Combined Forecasts */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Multi-Model Sales Forecast</CardTitle>
          <CardDescription>Actual vs predicted with confidence intervals</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.forecast_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#000" name="Actual" strokeWidth={2} />
                <Line
                  type="monotone"
                  dataKey="rf_predicted"
                  stroke="#3b82f6"
                  name="Random Forest"
                  strokeDasharray="5 5"
                />
                <Line type="monotone" dataKey="xgb_predicted" stroke="#10b981" name="XGBoost" strokeDasharray="5 5" />
                <Line type="monotone" dataKey="sarima_predicted" stroke="#f59e0b" name="SARIMA" strokeDasharray="5 5" />
                <Line
                  type="monotone"
                  dataKey="confidence_lower"
                  stroke="#d1d5db"
                  name="Lower Bound"
                  strokeDasharray="3 3"
                />
                <Line
                  type="monotone"
                  dataKey="confidence_upper"
                  stroke="#d1d5db"
                  name="Upper Bound"
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Feature Importance Rankings</CardTitle>
          <CardDescription>Top variables influencing predictions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.feature_importance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={150} />
                <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Model Performance Comparison */}
      <div className="grid gap-6 lg:grid-cols-3">
        {[
          { name: "Random Forest", model: data.model_performance.random_forest, color: "#3b82f6" },
          { name: "XGBoost", model: data.model_performance.xgboost, color: "#10b981" },
          { name: "SARIMA", model: data.model_performance.sarima, color: "#f59e0b" },
        ].map((m) => (
          <Card key={m.name} className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: m.color }} />
                {m.name}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">MAE</div>
                <div className="text-2xl font-bold">{m.model.mae.toFixed(2)}</div>
              </div>
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">RMSE</div>
                <div className="text-2xl font-bold">{m.model.rmse.toFixed(2)}</div>
              </div>
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">R² Score</div>
                <div className="text-2xl font-bold">{m.model.r_squared.toFixed(4)}</div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
