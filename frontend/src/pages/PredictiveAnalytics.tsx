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
import { TrendingUp, Target, Zap, AlertCircle, Loader2, BarChart3, PieChart } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

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
    total_products_analyzed: number
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
  product_insights: Array<{
    product_name: string
    total_quantity: number
    total_sales: number
    total_profit: number
    avg_unit_price: number
    profit_margin_pct: number
    forecasted_demand: number
  }>
  model_forecasts: Array<{
    model: string
    sku: string
    name: string
    data: Array<{
      date: string
      forecast: number
    }>
  }>
}

export default function PredictiveAnalytics() {
  const [data, setData] = useState<PredictiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>("all")
  const [selectedProduct, setSelectedProduct] = useState<string>("all")

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const response = await fetch(`${import.meta.env.VITE_API_URL}/analytics/predictive`)
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

      {/* Navigation Tabs for Model Forecasts */}
      {data.model_forecasts && data.model_forecasts.length > 0 && (
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Model Forecast Visualizations</CardTitle>
            <CardDescription>Interactive forecast charts for individual models and products</CardDescription>
            <div className="flex gap-4 mt-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Model:</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Models</SelectItem>
                    <SelectItem value="xgboost">XGBoost</SelectItem>
                    <SelectItem value="random_forest">Random Forest</SelectItem>
                    <SelectItem value="sarima">SARIMA</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Product:</label>
                <Select value={selectedProduct} onValueChange={setSelectedProduct}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Products</SelectItem>
                    {[...new Set(data.model_forecasts.map(f => f.sku))].map(sku => (
                      <SelectItem key={sku} value={sku}>{sku}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="xgboost" className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  XGBoost
                </TabsTrigger>
                <TabsTrigger value="random_forest" className="flex items-center gap-2">
                  <PieChart className="h-4 w-4" />
                  Random Forest
                </TabsTrigger>
                <TabsTrigger value="sarima" className="flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  SARIMA
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="mt-6">
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {data.model_forecasts
                    .filter(forecast =>
                      (selectedModel === "all" || forecast.model === selectedModel) &&
                      (selectedProduct === "all" || forecast.sku === selectedProduct)
                    )
                    .slice(0, 6)
                    .map((forecast, index) => (
                      <div key={index} className="border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
                        <div className="mb-3">
                          <h4 className="font-semibold text-sm truncate">{forecast.name}</h4>
                          <p className="text-xs text-muted-foreground">Model: {forecast.model} | SKU: {forecast.sku}</p>
                        </div>
                        <div className="aspect-video bg-muted rounded overflow-hidden">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={forecast.data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", fontSize: "12px" }}
                                labelStyle={{ color: "var(--foreground)" }}
                              />
                              <Line
                                type="monotone"
                                dataKey="forecast"
                                stroke={forecast.model === 'xgboost' ? '#10b981' : forecast.model === 'random_forest' ? '#3b82f6' : '#f59e0b'}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4, stroke: forecast.model === 'xgboost' ? '#10b981' : forecast.model === 'random_forest' ? '#3b82f6' : '#f59e0b', strokeWidth: 2 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    ))}
                </div>
              </TabsContent>

              <TabsContent value="xgboost" className="mt-6">
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {data.model_forecasts
                    .filter(forecast => forecast.model === 'xgboost' && (selectedProduct === "all" || forecast.sku === selectedProduct))
                    .map((forecast, index) => (
                      <div key={index} className="border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
                        <div className="mb-3">
                          <h4 className="font-semibold text-sm truncate">{forecast.name}</h4>
                          <p className="text-xs text-muted-foreground">XGBoost | SKU: {forecast.sku}</p>
                        </div>
                        <div className="aspect-video bg-muted rounded overflow-hidden">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={forecast.data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", fontSize: "12px" }}
                                labelStyle={{ color: "var(--foreground)" }}
                              />
                              <Line
                                type="monotone"
                                dataKey="forecast"
                                stroke="#10b981"
                                strokeWidth={3}
                                dot={false}
                                activeDot={{ r: 5, stroke: '#10b981', strokeWidth: 2 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    ))}
                </div>
              </TabsContent>

              <TabsContent value="random_forest" className="mt-6">
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {data.model_forecasts
                    .filter(forecast => forecast.model === 'random_forest' && (selectedProduct === "all" || forecast.sku === selectedProduct))
                    .map((forecast, index) => (
                      <div key={index} className="border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
                        <div className="mb-3">
                          <h4 className="font-semibold text-sm truncate">{forecast.name}</h4>
                          <p className="text-xs text-muted-foreground">Random Forest | SKU: {forecast.sku}</p>
                        </div>
                        <div className="aspect-video bg-muted rounded overflow-hidden">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={forecast.data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", fontSize: "12px" }}
                                labelStyle={{ color: "var(--foreground)" }}
                              />
                              <Line
                                type="monotone"
                                dataKey="forecast"
                                stroke="#3b82f6"
                                strokeWidth={3}
                                dot={false}
                                activeDot={{ r: 5, stroke: '#3b82f6', strokeWidth: 2 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    ))}
                </div>
              </TabsContent>

              <TabsContent value="sarima" className="mt-6">
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {data.model_forecasts
                    .filter(forecast => forecast.model === 'sarima' && (selectedProduct === "all" || forecast.sku === selectedProduct))
                    .map((forecast, index) => (
                      <div key={index} className="border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
                        <div className="mb-3">
                          <h4 className="font-semibold text-sm truncate">{forecast.name}</h4>
                          <p className="text-xs text-muted-foreground">SARIMA | SKU: {forecast.sku}</p>
                        </div>
                        <div className="aspect-video bg-muted rounded overflow-hidden">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={forecast.data}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)", fontSize: "12px" }}
                                labelStyle={{ color: "var(--foreground)" }}
                              />
                              <Line
                                type="monotone"
                                dataKey="forecast"
                                stroke="#f59e0b"
                                strokeWidth={3}
                                dot={false}
                                activeDot={{ r: 5, stroke: '#f59e0b', strokeWidth: 2 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    ))}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}

      {/* Product Insights */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle>Top Product Insights</CardTitle>
          <CardDescription>Product-level performance with predictive demand forecasts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {data.product_insights.slice(0, 10).map((product, index) => (
              <div key={index} className="border rounded-lg p-4 bg-card">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-lg">{product.product_name}</h4>
                  <div className="text-right">
                    <div className="text-sm text-muted-foreground">Profit Margin</div>
                    <div className={`text-lg font-bold ${product.profit_margin_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {product.profit_margin_pct.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Total Sales</div>
                    <div className="font-semibold">₱{product.total_sales.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Quantity Sold</div>
                    <div className="font-semibold">{product.total_quantity.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Avg Unit Price</div>
                    <div className="font-semibold">₱{product.avg_unit_price.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Forecasted Demand</div>
                    <div className="font-semibold">₱{product.forecasted_demand.toLocaleString()}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
