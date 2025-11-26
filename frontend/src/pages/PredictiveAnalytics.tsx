"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { TrendingUp, Target, Zap, AlertCircle, Loader2, BarChart3, PieChart, Image as ImageIcon, Brain, Activity, Eye, Layers, ChevronLeft, ChevronRight } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"

interface ModelMetrics {
  mae: number
  rmse: number
  r_squared: number
}

interface PredictiveImage {
  name: string
  image: string
  model: string
  type: string
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
  feature_importance?: Array<{ feature: string; importance: number }>
  model_performance?: {
    gradient?: ModelMetrics
    xgboost?: ModelMetrics
    lstm?: ModelMetrics
    random_forest?: ModelMetrics
    sarima?: ModelMetrics
  }
  product_insights?: Array<{
    product_name: string
    total_quantity: number
    total_sales: number
    total_profit: number
    avg_unit_price: number
    profit_margin_pct: number
    forecasted_demand: number
  }>
  model_forecasts?: Array<{
    model: string
    sku: string
    name: string
    data: Array<{
      date: string
      forecast: number
    }>
  }>
  predictive_images?: PredictiveImage[]
}

export default function PredictiveAnalytics() {
  const [data, setData] = useState<PredictiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>("all")
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0)

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

  // Reset image index when model filter changes
  useEffect(() => {
    setCurrentImageIndex(0)
  }, [selectedModel])

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

  // Group images by model
  const groupedImages = data.predictive_images?.reduce((acc, image) => {
    if (!acc[image.model]) acc[image.model] = []
    acc[image.model].push(image)
    return acc
  }, {} as Record<string, PredictiveImage[]>) || {}

  const getModelColor = (model: string) => {
    switch (model) {
      case 'gradient': return 'bg-red-100 text-red-800 border-red-200'
      case 'xgboost': return 'bg-green-100 text-green-800 border-green-200'
      case 'lstm': return 'bg-purple-100 text-purple-800 border-purple-200'
      case 'random_forest': return 'bg-blue-100 text-blue-800 border-blue-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getModelIcon = (model: string) => {
    switch (model) {
      case 'gradient': return <TrendingUp className="h-4 w-4" />
      case 'xgboost': return <Brain className="h-4 w-4" />
      case 'lstm': return <Activity className="h-4 w-4" />
      case 'random_forest': return <Layers className="h-4 w-4" />
      default: return <BarChart3 className="h-4 w-4" />
    }
  }

  const getFilteredImages = () => {
    if (!data?.predictive_images) return []
    return selectedModel === "all"
      ? data.predictive_images
      : data.predictive_images.filter(image => image.model === selectedModel)
  }

  const navigateImage = (direction: 'prev' | 'next') => {
    const filteredImages = getFilteredImages()
    if (filteredImages.length === 0) return

    setCurrentImageIndex(prev => {
      if (direction === 'next') {
        return (prev + 1) % filteredImages.length
      } else {
        return (prev - 1 + filteredImages.length) % filteredImages.length
      }
    })
  }

  return (
    <div className="flex-1 space-y-8 p-8 pt-6">
      {/* Header Section */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <TrendingUp className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Predictive Analytics Dashboard</h1>
            <p className="text-muted-foreground">Advanced ML forecasting and predictive insights</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Eye className="h-4 w-4" />
            Last updated: {new Date().toLocaleDateString()}
          </div>
          <Badge variant="outline" className="text-xs">
            {data.models_summary.total_models} Models Active
          </Badge>
        </div>
      </div>

      {/* KPI Overview Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Active Models"
          value={data.models_summary.total_models.toString()}
          change="ML ensemble"
          changeType="positive"
          icon={Zap}
          color="success"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Model Accuracy"
          value={`${(data.models_summary.avg_accuracy * 100).toFixed(1)}%`}
          change="R² average"
          changeType="positive"
          icon={Target}
          color="info"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Total Forecasts"
          value={data.models_summary.total_forecasts.toString()}
          change="predictions"
          changeType="positive"
          icon={TrendingUp}
          color="warning"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Products Analyzed"
          value={data.models_summary.total_products_analyzed.toString()}
          change="data points"
          changeType="positive"
          icon={AlertCircle}
          color="default"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
      </div>

      {/* Main Dashboard Content */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:grid-cols-3">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="forecasts" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Forecasts
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Insights
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Model Performance Summary */}
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Model Performance Overview
              </CardTitle>
              <CardDescription>Comparative analysis of all predictive models</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-4">
                {[
                  { name: "Gradient", model: data.model_performance?.gradient, color: "#dc2626" },
                  { name: "XGBoost", model: data.model_performance?.xgboost, color: "#10b981" },
                  { name: "LSTM", model: data.model_performance?.lstm, color: "#8b5cf6" },
                  { name: "Random Forest", model: data.model_performance?.random_forest, color: "#3b82f6" },
                ].map((m) => (
                  <div key={m.name} className="space-y-4 p-4 border rounded-lg bg-card/50">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: m.color }} />
                      <h4 className="font-semibold">{m.name}</h4>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">MAE:</span>
                        <span className="font-medium">{m.model?.mae?.toFixed(3) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">RMSE:</span>
                        <span className="font-medium">{m.model?.rmse?.toFixed(3) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">R² Score:</span>
                        <span className="font-medium">{m.model?.r_squared?.toFixed(3) || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Feature Importance */}
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Feature Importance Analysis
              </CardTitle>
              <CardDescription>Key variables driving predictive performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {data.feature_importance && data.feature_importance.slice(0, 8).map((feature: any, index: number) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{feature.feature}</span>
                      <span className="text-sm text-muted-foreground">
                        {((feature.importance || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-500"
                        style={{ width: `${(feature.importance || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>



        {/* Forecasts Tab */}
        <TabsContent value="forecasts" className="space-y-6">
          {data.predictive_images && data.predictive_images.length > 0 ? (
            <div className="space-y-6">
              {/* Model Filter */}
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium">Filter by Model:</span>
                <div className="flex gap-2">
                  <Badge
                    variant={selectedModel === "all" ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => setSelectedModel("all")}
                  >
                    All Models
                  </Badge>
                  {Object.keys(groupedImages).map(model => (
                    <Badge
                      key={model}
                      variant={selectedModel === model ? "default" : "outline"}
                      className={`cursor-pointer ${getModelColor(model)}`}
                      onClick={() => setSelectedModel(model)}
                    >
                      {model.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Image Carousel */}
              {(() => {
                const filteredImages = getFilteredImages()
                const currentImage = filteredImages[currentImageIndex]

                if (!currentImage) {
                  return (
                    <Card className="shadow-soft">
                      <CardContent className="flex flex-col items-center justify-center py-16">
                        <ImageIcon className="h-16 w-16 text-muted-foreground mb-4" />
                        <h3 className="text-lg font-semibold mb-2">No Images Available</h3>
                        <p className="text-muted-foreground text-center max-w-md">
                          No images match the selected filter criteria.
                        </p>
                      </CardContent>
                    </Card>
                  )
                }

                return (
                  <Card className="overflow-hidden shadow-soft">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div>
                            <CardTitle className="text-lg font-medium">{currentImage.name}</CardTitle>
                            <CardDescription className="capitalize">
                              Model: {currentImage.model} | Type: {currentImage.type.replace('_', ' ')}
                            </CardDescription>
                          </div>
                        </div>
                        {filteredImages.length > 1 && (
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => navigateImage('prev')}
                              className="p-2 rounded-md hover:bg-muted transition-colors"
                              disabled={filteredImages.length <= 1}
                            >
                              <ChevronLeft className="h-5 w-5" />
                            </button>
                            <span className="text-sm text-muted-foreground min-w-[60px] text-center">
                              {currentImageIndex + 1} of {filteredImages.length}
                            </span>
                            <button
                              onClick={() => navigateImage('next')}
                              className="p-2 rounded-md hover:bg-muted transition-colors"
                              disabled={filteredImages.length <= 1}
                            >
                              <ChevronRight className="h-5 w-5" />
                            </button>
                          </div>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent className="p-0">
                      <div className="aspect-[4/3] bg-muted/50">
                        <img
                          src={currentImage.image}
                          alt={currentImage.name}
                          className="w-full h-full object-contain p-4"
                          loading="lazy"
                        />
                      </div>
                    </CardContent>
                  </Card>
                )
              })()}
            </div>
          ) : (
            <Card className="shadow-soft">
              <CardContent className="flex flex-col items-center justify-center py-16">
                <ImageIcon className="h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Forecast Images Available</h3>
                <p className="text-muted-foreground text-center max-w-md">
                  Run predictive analytics to generate forecast plots and visualizations.
                  Images will appear here once the analysis is complete.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Product Performance Insights
              </CardTitle>
              <CardDescription>Top performing products with predictive demand analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {data.product_insights && data.product_insights.slice(0, 8).map((product, index) => (
                  <div key={index} className="border rounded-lg p-4 bg-card/50 hover:bg-card transition-colors">
                    <div key={index} className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold text-base truncate pr-2">{product.product_name}</h4>
                      <Badge variant={product.profit_margin_pct >= 0 ? "default" : "destructive"}>
                        {product.profit_margin_pct >= 0 ? '+' : ''}{product.profit_margin_pct.toFixed(1)}% margin
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className="space-y-1">
                        <div className="text-muted-foreground text-xs">Sales</div>
                        <div className="font-semibold">₱{product.total_sales.toLocaleString()}</div>
                      </div>
                      <div className="space-y-1">
                        <div className="text-muted-foreground text-xs">Quantity</div>
                        <div className="font-semibold">{product.total_quantity.toLocaleString()}</div>
                      </div>
                      <div className="space-y-1">
                        <div className="text-muted-foreground text-xs">Avg Price</div>
                        <div className="font-semibold">₱{product.avg_unit_price.toFixed(2)}</div>
                      </div>
                      <div className="space-y-1">
                        <div className="text-muted-foreground text-xs">Forecast</div>
                        <div className="font-semibold">₱{product.forecasted_demand.toLocaleString()}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
