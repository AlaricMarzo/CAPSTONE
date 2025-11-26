"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { MetricCard } from "@/components/Dashboard/MetricCard"
import { TrendingUp, Target, BarChart3, Zap, Image as ImageIcon, ChevronLeft, ChevronRight } from "lucide-react"

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
  forecast_data: any[]
  feature_importance: any[]
  model_performance: any
  product_insights: any[]
  model_forecasts: any[]
  predictive_images: PredictiveImage[]
}

export default function PredictiveDashboard() {
  const [data, setData] = useState<PredictiveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentImageIndex, setCurrentImageIndex] = useState<{ [key: string]: number }>({
    gradient: 0,
    xgboost: 0,
    lstm: 0,
    random_forest: 0
  })

  const getImagesByModel = (model: string) => {
    return data?.predictive_images?.filter(image => image.model === model) || []
  }

  const navigateImage = (model: string, direction: 'prev' | 'next') => {
    const images = getImagesByModel(model)
    if (images.length === 0) return

    setCurrentImageIndex(prev => ({
      ...prev,
      [model]: direction === 'next'
        ? (prev[model] + 1) % images.length
        : (prev[model] - 1 + images.length) % images.length
    }))
  }

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
          value={data.models_summary.total_models.toString()}
          change="ML models"
          changeType="positive"
          icon={BarChart3}
          color="success"
        />
        <MetricCard
          title="Forecasts"
          value={data.models_summary.total_forecasts.toString()}
          change="generated"
          changeType="positive"
          icon={Target}
          color="info"
        />
        <MetricCard
          title="Accuracy"
          value={`${(data.models_summary.avg_accuracy * 100).toFixed(1)}%`}
          change="average"
          changeType="positive"
          icon={TrendingUp}
          color="warning"
        />
        <MetricCard
          title="Products Analyzed"
          value={data.models_summary.total_products_analyzed.toString()}
          change="analyzed"
          changeType="positive"
          icon={Zap}
          color="default"
        />
      </div>

      {/* Predictive Model Images */}
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <ImageIcon className="h-5 w-5" />
          <h2 className="text-2xl font-semibold">Generated Forecast Plots</h2>
        </div>

        {data.predictive_images && data.predictive_images.length > 0 ? (
          <Tabs defaultValue="gradient" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="gradient">Gradient</TabsTrigger>
              <TabsTrigger value="xgboost">XGBoost</TabsTrigger>
              <TabsTrigger value="lstm">LSTM</TabsTrigger>
              <TabsTrigger value="random_forest">Random Forest</TabsTrigger>
            </TabsList>

            {(['gradient', 'xgboost', 'lstm', 'random_forest'] as const).map((model) => {
              const images = getImagesByModel(model)
              const currentImage = images[currentImageIndex[model]]

              return (
                <TabsContent key={model} value={model} className="mt-6">
                  {images.length > 0 ? (
                    <Card className="overflow-hidden">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <CardTitle className="text-lg font-medium">{currentImage.name}</CardTitle>
                            <CardDescription className="capitalize">
                              Model: {currentImage.model} | Type: {currentImage.type.replace('_', ' ')}
                            </CardDescription>
                          </div>
                          {images.length > 1 && (
                            <div className="flex items-center gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigateImage(model, 'prev')}
                                disabled={images.length <= 1}
                              >
                                <ChevronLeft className="h-4 w-4" />
                              </Button>
                              <span className="text-sm text-muted-foreground">
                                {currentImageIndex[model] + 1} of {images.length}
                              </span>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigateImage(model, 'next')}
                                disabled={images.length <= 1}
                              >
                                <ChevronRight className="h-4 w-4" />
                              </Button>
                            </div>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="p-0">
                        <div className="aspect-[4/3] overflow-hidden bg-muted">
                          <img
                            src={currentImage.image}
                            alt={currentImage.name}
                            className="w-full h-full object-contain"
                            loading="lazy"
                          />
                        </div>
                      </CardContent>
                    </Card>
                  ) : (
                    <Card>
                      <CardContent className="flex flex-col items-center justify-center py-12">
                        <ImageIcon className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-muted-foreground text-center">
                          No {model.replace('_', ' ').toUpperCase()} images available.
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              )
            })}
          </Tabs>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <ImageIcon className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground text-center">
                No forecast images available. Run predictive analytics to generate plots.
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Model Performance Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Summary</CardTitle>
          <CardDescription>Performance metrics for each predictive model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {data.model_performance && Object.entries(data.model_performance).map(([model, metrics]: [string, any]) => (
              <div key={model} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="font-medium capitalize">{model.replace('_', ' ')}</div>
                <div className="flex gap-4 text-sm">
                  <div>MAE: {metrics.mae?.toFixed(3) || 'N/A'}</div>
                  <div>RMSE: {metrics.rmse?.toFixed(3) || 'N/A'}</div>
                  <div>R²: {metrics.r_squared?.toFixed(3) || 'N/A'}</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance</CardTitle>
          <CardDescription>Top predictive features used in the models</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {data.feature_importance && data.feature_importance.map((feature: any, index: number) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm font-medium">{feature.feature}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full"
                      style={{ width: `${(feature.importance || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-muted-foreground w-12">
                    {((feature.importance || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
