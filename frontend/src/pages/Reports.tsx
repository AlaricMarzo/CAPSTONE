import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, FileText, Image, Loader2, AlertCircle, BarChart3, TrendingUp, Zap } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

interface FileInfo {
  name: string
  path: string
  type: 'csv' | 'png'
  category: string
}

interface FilesData {
  descriptive?: FileInfo[]
  prescriptive?: FileInfo[]
  predictive?: FileInfo[]
}

export default function ReportsPage() {
  const [files, setFiles] = useState<FilesData>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [downloading, setDownloading] = useState<string | null>(null)

  useEffect(() => {
    fetchFiles()
  }, [])

  const fetchFiles = async () => {
    try {
      setLoading(true)
      const response = await fetch("/api/analytics/files")
      if (!response.ok) throw new Error("Failed to fetch files")
      const result = await response.json()
      setFiles(result.files || {})
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
      console.error("[v0] Error fetching files:", err)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (category: string, filename: string) => {
    try {
      setDownloading(`${category}-${filename}`)
      const response = await fetch(`/api/analytics/download/${category}/${filename}`)
      if (!response.ok) throw new Error("Failed to download file")

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error("[v0] Error downloading file:", err)
      alert(`Failed to download ${filename}: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setDownloading(null)
    }
  }

  const renderFileSection = (title: string, category: keyof FilesData, filesList: FileInfo[], icon: React.ReactNode, description: string) => {
    const csvFiles = filesList.filter(f => f.type === 'csv')
    const pngFiles = filesList.filter(f => f.type === 'png')

    return (
      <AccordionItem value={category} className="border border-border/50 rounded-lg bg-gradient-to-br from-card to-card/50 shadow-lg hover:shadow-xl transition-all duration-300">
        <AccordionTrigger className="px-6 py-4 hover:no-underline">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-primary/10 text-primary">
              {icon}
            </div>
            <div className="text-left">
              <h3 className="text-lg font-semibold text-foreground">{title}</h3>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
          </div>
        </AccordionTrigger>
        <AccordionContent className="px-6 pb-4">
          <div className="space-y-6">
            {/* PNG Files - Display as images */}
            {pngFiles.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-4 flex items-center gap-2">
                  <Image className="h-4 w-4" />
                  Generated Charts & Visualizations ({pngFiles.length})
                </h4>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {pngFiles.map((file) => (
                    <div key={file.name} className="border border-border/50 rounded-lg p-4 bg-background/50 hover:bg-background/80 transition-colors duration-200">
                      <div className="aspect-video bg-muted/30 rounded flex items-center justify-center mb-3 overflow-hidden">
                        <img
                          src={`/api/analytics/download/${category}/${file.name}`}
                          alt={file.name}
                          className="max-w-full max-h-full object-contain rounded transition-transform duration-200 hover:scale-105"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none'
                            e.currentTarget.nextElementSibling!.classList.remove('hidden')
                          }}
                        />
                        <div className="text-muted-foreground text-sm hidden">
                          Preview not available
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate" title={file.name}>
                          {file.name}
                        </span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleDownload(category, file.name)}
                          disabled={downloading === `${category}-${file.name}`}
                          className="ml-2 hover:bg-primary hover:text-primary-foreground transition-colors duration-200"
                        >
                          {downloading === `${category}-${file.name}` ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Download className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* CSV Files */}
            {csvFiles.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-4 flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Data Files ({csvFiles.length})
                </h4>
                <div className="space-y-3">
                  {csvFiles.map((file) => (
                    <div key={file.name} className="flex items-center justify-between p-4 border border-border/50 rounded-lg bg-background/50 hover:bg-background/80 transition-colors duration-200">
                      <div className="flex items-center gap-3">
                        <FileText className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <span className="text-sm font-medium block">{file.name}</span>
                          <span className="text-xs text-muted-foreground">CSV File</span>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownload(category, file.name)}
                        disabled={downloading === `${category}-${file.name}`}
                        className="hover:bg-primary hover:text-primary-foreground transition-colors duration-200"
                      >
                        {downloading === `${category}-${file.name}` ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <>
                            <Download className="h-4 w-4 mr-2" />
                            Download
                          </>
                        )}
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {csvFiles.length === 0 && pngFiles.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No files available for {title.toLowerCase()} analytics</p>
              </div>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    )
  }

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading available reports...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Error loading reports: {error}
          </AlertDescription>
        </Alert>
        <Button onClick={fetchFiles} variant="outline">
          Try Again
        </Button>
      </div>
    )
  }

  const hasFiles = Object.values(files).some(arr => arr && arr.length > 0)

  return (
    <div className="flex-1 space-y-8 p-8 pt-6 bg-gradient-to-br from-background via-background to-muted/20 min-h-screen">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
          <BarChart3 className="h-8 w-8 text-primary" />
        </div>
        <h1 className="text-4xl font-bold text-foreground bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
          Reports & Downloads
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          View generated charts and download CSV data files from all analytical models
        </p>
      </div>

      {!hasFiles ? (
        <Card className="shadow-xl border-border/50 bg-gradient-to-br from-card to-card/50 max-w-md mx-auto">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-16 w-16 text-muted-foreground/50 mb-4" />
            <h3 className="text-xl font-medium text-foreground mb-2">No Reports Available</h3>
            <p className="text-muted-foreground text-center mb-6">
              Run analytics to generate reports and visualizations that can be viewed and downloaded here.
            </p>
            <Button onClick={fetchFiles} variant="outline" className="hover:bg-primary hover:text-primary-foreground transition-colors duration-200">
              Refresh
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="max-w-7xl mx-auto">
          <Accordion type="multiple" className="space-y-4">
            {files.descriptive && files.descriptive.length > 0 &&
              renderFileSection(
                "Descriptive Analytics",
                "descriptive",
                files.descriptive,
                <BarChart3 className="h-5 w-5" />,
                "Historical data analysis, trends, and clustering insights"
              )
            }

            {files.predictive && files.predictive.length > 0 &&
              renderFileSection(
                "Predictive Analytics",
                "predictive",
                files.predictive,
                <TrendingUp className="h-5 w-5" />,
                "Forecasting models and predictive insights"
              )
            }

            {files.prescriptive && files.prescriptive.length > 0 &&
              renderFileSection(
                "Prescriptive Analytics",
                "prescriptive",
                files.prescriptive,
                <Zap className="h-5 w-5" />,
                "Optimization recommendations and actionable insights"
              )
            }
          </Accordion>
        </div>
      )}
    </div>
  )
}
