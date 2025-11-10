import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, FileText, Image, Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

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
      const response = await fetch("http://localhost:5050/api/analytics/files")
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
      const response = await fetch(`http://localhost:5050/api/analytics/download/${category}/${filename}`)
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

  const renderFileSection = (title: string, category: keyof FilesData, filesList: FileInfo[]) => {
    const csvFiles = filesList.filter(f => f.type === 'csv')
    const pngFiles = filesList.filter(f => f.type === 'png')

    return (
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            {title}
          </CardTitle>
          <CardDescription>
            Download generated reports and visualizations from {title.toLowerCase()} analytics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* PNG Files - Display as images */}
          {pngFiles.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
                <Image className="h-4 w-4" />
                Generated Charts & Visualizations
              </h4>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {pngFiles.map((file) => (
                  <div key={file.name} className="border rounded-lg p-4 space-y-3">
                    <div className="aspect-video bg-muted rounded flex items-center justify-center">
                      <img
                        src={`http://localhost:5050/api/analytics/download/${category}/${file.name}`}
                        alt={file.name}
                        className="max-w-full max-h-full object-contain rounded"
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
                        className="ml-2"
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
              <h4 className="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Data Files
              </h4>
              <div className="space-y-2">
                {csvFiles.map((file) => (
                  <div key={file.name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium">{file.name}</span>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleDownload(category, file.name)}
                      disabled={downloading === `${category}-${file.name}`}
                    >
                      {downloading === `${category}-${file.name}` ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-2" />
                          Download CSV
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
              No files available for {title.toLowerCase()} analytics
            </div>
          )}
        </CardContent>
      </Card>
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
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Reports & Downloads</h1>
        <p className="text-muted-foreground">
          View generated charts and download CSV data files from all analytical models
        </p>
      </div>

      {!hasFiles ? (
        <Card className="shadow-soft">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No Reports Available</h3>
            <p className="text-muted-foreground text-center mb-4">
              Run analytics to generate reports and visualizations that can be viewed and downloaded here.
            </p>
            <Button onClick={fetchFiles} variant="outline">
              Refresh
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {files.descriptive && files.descriptive.length > 0 &&
            renderFileSection("Descriptive Analytics", "descriptive", files.descriptive)
          }

          {files.prescriptive && files.prescriptive.length > 0 &&
            renderFileSection("Prescriptive Analytics", "prescriptive", files.prescriptive)
          }

          {files.predictive && files.predictive.length > 0 &&
            renderFileSection("Predictive Analytics", "predictive", files.predictive)
          }
        </div>
      )}
    </div>
  )
}
