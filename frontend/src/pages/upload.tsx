"use client"

import type React from "react"
import { useState } from "react"
import { Upload, FileText, CheckCircle, XCircle, Loader2, Shield, Database } from "lucide-react"

export default function DataUploadPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [result, setResult] = useState<any>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [pollingInterval, setPollingInterval] = useState<number | null>(null)

  const resetForm = () => {
    setSelectedFiles([])
    setResult(null)
    setJobId(null)
    if (pollingInterval) {
      clearInterval(pollingInterval)
      setPollingInterval(null)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files))
      setResult(null)
    }
  }

  const pollJobStatus = async (jobId: string) => {
    try {
      const response = await fetch(`http://localhost:5050/api/job/${jobId}`)
      const data = await response.json()

      if (data.success) {
        setUploadProgress(data.progress)
        setResult({
          success: data.status === 'completed',
          error: data.error,
          filesProcessed: data.filesProcessed,
          cleaningResults: data.cleaningResults,
          message: data.message,
          runId: jobId,
          rowsProcessed: data.cleaningResults?.rowsProcessed || 0
        })

        if (data.status === 'completed' || data.status === 'failed') {
          setUploading(false)
          if (pollingInterval) {
            clearInterval(pollingInterval)
            setPollingInterval(null)
          }
        }
      }
    } catch (error) {
      console.error('Error polling job status:', error)
    }
  }

  const handleUpload = () => {
    if (selectedFiles.length === 0) return

    setUploading(true)
    setUploadProgress(0)
    setResult(null)
    setJobId(null)

    const formData = new FormData()
    selectedFiles.forEach((file) => {
      formData.append("files", file)
    })

    fetch("http://localhost:    /api/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success && data.jobId) {
          setJobId(data.jobId)
          // Start polling for job status
          const interval = setInterval(() => pollJobStatus(data.jobId), 1000)
          setPollingInterval(interval)
        } else {
          setResult({ success: false, error: data.error || "Upload failed" })
          setUploading(false)
        }
      })
      .catch((error) => {
        setResult({ success: false, error: error.message })
        setUploading(false)
      })
  }

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="mx-auto max-w-4xl space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900">Sales Data Pipeline</h1>
          <p className="mt-2 text-lg text-gray-600">Upload multiple CSV files for cleaning and database loading</p>
        </div>

        <div className="rounded-lg border bg-white shadow-sm">
          <div className="p-6 space-y-4">
            <div>
              <h3 className="text-lg font-semibold">Upload CSV Files</h3>
              <p className="text-sm text-gray-600">
                Select one or more CSV files containing sales data. Files will be validated, cleaned, and loaded into
                the database.
              </p>
            </div>

            <div className="flex items-center gap-4">
              <label
                htmlFor="file-input"
                className="px-4 py-2 border rounded-md hover:bg-gray-50 flex items-center gap-2 cursor-pointer"
              >
                <Upload className="h-4 w-4" />
                Select Files
              </label>
              <input
                id="file-input"
                type="file"
                multiple
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              <span className="text-sm text-gray-500">
                {selectedFiles.length === 0 ? "No files selected" : `${selectedFiles.length} file(s) selected`}
              </span>
            </div>

            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Selected Files:</p>
                <div className="space-y-2">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between rounded-lg border bg-white p-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-blue-500" />
                        <span className="text-sm">{file.name}</span>
                        <span className="text-xs text-gray-500">({(file.size / 1024).toFixed(2)} KB)</span>
                      </div>
                      <button className="p-1 hover:bg-gray-100 rounded" onClick={() => removeFile(index)}>
                        <XCircle className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result?.success ? (
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  disabled
                  className="w-full sm:w-auto px-4 py-2 bg-green-600 text-white rounded-md flex items-center justify-center gap-2"
                >
                  <CheckCircle className="h-4 w-4" />
                  Imported successfully
                </button>
                <button
                  onClick={resetForm}
                  className="w-full sm:w-auto px-4 py-2 border rounded-md hover:bg-gray-50"
                >
                  Upload more files
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <button
                  onClick={handleUpload}
                  disabled={selectedFiles.length === 0 || uploading}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload and Process
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6">
            <h3 className="text-xl font-semibold text-white flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Processing Progress
            </h3>
          </div>

          <div className="p-6">
            {uploading ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">Processing Progress</span>
                  <span className="text-sm font-bold text-blue-600">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-indigo-600 h-3 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-600 flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                  {uploadProgress < 20 ? "Uploading files to server..." :
                   uploadProgress < 40 ? "Validating file headers..." :
                   uploadProgress < 60 ? "Normalizing units and extracting dates..." :
                   uploadProgress < 80 ? "Filtering data quality..." :
                   uploadProgress < 100 ? "Running data cleaning script..." :
                   "Finalizing results..."}
                </p>
              </div>
            ) : result ? (
              <div className="space-y-4">
                {result.success ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <span className="font-semibold text-green-800">Files Processed</span>
                      </div>
                      <p className="text-2xl font-bold text-green-600">{result.filesProcessed}</p>
                    </div>
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Database className="h-5 w-5 text-blue-600" />
                        <span className="font-semibold text-blue-800">Rows Loaded</span>
                      </div>
                      <p className="text-2xl font-bold text-blue-600">{result.rowsProcessed || 0}</p>
                    </div>
                    <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 md:col-span-2">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Shield className="h-4 w-4 text-purple-600" />
                          <span className="font-semibold text-purple-800">Processing Details</span>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <p><strong>Status:</strong> Completed</p>
                          <p><strong>Run ID:</strong> {result.runId}</p>
                          {result.message && <p><strong>Message:</strong> {result.message}</p>}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <XCircle className="h-5 w-5 text-red-600" />
                      <span className="font-semibold text-red-800">Processing Failed</span>
                    </div>
                    <p className="text-red-700">{result.error}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No processing in progress</p>
                <p className="text-sm text-gray-400 mt-1">Upload files to see processing status</p>
              </div>
            )}
          </div>
        </div>



        <div className="rounded-lg border bg-white shadow-sm p-6">
          <h3 className="text-lg font-semibold mb-4">Pipeline Steps</h3>
          <ol className="space-y-3 text-sm">
            <li className="flex gap-2">
              <span className="font-semibold">1.</span>
              <span>Header validation - Ensures all required columns are present</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">2.</span>
              <span>Unit normalization - Standardizes units (pcs, box, kg, etc.)</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">3.</span>
              <span>Expiration date extraction - Parses dates from descriptions</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">4.</span>
              <span>Data quality filtering - Removes incomplete, duplicate, and placeholder rows</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">5.</span>
              <span>Transaction classification - Tags as SALE, RETURN, or ADJUSTMENT</span>
            </li>
            <li className="flex gap-2">
              <span className="font-semibold">6.</span>
              <span>Database loading - Loads into staging, dimensions, and fact tables</span>
            </li>
          </ol>
        </div>
      </div>
    </div>
  )
}
