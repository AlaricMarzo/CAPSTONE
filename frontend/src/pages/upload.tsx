"use client"

import type React from "react"
import { useState } from "react"
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react"

export default function DataUploadPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const resetForm = () => {
  setSelectedFiles([])
  setResult(null)
 }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files))
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return

    setUploading(true)
    setResult(null)

    try {
      const formData = new FormData()
      selectedFiles.forEach((file) => {
        formData.append("files", file)
      })

      const response = await fetch("http://localhost:4000/api/upload", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()
      setResult(data)

      if (data.success) {
        setSelectedFiles([])
      }
    } catch (error: any) {
      setResult({
        success: false,
        error: error.message,
      })
    } finally {
      setUploading(false)
    }
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
              <button
                onClick={handleUpload}
                disabled={selectedFiles.length === 0 || uploading}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" />
                    Upload and Process
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {result && (
          <div
            className={`rounded-lg border p-4 ${result.success ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}
          >
            <div className="flex gap-2">
              {result.success ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <XCircle className="h-5 w-5 text-red-600" />
              )}
              <div className="flex-1">
                {result.success ? (
                  <div className="space-y-2">
                    <p className="font-medium">Upload successful!</p>
                    {result.loadResult && (
                      <div className="text-sm">
                        <p>Files processed: {result.filesProcessed}</p>
                        <p>Rows loaded: {result.loadResult.rows_loaded}</p>
                        <p>Status: {result.loadResult.status || "completed"}</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-red-800">{result.error}</p>
                )}
              </div>
            </div>
          </div>
        )}

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
