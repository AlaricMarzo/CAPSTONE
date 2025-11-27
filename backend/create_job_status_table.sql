-- Create job_status table for tracking file upload job progress
CREATE TABLE IF NOT EXISTS job_status (
  job_id VARCHAR(36) PRIMARY KEY,
  status VARCHAR(50) NOT NULL DEFAULT 'pending',
  progress INTEGER DEFAULT 0,
  message TEXT,
  data JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_job_status_updated_at ON job_status(updated_at);
CREATE INDEX IF NOT EXISTS idx_job_status_status ON job_status(status);

-- Add comment
COMMENT ON TABLE job_status IS 'Tracks the status and progress of file upload and processing jobs';
