async function processUpload(jobId, files) {
  try {
    jobs.set(jobId, { status: 'processing', progress: 10, message: 'Files uploaded, starting data cleaning...' })

    const cleanedDir = path.join(__dirname, "../../cleaned")
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true })

    const uploadedFiles = files.map((file) => file.path)
    const timestamp = Date.now()
    const outputPath = path.join(cleanedDir, `combined_cleaned_${timestamp}.csv`)
    const cleanScriptPath = path.join(__dirname, "../../scripts/clean_data.py")

    console.log("[Backend] Starting data cleaning and loading...")
    jobs.set(jobId, { status: 'processing', progress: 20, message: 'Validating file headers...' })

    // Simulate header validation progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 40, message: 'Normalizing units and extracting dates...' })

    // Simulate unit normalization progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 60, message: 'Filtering data quality...' })

    // Simulate data quality filtering progress
    await new Promise(resolve => setTimeout(resolve, 500))
    jobs.set(jobId, { status: 'processing', progress: 80, message: 'Running data cleaning script...' })
=======
// -------------------- Background Processing Function --------------------
async function processUpload(jobId, files) {
  try {
    const jobData1 = { status: 'processing', progress: 10, message: 'Files uploaded, starting data cleaning...' }
    jobs.set(jobId, jobData1)
    await saveJobToDB(jobId, jobData1)

    const cleanedDir = path.join(__dirname, "../../cleaned")
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true })

    const uploadedFiles = files.map((file) => file.path)
    const timestamp = Date.now()
    const outputPath = path.join(cleanedDir, `combined_cleaned_${timestamp}.csv`)
    const cleanScriptPath = path.join(__dirname, "../../scripts/clean_data.py")

    console.log("[Backend] Starting data cleaning and loading...")
    const jobData2 = { status: 'processing', progress: 20, message: 'Validating file headers...' }
    jobs.set(jobId, jobData2)
    await saveJobToDB(jobId, jobData2)

    // Simulate header validation progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData3 = { status: 'processing', progress: 40, message: 'Normalizing units and extracting dates...' }
    jobs.set(jobId, jobData3)
    await saveJobToDB(jobId, jobData3)

    // Simulate unit normalization progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData4 = { status: 'processing', progress: 60, message: 'Filtering data quality...' }
    jobs.set(jobId, jobData4)
    await saveJobToDB(jobId, jobData4)

    // Simulate data quality filtering progress
    await new Promise(resolve => setTimeout(resolve, 500))
    const jobData5 = { status: 'processing', progress: 80, message: 'Running data cleaning script...' }
    jobs.set(jobId, jobData5)
    await saveJobToDB(jobId, jobData5)
