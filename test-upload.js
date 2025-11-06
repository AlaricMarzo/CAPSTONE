const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

// Path to a CSV file to upload
const filePath = path.join(__dirname, 'ANC - 4 YEARS (1).csv');

async function testUpload() {
  try {
    // Create a form data object
    const formData = new FormData();
    formData.append('files', fs.createReadStream(filePath));

    // Upload the file
    console.log('Uploading file...');
    const response = await axios.post('http://localhost:5050/api/upload', formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    console.log('Upload response:', JSON.stringify(response.data, null, 2));

    // If we got a job ID, poll for status
    if (response.data.jobId) {
      const jobId = response.data.jobId;
      console.log(`\nPolling job status for job ID: ${jobId}`);
      
      // Poll every 2 seconds
      const interval = setInterval(async () => {
        try {
          const statusResponse = await axios.get(`http://localhost:5050/api/job/${jobId}`);
          console.log(`\nJob status (${new Date().toLocaleTimeString()}):`, 
            JSON.stringify(statusResponse.data, null, 2));
          
          // If job is completed or failed, stop polling
          if (statusResponse.data.status === 'completed' || statusResponse.data.status === 'failed') {
            clearInterval(interval);
            
            // If completed, run analytics
            if (statusResponse.data.status === 'completed') {
              console.log('\n=== RUNNING DESCRIPTIVE ANALYTICS ===');
              try {
                const descriptiveResponse = await axios.post('http://localhost:5050/api/analytics/run-descriptive');
                console.log('Descriptive analytics response:', 
                  JSON.stringify(descriptiveResponse.data, null, 2));
                
                console.log('\n=== RUNNING PREDICTIVE ANALYTICS ===');
                const predictiveResponse = await axios.post('http://localhost:5050/api/analytics/run-predictive');
                console.log('Predictive analytics response:', 
                  JSON.stringify(predictiveResponse.data, null, 2));
                
                console.log('\n=== TEST COMPLETED SUCCESSFULLY ===');
              } catch (analyticsError) {
                console.error('Error running analytics:', analyticsError.message);
                if (analyticsError.response) {
                  console.error('Response data:', analyticsError.response.data);
                }
              }
            }
          }
        } catch (error) {
          console.error('Error polling job status:', error.message);
          clearInterval(interval);
        }
      }, 2000);
    }
  } catch (error) {
    console.error('Error uploading file:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
  }
}

testUpload();
