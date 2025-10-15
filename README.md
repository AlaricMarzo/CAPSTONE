# CAPSTONE Project

Full-stack application with React frontend and Node.js backend.

## Project Structure

\`\`\`
CAPSTONE/
├── frontend/          # React + Vite frontend application
├── backend/           # Node.js backend API
├── scripts/           # Python data processing scripts
├── uploads/           # Uploaded files directory
└── README.md          # This file
\`\`\`

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0
- Python 3.x (for data processing scripts)
- PostgreSQL database

### Installation

Install dependencies for both frontend and backend:

\`\`\`bash
npm run install:all
\`\`\`

Or install individually:

\`\`\`bash
# Frontend only
npm run install:frontend

# Backend only
npm run install:backend
\`\`\`

Install Python dependencies:

\`\`\`bash
pip install pandas psycopg2-binary python-dotenv
\`\`\`

### Development

Run both frontend and backend concurrently:

\`\`\`bash
npm run dev
\`\`\`

Or run individually:

\`\`\`bash
# Frontend only (usually runs on http://localhost:5173)
npm run dev:frontend

# Backend only (check backend package.json for port)
npm run dev:backend
\`\`\`

### Building for Production

Build both applications:

\`\`\`bash
npm run build
\`\`\`

Or build individually:

\`\`\`bash
npm run build:frontend
npm run build:backend
\`\`\`

## Environment Variables

Create `.env` files in both `frontend/` and `backend/` directories:

### Frontend (.env)
\`\`\`
VITE_API_URL=http://localhost:3000
\`\`\`

### Backend (.env)
\`\`\`
PORT=3000
DATABASE_URL=your_database_url
\`\`\`

### Root (.env) - For Python Scripts
\`\`\`
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
\`\`\`

## Tech Stack

### Frontend
- React
- Vite
- Tailwind CSS
- TypeScript

### Backend
- Node.js
- Express (or your backend framework)
- TypeScript

### Data Processing
- Python 3.x
- pandas
- psycopg2

## Features

### CSV Upload & Processing
- Upload multiple CSV files through the web interface
- Automatic data cleaning and validation
- Database loading with error tracking
- Duplicate detection and handling

### Data Cleaning
The cleaning script (`scripts/clean_data.py`) performs:
- Column standardization
- Data type validation
- Missing value handling
- Duplicate removal
- Error logging

### Database Loading
The loading script (`scripts/load_to_database.py`) handles:
- Batch inserts for performance
- Duplicate detection
- Error tracking and reporting
- Transaction management

## Development Workflow

1. Make changes in either `frontend/` or `backend/` directories
2. Both servers will hot-reload automatically
3. Frontend proxies API requests to backend during development
4. Commit changes to git as usual

## Deployment

- Frontend: Deploy to Vercel, Netlify, or any static hosting
- Backend: Deploy to Vercel, Railway, Render, or any Node.js hosting

Make sure to set environment variables in your deployment platform.

## Troubleshooting

### Python Script Errors
- Ensure Python 3.x is installed and accessible via `python3` command
- Check that all Python dependencies are installed
- Verify DATABASE_URL is correctly set in root .env file
- Check console logs for detailed error messages

### Database Connection Issues
- Verify PostgreSQL is running
- Check DATABASE_URL format and credentials
- Ensure database exists and user has proper permissions

### Upload Issues
- Check that `uploads/` directory has write permissions
- Verify file size limits in your deployment platform
- Check browser console for client-side errors
