FROM node:18-alpine

# Install Python and pip
RUN apk add --no-cache python3 py3-pip

# Create a virtual environment and install packages
RUN python3 -m venv /opt/venv
COPY requirements.txt ./
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy frontend and backend
COPY frontend/ ./frontend/
COPY backend/ ./backend/

# Install frontend dependencies
RUN cd frontend && npm install

# Install backend dependencies
RUN cd backend && npm install

# Build the application
RUN npm run build

# Expose ports
EXPOSE 8080 5050

# Start the application with virtual environment activated
CMD ["/opt/venv/bin/python3", "-c", "import sys; sys.path.insert(0, '/opt/venv/lib/python3.11/site-packages'); import subprocess; subprocess.run(['npm', 'start'])"]
