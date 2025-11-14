FROM node:18-alpine

# Install Python and pip
RUN apk add --no-cache python3 py3-pip

# Install pandas and required dependencies
RUN pip3 install --no-cache-dir pandas openpyxl

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

# Start the application
CMD ["npm", "start"]
