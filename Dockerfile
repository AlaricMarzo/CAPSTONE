FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy root package files and install root dependencies
COPY package*.json ./
RUN npm install

# Copy and install frontend dependencies
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

# Copy and install backend dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm install

# Copy source code
COPY frontend/ ./frontend/
COPY backend/ ./backend/

# Build the frontend (backend doesn't have a build script, so skip build:backend)
RUN npm run build:frontend

# Expose ports
EXPOSE 8080 5050

# Start the application
CMD ["npm", "start"]
