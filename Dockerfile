# Use Node.js 22 with more memory
FROM node:22

# Set environment variables for better npm performance
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV npm_config_cache=/tmp/.npm

# Install Python 3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY backend/package*.json ./backend/
COPY frontend/package*.json ./frontend/

# Install Node.js dependencies with retry and verbose logging
RUN npm cache clean --force && \
    npm install --verbose --no-optional || npm install --verbose --no-optional

# Install backend dependencies
RUN cd backend && npm install --verbose --no-optional || npm install --verbose --no-optional

# Install frontend dependencies with specific flags to avoid SWC issues
RUN cd frontend && \
    npm install --verbose --no-optional --legacy-peer-deps || \
    npm install --verbose --no-optional --legacy-peer-deps

# Copy Python requirements and install
COPY backend/requirements.txt ./backend/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r backend/requirements.txt

# Copy source code
COPY . .

# Build the frontend
RUN npm run build

# Expose port
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5050/health || exit 1

# Start the application
CMD ["npm", "start"]
