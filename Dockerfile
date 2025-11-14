FROM node:18-alpine

# Install Python and pip
RUN apk add --no-cache python3 py3-pip

# All work happens under /app
WORKDIR /app

# --- Python setup (virtualenv + requirements) ---

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Make the venv Python/pip the default in this container
ENV PATH="/opt/venv/bin:$PATH"

# Copy Python requirements and install them into the venv
# (adjust path if your requirements file is elsewhere)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# --- Node / app setup ---

# Copy root package files and install root deps
COPY package*.json ./
RUN npm install

# Copy frontend and backend source
COPY frontend/ ./frontend/
COPY backend/ ./backend/

# Install frontend deps and build frontend
RUN cd frontend && npm install && npm run build

# Install backend deps
RUN cd backend && npm install

# Expose ports (frontend 8080, backend 5050)
EXPOSE 8080 5050

# Tell your Node backend which Python to use
# (server.js reads process.env.PYTHON_CMD || 'python3')
ENV PYTHON_CMD=python3

# Start the app (root package.json "start" should start backend & serve frontend)
CMD ["npm", "start"]
