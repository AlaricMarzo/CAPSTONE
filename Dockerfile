FROM node:18-alpine

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

# Build the application
RUN npm run build

# Expose ports
EXPOSE 8080 5050

# Start the application
CMD ["npm", "start"]
