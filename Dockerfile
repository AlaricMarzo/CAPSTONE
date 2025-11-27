FROM node:18-alpine

# Install Python 3, pip, and build deps for psycopg2
RUN apk add --no-cache python3 py3-pip python3-dev postgresql-dev gcc musl-dev

# Symlinks for python / python3
RUN ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/python3 /usr/local/bin/python3

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Python deps (inside venv)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Root Node deps
COPY package*.json .
RUN npm install

# Copy project
COPY . .

# Frontend: install deps AND build to /app/frontend/dist
RUN cd frontend && npm install && npm run build

# Backend deps
RUN cd backend && npm install

WORKDIR /app/backend
EXPOSE 8080

CMD ["npm", "start"]
