FROM node:18

# Limit NumPy / OpenBLAS threads so Railway doesn't explode
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1

# Install Python 3, pip, and build deps for psycopg2
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev libpq-dev gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

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

# Ensure cleaned directory exists
RUN mkdir -p /app/backend/cleaned

# Frontend: install deps AND build to /app/frontend/dist
RUN cd frontend && npm install && npm run build
