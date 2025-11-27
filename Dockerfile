FROM node:18-alpine

# 1) Install Python + pip + venv and create symlinks
RUN apk add --no-cache python3 py3-pip python3-venv \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/python3 /usr/local/bin/python3

# 2) Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 3) Install Python dependencies inside venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Install root Node dependencies
COPY package*.json .
RUN npm install

# 5) Copy the whole project
COPY . .

# 6) Install frontend dependencies
RUN cd frontend && npm install

# 7) Install backend dependencies
RUN cd backend && npm install

WORKDIR /app/backend
EXPOSE 8080

CMD ["npm", "start"]
