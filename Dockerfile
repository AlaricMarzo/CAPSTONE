FROM node:18-alpine

# Install Python 3 + pip
RUN apk add --no-cache python3 py3-pip

# Create symlinks so "python" and "python3" both work
RUN ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/python3 /usr/local/bin/python3

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python dependencies inside venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install root Node dependencies
COPY package*.json .
RUN npm install

# Copy full project
COPY . .

# Install frontend dependencies
RUN cd frontend && npm install

# Install backend dependencies
RUN cd backend && npm install

WORKDIR /app/backend
EXPOSE 8080

CMD ["npm", "start"]
