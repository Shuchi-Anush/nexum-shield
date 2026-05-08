FROM node:22-alpine

WORKDIR /app

# Copy dependency manifests first

COPY package*.json ./

# Install dependencies

RUN npm install

# Copy application source

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
