FROM node:20.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Expose the React development server port (default is 3000)
EXPOSE 5173

# Command to run the React app (dev mode)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]