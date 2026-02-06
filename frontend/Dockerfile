# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Configure Nginx for non-root execution
# 1. Remove the 'user' directive from the main config to avoid warnings
# 2. Change pid file location to /tmp/nginx.pid
# 3. Create cache directories and set permissions
RUN sed -i 's/^user/#user/' /etc/nginx/nginx.conf && \
    sed -i 's|/var/run/nginx.pid|/tmp/nginx.pid|' /etc/nginx/nginx.conf && \
    mkdir -p /var/cache/nginx /var/log/nginx /tmp/nginx && \
    chown -R 1000:0 /var/cache/nginx /var/log/nginx /etc/nginx/conf.d /tmp/nginx /usr/share/nginx/html

# Create non-root user for security (reuse existing user or create new)
# nginx:alpine already has 'nginx' user, but we can stick to the plan or use generic ID
# using numeric ID is safer for k8s/etc.
USER 1000

# Expose port 8080
EXPOSE 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
