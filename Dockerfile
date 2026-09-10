FROM nginx:alpine-slim

# Replace default Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy pre-compiled static documentation
COPY site /usr/share/nginx/html

# Cloud Run default port
EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
