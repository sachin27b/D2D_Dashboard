# Start from Python 3.10.11 slim image
FROM python:3.10.11

# Install Java 17
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set working directory
WORKDIR /D2D_Dashboard

# Copy requirements first and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything into /app
COPY . .

# Now change into the dashboard directory
WORKDIR /D2D_Dashboard/dashboard

# Expose port (if needed)
EXPOSE 8050

# Run the dashboard script
CMD ["python", "dashboard.py"]

