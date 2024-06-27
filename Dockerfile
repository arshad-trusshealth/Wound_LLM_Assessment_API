# Use the official Python image from the Docker Hub
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . .

RUN /bin/bash -c './install-ollama.sh'
# RUN /bin/bash -c './start-ollama.sh'

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port your Flask app will listen on
EXPOSE 8081

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8081"]
