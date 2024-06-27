FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
WORKDIR /app
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x ./install-ollama.sh
RUN chmod +x ./start-ollama.sh
RUN /bin/bash -c './install-ollama.sh'
EXPOSE 8081
ENV FLASK_APP=app.py
ENV OLLAMA_HOST=0.0.0.0
CMD ["/bin/bash", "-c", "./start-ollama.sh && flask run --host=0.0.0.0 --port=8081"]
