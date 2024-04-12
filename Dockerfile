FROM python:3.10-slim
ARG openai_api_key
ENV OPENAI_API_KEY $openai_api_key
WORKDIR /app
RUN apt-get update && \
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]
