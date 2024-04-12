## ask-the-guidelines

Running NLM extractor:
```
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
```

Test image:
```
docker build -t ask-the-guidelines:latest --build-arg openai_api_key=$OPENAI_API_KEY .
docker run --rm -p 8501:8501 ask-the-guidelines:latest
```
