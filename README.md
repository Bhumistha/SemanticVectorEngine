# Semantic Search System

This project implements a scalable semantic search engine using:

- Sentence Transformers
- FAISS vector database
- Semantic caching
- Query clustering
- Cross encoder reranking
- FastAPI deployment
- Docker containerization

## Architecture

Query → Embedding → Cache → Cluster Routing → FAISS Search → Reranking → Results

## Installation

git clone repo

cd semantic-search-system

pip install -r requirements.txt

## Run API

uvicorn main:app --reload

## Docker

docker build -t semantic-search .

docker run -p 8000:8000 semantic-search