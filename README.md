SemanticVectorEngine








SemanticVectorEngine is a scalable AI-powered semantic search system that retrieves contextually similar results using sentence embeddings and vector similarity search. The system integrates FAISS indexing, semantic caching, query clustering, and cross-encoder reranking to improve search performance and accuracy.

The project is built with FastAPI for backend APIs, Streamlit for an interactive interface, and Docker for containerized deployment.

Highlights

Production-style semantic search pipeline

Fast vector similarity search using FAISS

Semantic caching to avoid repeated computations

Query clustering for optimized search routing

Cross-encoder reranking for improved result relevance

FastAPI backend for scalable API services

Streamlit interface for interactive querying

Dockerized deployment for portability

Features

Semantic similarity search using Sentence Transformers

High-performance vector search using FAISS

Semantic caching for faster repeated queries

Query clustering for efficient routing

Cross-encoder reranking for improved ranking quality

REST API built with FastAPI

Interactive UI using Streamlit

Containerized deployment using Docker

System Architecture

Search pipeline flow:

Query
↓
Embedding Generation
↓
Semantic Cache Check
↓
Cluster Routing
↓
FAISS Vector Search
↓
Cross Encoder Reranking
↓
Final Ranked Results

Architecture Diagram:

Demo

Example interface of the semantic search system:

Tech Stack

Python
FastAPI
FAISS
Sentence Transformers
Streamlit
NumPy
Scikit-learn
Docker

Project Structure

api/ — FastAPI backend
services/ — Search pipeline services
vector_db/ — FAISS index management
clustering/ — Query clustering logic
cache/ — Semantic cache implementation
analysis/ — Cluster analysis and visualization
scripts/ — Index building scripts
tests/ — Unit tests
docs/ — Architecture diagrams
streamlit_app.py — Streamlit UI

Installation

Clone the repository:

git clone https://github.com/Bhumistha/SemanticVectorEngine.git

cd SemanticVectorEngine

Create virtual environment:

python -m venv venv

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the API

Start the FastAPI server:

uvicorn api.main:app --reload

API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

Run the Streamlit Interface

streamlit run streamlit_app.py

Open in browser:

http://localhost:8501

Docker Deployment

Build Docker image:

docker build -t semantic-vector-engine .

Run container:

docker run -p 8000:8000 semantic-vector-engine

Future Improvements

Distributed FAISS indexing
Hybrid search (semantic + keyword search)
Query intent detection
Cloud deployment with scalable vector databases

Author

Bhumistha Sahoo

GitHub:
https://github.com/Bhumistha
