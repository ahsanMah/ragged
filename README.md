# ragged

A locally-hosted Retrieval Augmented Generation (RAG) system for processing and querying PDF documents.

## Overview

This project creates a web-based interface for document analysis using:
- Local LLM inference with quantized GGUF models
- Document chunking and embedding using sentence transformers
- Semantic search via k-NN retrieval
- Interactive UI with Gradio

## Usage

1. Start the application: `python ui.py`
2. Upload PDF documents and click "Process Documents"
3. Ask questions in the chat interface
4. View responses based on document content

## Project Roadmap

- [x] Basic PDF parsing and embedding
- [x] Simple Gradio UI implementation
- [x] k-NN retrieval of relevant chunks
- [x] LLM integration with context-enhanced prompts
- [x] Better chunking - using sentence boundaries
- [ ] Fit KNN when processing embeddings (improve efficiency)
- [ ] UI block for viewing retrieved snippets
- [ ] Store metadata (filename, page) alongside text chunks
- [ ] Project directory with files and cache management
- [ ] Add cancellation controls for processing/generation
- [ ] Research persistent vector storage solutions
- [ ] Summarized chunks via smaller LLM

## Implementation Details

### Tech Stack

- **LLM**: DeepSeek-R1-Distill-Qwen-7B quantized via GGUF
- **Embedding Model**: NovaSearch/stella_en_400M_v5 via SentenceTransformers
- **Document Processing**: PyPDF for text extraction
- **Vector Search**: Scikit-learn NearestNeighbors
- **Interface**: Gradio web UI to upload documents and chat through chatbot interface
- **GPU Acceleration**: Optional GPU offloading for LLM inference


### Document Processing Pipeline

1. **Parsing**: Extract text from PDFs using PyPDF
2. **Chunking**: Split text into manageable chunks (currently 256 words)
3. **Embedding**: Generate vector representations using SentenceTransformer
4. **Storage**: Save embeddings and text chunks to numpy arrays

### Retrieval and Generation

1. **Query Embedding**: Convert user query to vector representation
2. **k-NN Search**: Find most similar document chunks
3. **Context Construction**: Combine relevant chunks into prompt context
4. **LLM Generation**: Generate response with context-enhanced prompt

## Development Log

### Week - 2/24/2025

- Built GUI using Gradio
  - Implemented file upload and chat interface
  - Created processing workflow for documents
  - Updated embedding pipeline
    - 256-word chunks with SentenceTransformer (number of tokens probably higher)
    - Numpy storage for embeddings and text chunks

### Week - 2/17/2025

- Implemented PDF parsing + embedding + LLM prompting
- Testing observations:
  - Specific queries perform better than general ones
  - R1 distilled models' thinking steps *seem* to improve responses
  - 7B model with Q4/Q5 quantization offers good performance
  - n_gpu_layers parameter controls GPU offloading
  - Successfully tested on AMD with Vulkan

## Technical Notes

### RAG Implementation

- Query-based search of precomputed document embeddings
- Context chunks concatenated with query for LLM generation
- Using STELLA model for high-quality embeddings

### Text Processing Challenges

- Current chunking can split sentences at hard boundaries
- Potential improvements:
  - Respecting paragraph/sentence boundaries 
  - Using sliding windows for better coverage
  - Adding summarization of larger contexts

### LLM Configuration

- Using DeepSeek-R1-Distill-Qwen quantized models
- Q4/Q5 quantization offers good performance/quality balance
- Loaded via llama-cpp-python for efficient inference
- Working with 65K context window

### Embedding Strategy

- Sentence-transformers for vector generation
- k-NN for similarity search
- Embedding stored in numpy arrays for simplicity
- Future work: persistent vector storage

## Requirements

- Python 3.8+
- llama-cpp-python (optionally compiled with GPU support)
- SentenceTransformers
- NumPy
- Gradio
- Torch

## Future Directions

- Support for more document types (DOCX, HTML, TXT)
- Persistent vector database integration
- Improved chunking with semantic boundaries
- Reranking for better retrieval precision
- User-selectable LLM models
- Metadata-aware retrieval

