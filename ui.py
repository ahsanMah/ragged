import gradio as gr
import time
import random

from ragged.embedding import Embedder
from ragged.parsers import Parser

# Dummy functions for demonstration purposes
def process_documents(documents):
    """
    This function processes uploaded documents:
    1. Extracts text from documents in chunks
    2. Creates embeddings
    3. TODO: Stores in vector database
    """
    
    parser = Parser(chunksize=128)
    embedder = Embedder(device="cpu")

    processed_files = []
    embeddings = []
    for doc in documents:
        # In a real app, you would process each document
        processed_files.append(doc.name)
        text_chunks = list(parser.parse_pdf(doc))
        embeddings.extend(embedder.embed(text_chunks))
    
    return f"Processed {len(processed_files)} documents\nExtracted: {len(embeddings)} embeddings"

def generate_response(query: str, history: dict, documents: list):
    """
    Dummy function to generate streaming responses from the LLM.
    In a real application, this would:
    1. Retrieve relevant context from the vector DB
    2. Send query + context to LLM
    3. Stream the response
    """
    print(query, documents, history)
    if not documents:
        yield "Please upload documents first!"
        return
    
    # Simulate streaming response
    response = []
    all_words = f"Based on the documents you've provided, I can answer that {query} " + \
                f"requires careful analysis"
                
    words = all_words.split()
    
    for word in words:
        response.append(word)
        time.sleep(0.1)  # Simulating stream delay
        yield " ".join(response)

# Define the Gradio interface
def create_rag_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LLM RAG Application")
        gr.Markdown("Upload documents and ask questions based on their content")
        
        with gr.Row():
            # Left panel for document upload
            with gr.Column(scale=1):
                gr.Markdown("### Document Upload")
                file_output = gr.Textbox(label="Processing Status")
                uploaded_files = gr.File(
                    file_count="multiple",
                    label="Upload Documents",
                    file_types=[".pdf", ".txt", ".docx"]
                )
                upload_button = gr.Button("Process Documents")
                
            # Right panel for Q&A
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions")
                chatbot = gr.ChatInterface(
                    fn=generate_response,
                    type="messages",
                    stop_btn=True,
                    additional_inputs=[uploaded_files],
                    description="Ask something about the uploaded documents..."
                )
                
        # Set up callbacks
        upload_button.click(
            fn=process_documents,
            inputs=uploaded_files,
            outputs=file_output
        )
        
    return demo

# Create and launch the app

# if __name__ == "__main__":
demo = create_rag_interface()
demo.queue()  # Enable queueing for better handling of multiple users
demo.launch(share=False)  # Set share=False in production