import os
import time

import gradio as gr
import numpy as np
from llama_cpp import Llama

from config import Config
from ragged.embedding import Embedder
from ragged.parsers import Parser

llm = Llama.from_pretrained(
    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    filename="*Q4_K_M.gguf",
    # filename="*Q8_0.gguf",
    n_gpu_layers=24,
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=65536, # Uncomment to increase the context window
    verbose=True,
    local_dir=Config.MODEL_DIR
)


def process_documents(documents: list, cache: bool = False):
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
        print(text_chunks)
        embeddings.extend(embedder.embed(text_chunks))

    embeddings = np.asarray(embeddings)
    os.makedirs("/tmp/ragged", exist_ok=True)
    np.save("/tmp/ragged/embeddings.npy", embeddings, allow_pickle=False)
    np.save("/tmp/ragged/textchunks.npy", text_chunks, allow_pickle=False)

    del parser, embedder

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

    # Load embeddings
    embeddings = np.load("/tmp/ragged/embeddings.npy", allow_pickle=False)
    docs = np.load("/tmp/ragged/textchunks.npy", allow_pickle=False)
    embedder = Embedder(device="cpu")

    response = []
    # First response
    if not history:
        start_response = f"Based on the documents you've provided, I have extracted {len(embeddings)} snippets of information."
        response.append(start_response)

    knn_indices = embedder.k_nearest_neighbors(embeddings, query)
    retreived_docs = "\n".join(
        [f"snippet-{i}: {docs[idx]}" for i, idx in enumerate(knn_indices)]
    )

    prompt = f"""Query: {query}
    Using the following contextual information, respond to the query.
    Recall that the context should be taken as a **primary** source.
    You may use your knowledge to only adapt the context to answer the question.
    <context_start>
    {retreived_docs}
    <context_end>
    Response: <think>
    """
    print(prompt)

    output = llm.create_completion(
        prompt,  # Prompt
        max_tokens=None,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
        echo=True,  # Echo the prompt back in the output
        stream=True
    )  # Generate a completion, can also call create_completion

    print(">>>> Prompt:")
    print(prompt)
    print(">>>> Streaming output:")

    # Iterate over the output and print it.
    response = []
    for token in output:
        text = token["choices"][0]["text"]
        response.append(text)
        yield "".join(response)


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
                    file_types=[".pdf", ".txt", ".docx"],
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
                    description="Ask something about the uploaded documents...",
                )

        # Set up callbacks
        upload_button.click(
            fn=process_documents, inputs=uploaded_files, outputs=file_output
        )

    return demo


# Create and launch the app

# if __name__ == "__main__":
demo = create_rag_interface()
demo.queue()  # Enable queueing for better handling of multiple users
demo.launch(share=False)  # Set share=False in production
