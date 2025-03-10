import json
import os
import pdb
import time
from typing import List, Optional

import gradio as gr
import numpy as np
from llama_cpp import Llama

from config import Config
from ragged.documents import Document
from ragged.embedding import Embedder
from ragged.parsers import Parser

llm = Llama.from_pretrained(
    repo_id="bartowski/microsoft_Phi-4-mini-instruct-GGUF",
    filename="*Q4_K_M.gguf",
    # repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    # filename="*q8_0.gguf",
    n_gpu_layers=20,
    n_ctx=32768,
    verbose=True,
    local_dir=Config.MODEL_DIR,
)


def llm_prompt_template(query, context):
    common_instruction = f"""Using the following contextual information, respond to the query.
    Recall that the context should be taken as a **primary** source.
    You may use your knowledge to only adapt the context to answer the question.
    Only answer the query. Keep your response brief. DO NOT ANSWER MORE THAN ASKED.
    <context>
    {context}
    </context>
    """

    system_message_flavored = f"""<|system|>{common_instruction}<|end|>
    <|user|>{query}<|end|><|assistant|>
    """

    return system_message_flavored


def process_documents(documents: List, project_dir: str | os.PathLike = "/tmp/ragged"):
    """
    This function processes uploaded documents:
    1. Extracts text from documents in chunks
    2. Creates embeddings
    3. TODO: Stores in vector database
    """

    parser = Parser(chunksize=256)
    embedder = Embedder(device="cpu")
    os.makedirs(project_dir, exist_ok=True)

    processed_files = []
    embeddings = []
    for doc in documents:
        processed_files.append(doc.name)
        # unzip tuples and zip into separate arrays
        text_chunks, metadata = zip(*list(parser.parse(doc)))
        embeddings = np.asarray(embedder.embed(text_chunks))

        doc = Document(doc.name, project_dir)
        doc.store(text_chunks, embeddings=embeddings, metadata=metadata)

    # embeddings = np.asarray(embeddings)
    # np.save(os.path.join(project_dir, "embeddings.npy"), embeddings, allow_pickle=False)
    # np.save("/tmp/ragged/textchunks.npy", text_chunks, allow_pickle=False)

    # serialized_metadata = [m.to_dict() for m in metadata]
    # with open("/tmp/ragged/metadata.json", "w") as f:
    #     json.dump(serialized_metadata, f)

    del parser, embedder

    return f"Processed {len(processed_files)} documents\nExtracted: {len(embeddings)} embeddings"


def generate_response(
    query: str, history: dict, reference_documents: List, project_dir: Optional[List]
):
    """
    Dummy function to generate streaming responses from the LLM.
    In a real application, this would:
    1. Retrieve relevant context from the vector DB
    2. Send query + context to LLM
    3. Stream the response
    """
    print(query, reference_documents, history)
    if not reference_documents:
        yield "Please upload documents first!"
        return

    # Load embeddings
    documents, embeddings_arr, text_chunks = [], [], []
    for doc in reference_documents:
        doc_kwargs = {"document_name": doc.name}
        if project_dir is not None:
            doc_kwargs.update("storage_dir", project_dir)
        doc = Document(**doc_kwargs)
        embeddings, text, _ = doc.load()
        embeddings_arr.append(embeddings)
        text_chunks.extend(text)
        documents.append(doc)

    embeddings_arr = np.concatenate(embeddings_arr)
    embedder = Embedder(device="cpu")

    response = []
    # First response
    if not history:
        start_response = f"Based on the documents you've provided, I have extracted {len(embeddings)} snippets of information."
        response.append(start_response)

    knn_indices = embedder.k_nearest_neighbors(embeddings, query)
    retreived_docs = "\n".join(
        [f"snippet-{i}: {text_chunks[idx]}" for i, idx in enumerate(knn_indices)]
    )
    # pdb.set_trace()
    if len(history) > 0:
        chat_history = []
        for message in history:
            if message["role"] == "user":
                role_token = "<|user|>"
            else:
                role_token = "<|assistant|>"

            chat_history.append(f"""{role_token}{message['content']}</s>""")
        prompt_history = "".join(chat_history)
    else:
        prompt_history = ""

    templated_query = llm_prompt_template(query, retreived_docs)
    prompt = "".join([prompt_history, templated_query])

    output = llm.create_completion(
        prompt,  # Prompt
        max_tokens=1024,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
        echo=False,  # Echo the prompt back in the output
        stream=True,
        stop=["<|User|>", "<|Assistant|>", "<|assistant|>", "<|user|>"],
    )  # Generate a completion, can also call create_completion

    print(">>>> Prompt:")
    print(prompt)
    print(">>>> Streaming output:")

    # Iterate over the output and print it.
    response = []
    for token in output:
        text = token["choices"][0]["text"]
        print(text, end="", flush=True)
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
                    value=["assets/example.pdf"],
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
