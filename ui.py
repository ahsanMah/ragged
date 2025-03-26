import asyncio
import json
import os
import pdb
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import gradio as gr
import numpy as np

from ragged.documents import DocumentDB
from ragged.embedding import Embedder
from ragged.models import model_manager
from ragged.parsers import Parser


def llm_prompt_template(query: str, context: str) -> str:
    common_instruction = f"""Using the following contextual information, respond to the query.
    Recall that the context should be taken as a **primary** source.
    You may use your knowledge to only adapt the context to answer the question.
    Only answer the query. Keep your response brief. DO NOT ANSWER MORE THAN ASKED.
    When referencing the context, use the following format: [<source-id>].
    <context>
    {context}
    </context>
    """

    system_message_flavored = f"""<|system|>{common_instruction}<|end|>
    <|user|>Query:{query}<|end|><|assistant|>
    """

    return system_message_flavored


def process_documents(
    documents: List[gr.File], project_files: Union[str, Path] = None
) -> str:
    """
    This function processes uploaded documents:
    1. Extracts text from documents in chunks
    2. Creates embeddings
    3. TODO: Stores in vector database

    Args:
        documents: List of uploaded files through Gradio interface
        project_files: Directory to store processed documents

    Returns:
        str: Status message indicating number of processed files and embeddings
    """

    parser = Parser(chunksize=256)
    embedder = Embedder()

    if project_files is not None:
        documents = [(DocumentDB.get_name_from_path(path), path) for path in project_files]
        project_dir = os.path.dirname(project_files[0])
        project_name = os.path.basename(project_dir)
    else:
        project_dir = Path("/tmp/ragged")
        project_name = "ragged"
        os.makedirs(project_dir, exist_ok=True)
        documents = [(fileobj.name, fileobj) for fileobj in documents]

    processed_files: List[str] = []
    embeddings: List[np.ndarray] = []
    doc = DocumentDB(project_name, project_dir)
    for name, path in documents:
        print(f"Processing {name}")
        processed_files.append(name)
        # unzip tuples and zip into separate arrays
        text_chunks, metadata = zip(*list(parser.parse(path)))
        embeddings = np.asarray(
            embedder.embed(text_chunks, batch_size=min(len(text_chunks), 32))
        )

        doc.store(text_chunks, embeddings=embeddings, metadata=metadata)

    del parser, doc

    return f"Processed {len(processed_files)} documents\nExtracted: {len(embeddings)} embeddings"


def generate_response(
    query: str,
    history: List[Dict[str, str]],
    reference_documents: List[gr.File],
    project_files: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """
    Generate streaming responses from the LLM based on document context.

    Args:
        query: User's question
        history: Chat history containing previous interactions
        reference_documents: List of uploaded documents to search through
        project_dir: Optional directory where documents are stored
        context_box: Gradio Textbox to display retrieved context

    Yields:
        str: Generated response stream
    """

    if not (reference_documents or project_files):
        yield "Please upload documents first!"
        return

    project_name = "ragged"
    project_dir = "/tmp/ragged"
    if project_files is not None:
        # pdb.set_trace()
        project_dir = os.path.dirname(project_files[0])
        project_name = os.path.basename(project_dir)
        # reference_documents = [DocumentDB.get_name_from_path(p) for p in project_files]

    # Load embeddings
    doc = DocumentDB(project_name, project_dir)
    embeddings: List[np.ndarray] = []
    text_chunks: List[str] = []
    metadata: List[str] = []
    embeddings, text_chunks, metadata = doc.load_all()


    embeddings_arr_combined = embeddings #np.concatenate(embeddings_arr)
    print(">>>> Embeddings combined:", embeddings_arr_combined.shape)

    embedder = Embedder()

    response: List[str] = []
    context_data: List[gr.DataFrame] = []
    retrieved_docs: List[str] = []

    # First response
    if not history:
        start_response = f"Based on the documents you've provided, I have extracted {len(embeddings_arr_combined)} snippets of information.\n\n"
        response.append(start_response)

    # Get the nearest neighbors
    scores, knn_indices = embedder.k_nearest_neighbors(embeddings_arr_combined, query)

    # Get the text and metadata for the nearest neighbors
    for i, idx in enumerate(knn_indices):
        retrieved_docs.append(f"source-{i}: {text_chunks[idx]}")

        # Create properly formatted data for the DataFrame
        context_data.append(
            [i, text_chunks[idx], metadata[idx].filename, f"{1-scores[i]:.2f}"]
        )

    retrieved_docs = "\n".join(retrieved_docs)
    context_data = gr.DataFrame(context_data)

    yield "".join(response), context_data

    if len(history) > 0:
        chat_history: List[str] = []
        for message in history:
            role_token = "<|user|>" if message["role"] == "user" else "<|assistant|>"
            chat_history.append(f"""{role_token}{message["content"]}</s>""")
        prompt_history = "".join(chat_history)
    else:
        prompt_history = ""

    templated_query = llm_prompt_template(query, retrieved_docs)
    prompt = "".join([prompt_history, templated_query])

    llm = model_manager.get_llm()
    output = llm.create_completion(
        prompt,
        max_tokens=4096,
        echo=False,
        stream=True,
        stop=["<|User|>", "<|Assistant|>", "<|assistant|>", "<|user|>"],
    )

    print(">>>> Prompt:")
    print(prompt)
    print(">>>> Streaming output:")

    # Iterate over the output and print it.
    for token in output:
        text = token["choices"][0]["text"]
        print(text, end="", flush=True)
        response.append(text)
        yield "".join(response), context_data


# Define the Gradio interface
def create_rag_interface() -> gr.Blocks:
    """
    Creates the Gradio interface for the RAG application

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft(text_size="sm"), fill_width=True) as demo:
        gr.Markdown("# LLM RAG Application")
        gr.Markdown("Upload documents and ask questions based on their content")

        # Status indicator for model state
        model_status = gr.Markdown("⏳ Loading models...")

        with gr.Row():
            # Left panel for document upload
            with gr.Column(scale=3):
                gr.Markdown("### Document Upload")
                file_output = gr.Textbox(label="Processing Status")
                uploaded_files = gr.File(
                    file_count="multiple",
                    label="Upload Documents",
                    file_types=[".pdf", ".txt", ".docx"],
                    value=["assets/example.pdf"],
                    interactive=False,  # Start disabled
                )

                with gr.Accordion("(Optional) Select Project Directory", open=False):
                    project_dir = gr.FileExplorer(
                        glob="**/*.[ptc][dxs][ftv]",  # match pdf, txt, csv
                        ignore_glob=".**/*",
                        interactive=True,
                    )
                upload_button = gr.Button(
                    "Process Documents", interactive=True
                )  # Start disabled

                with gr.Accordion("View Source Snippets", open=False):
                    context_display = gr.Dataframe(
                        headers=["ID", "Content", "Source", "Score"],
                        label="Retrieved Document Snippets",
                        interactive=False,
                        wrap=True,
                        column_widths=["10%", "60%", "15%", "15%"],
                        max_chars=128,
                    )

            # Right panel for Q&A and context
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions")

                chat = gr.ChatInterface(
                    fn=generate_response,
                    type="messages",
                    stop_btn=True,
                    additional_inputs=[uploaded_files, project_dir],
                    additional_outputs=[context_display],
                    description="Ask something about the uploaded documents...",
                    # interactive=False,  # Start disabled
                )

        async def load_models():
            """Background task to load models and enable UI"""
            try:
                start_time = time.time()
                await model_manager.initialize()
                end_time = time.time()

                # Enable UI components
                return {
                    model_status: "✓ Models loaded in {:.1f}s".format(
                        end_time - start_time
                    ),
                    uploaded_files: gr.update(interactive=True),
                    upload_button: gr.update(interactive=True),
                }
            except Exception as e:
                return {
                    model_status: f"❌ Error loading models: {str(e)}",
                }

        # Set up callbacks
        upload_button.click(
            fn=process_documents, inputs=[uploaded_files, project_dir], outputs=file_output
        )

        # Start loading models in background when UI loads
        demo.load(
            fn=load_models,
            outputs=[
                model_status,
                uploaded_files,
                upload_button,
            ],
        )

    return demo


def main():
    demo = create_rag_interface()

    try:
        demo.launch(share=False)
    finally:
        model_manager.close()
        os.removedirs("/tmp/ragged")


if __name__ == "__main__":
    main()
