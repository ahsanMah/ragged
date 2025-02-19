import os
import pdb

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# from src.config import Config
from ragged.parsers import Parser

docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

parser = Parser()
text = parser.parse_pdf("assets/example.pdf")
for i, chunk in enumerate(text):
    docs.append(chunk)

# models_dir = os.environ.get("LLAMA_MODELS_DIR", "./models")


# llm = Llama(
#     model_path=os.path.join(models_dir, "7B/llama-model.gguf",
    # n_gpu_layers=-1, # Uncomment to use GPU acc`eleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
# )



# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`
query_prompt_name = "s2p_query"
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts

# ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
# on gpu
device = "cpu"
# model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).to(device)
# you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
model = SentenceTransformer(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device="cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)
query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
doc_embeddings = model.encode(docs, show_progress_bar=True)
print(model.similarity_fn_name)
print(query_embeddings.shape, doc_embeddings.shape)
# (N, 1024) (N, 1024)

similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)

ridx = similarities.argmax(dim=1)
retreived_docs = [docs[i] for i in ridx]
print(retreived_docs)

prompt = f"""{queries[0]}\n
Using the following information, respond to the query.
<context_start>
{retreived_docs[0]}
<context_end>
Response: 
"""
print(prompt)

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="*q8_0.gguf",
    verbose=True,
)

output = llm(
    prompt,  # Prompt
    max_tokens=128,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    stop=[
        "Q:",
        "\n",
    ],  # Stop generating just before the model would generate a new question
    echo=True,  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion

llm.close()
print(output)
