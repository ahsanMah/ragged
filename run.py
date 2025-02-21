import os
import pdb

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from config import Config
from ragged.embedding import Embedder
from ragged.parsers import Parser

embedder = Embedder(device="cpu")

docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

parser = Parser(chunksize=128)
text = parser.parse_pdf("assets/example.pdf")
for i, chunk in enumerate(text):
    docs.append(chunk)

query_prompt_name = "s2p_query"
query = "What follows the phrase '|  Yang  |'?"
embeddings = embedder.embed(docs)
knn_indices = embedder.k_nearest_neighbors(embeddings, query)

# models_dir = os.environ.get("LLAMA_MODELS_DIR", "./models")


# llm = Llama(
#     model_path=os.path.join(models_dir, "7B/llama-model.gguf",
# n_gpu_layers=-1, # Uncomment to use GPU acc`eleration
# seed=1337, # Uncomment to set a specific seed
# n_ctx=2048, # Uncomment to increase the context window
# )

retreived_docs = ", ".join([f"{i}: {docs[i]}" for i in knn_indices])

prompt = f"""Query: {query}
Using the following contextual information, respond to the query.
Recall that the context should be taken as a **primary** source.
You may use your knowledge to only adapt the context to answer the question.
<context_start>
{retreived_docs}
<context_end>
Response: 
"""

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
#     filename="*q8_0.gguf",
#     verbose=True,
# )

# llm = Llama.from_pretrained(
#     repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
#     filename="*Q6_K_L.gguf",
#     verbose=True,
# )

llm = Llama(
    model_path=os.path.join(
        Config.MODEL_DIR, "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    ),
    n_gpu_layers=46,  # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=4096, # Uncomment to increase the context window
    verbose=True,
)
# pdb.set_trace()
output = llm.create_completion(
    prompt,  # Prompt
    max_tokens=None,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    echo=True,  # Echo the prompt back in the output
    stream=True
)  # Generate a completion, can also call create_completion

print(prompt)
print("Streaming output:")

# Iterate over the output and print it.
for response in output:
    print(response["choices"][0]["text"], end="", flush=True)

print(response)


system_message = """
Using the following contextual information, respond to the query.
Recall that the context should be taken as a **primary** source.
You may use your knowledge to only adapt the context to answer the question.
<context_start>
{retreived_docs}
<context_end>
"""

output = llm.create_chat_completion(
    messages=[
        { "role": "system", "content": system_message },
        { "role": "user", "content": query },
    ]
    # echo=True,  # Echo the prompt back in the output
    # stream=True,
)  #
print(output)

llm.close()


# import os

# import torch
# from sklearn.preprocessing import normalize
# from transformers import AutoModel, AutoTokenizer

# query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
# queries = [
#     "What are some ways to reduce stress?",
#     "What are the benefits of drinking green tea?",
# ]
# queries = [query_prompt + query for query in queries]
# # docs do not need any prompts
# docs = [
#     "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
#     "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
# ]

# # The path of your model after cloning it
# model_dir = Config.EMBEDDING_MODEL_DIR
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vector_dim = 1024
# vector_linear_directory = f"2_Dense_{vector_dim}"
# model = AutoModel.from_pretrained(
#     model_dir,
#     trust_remote_code=True,
#     use_memory_efficient_attention=False,
#     unpad_inputs=False,
# ).eval()
# # you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
# # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,use_memory_efficient_attention=False,unpad_inputs=False).to(device).eval()
# tokenizer = AutoTokenizer.from_pretrained(
#     model_dir,
#     trust_remote_code=True,
# )
# vector_linear = torch.nn.Linear(
#     in_features=model.config.hidden_size, out_features=vector_dim, device=device
# )
# vector_linear_dict = {
#     k.replace("linear.", ""): v
#     for k, v in torch.load(
#         os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin"),
#         map_location=device,
#     ).items()
# }
# vector_linear.load_state_dict(vector_linear_dict)
# vector_linear.to(device)

# # Time the embedding process
# import time

# start = time.time()

# # Embed the queries
# with torch.no_grad():
#     input_data = tokenizer(
#         queries, padding="longest", truncation=True, max_length=512, return_tensors="pt"
#     )
#     input_data = {k: v.to(device) for k, v in input_data.items()}
#     attention_mask = input_data["attention_mask"]
#     last_hidden_state = model(**input_data)[0]
#     last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#     query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())

# stop = time.time()
# print(f"Time to embed queries: {stop - start:.2f} seconds")
# start = time.time()

# # Embed the documents
# with torch.no_grad():
#     input_data = tokenizer(
#         docs, padding="longest", truncation=True, max_length=512, return_tensors="pt"
#     )
#     input_data = {k: v.to(device) for k, v in input_data.items()}
#     attention_mask = input_data["attention_mask"]
#     last_hidden_state = model(**input_data)[0]
#     last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#     docs_vectors = normalize(vector_linear(docs_vectors).cpu().numpy())
# stop = time.time()
# print(f"Time to embed documents: {stop - start:.2f} seconds")

# print(query_vectors.shape, docs_vectors.shape)
# # (2, 1024) (2, 1024)

# similarities = query_vectors @ docs_vectors.T
# print(similarities)
# # [[0.8397531  0.29900077]
# #  [0.32818374 0.80954516]]
