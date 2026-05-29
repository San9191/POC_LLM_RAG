# Intro.ipynb Documentation

This file documents the `RAG/Intro.ipynb` notebook, explaining every code block, every library used, and the most important lines to understand how the notebook builds a simple Retrieval-Augmented Generation (RAG) demo.

---

## Overview

`Intro.ipynb` is organized as a multi-day introduction to RAG. It contains:

- Week 5 Day 1: a simple knowledge-base-driven chat app using OpenAI and Gradio.
- Week 5 Day 2: a vectorization workflow using LangChain, Hugging Face embeddings, Chroma vector store, and visualization with t-SNE and Plotly.
- Week 5 Days 3-5: placeholders for later RAG work.

The notebook uses both basic Python utilities and higher-level tools to load, index, and search content.

---

## Tools and Libraries Used

### Standard Python and utility libraries

- `os`: access file paths and environment variables.
- `glob`: search the file system with wildcard patterns.
- `pathlib.Path`: manage file names and paths in a cross-platform way.
- `dotenv.load_dotenv`: load environment variables from a `.env` file.
- `print`: display results and debugging output.

### OpenAI / LLM tools

- `openai.OpenAI`: the OpenAI Python client to call the chat completion API.
- `MODEL = "gpt-4.1-nano"`: selects the OpenAI model used for generation.

### Web UI tool

- `gradio as gr`: used to launch a simple interactive chat interface from the notebook.

### RAG and embeddings tools

- `tiktoken`: tokenization for counting tokens for a model.
- `numpy as np`: numeric arrays for vector handling.
- `langchain_openai.OpenAIEmbeddings`: optional OpenAI embeddings interface (imported but not used in the active code).
- `langchain_chroma.Chroma`: the Chroma vector store implementation.
- `langchain_huggingface.HuggingFaceEmbeddings`: transformer-based embeddings from Hugging Face.
- `langchain_community.document_loaders.DirectoryLoader`: load documents recursively from folders.
- `langchain_community.document_loaders.TextLoader`: read plain text / markdown files.
- `langchain_text_splitters.RecursiveCharacterTextSplitter`: split long documents into smaller chunks for embedding.
- `sklearn.manifold.TSNE`: dimensionality reduction for visualization.
- `plotly.graph_objects as go`: interactive charts to visualize vector embeddings.

---

## Day 1: Simple RAG Chat Interface

### Imports and model setup

```python
import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI
```

- `glob` and `Path` are used to find and normalize files in the knowledge base.
- `load_dotenv` is imported so the notebook can optionally read API keys from a `.env` file.
- `gradio` is used later to build a chat UI.
- `OpenAI` is the OpenAI client that makes requests to the model.

```python
MODEL = "gpt-4.1-nano"
openai = OpenAI()
```

- `MODEL` defines the target LLM.
- `openai = OpenAI()` creates a client instance to call the OpenAI API.

### Building the knowledge dictionary

```python
knowledge = {}

filenames = glob.glob("knowledge-base/employees/*")

for filename in filenames:
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()
```

- `knowledge = {}` creates an empty dictionary to store text by key.
- `glob.glob("knowledge-base/employees/*")` finds every file in the `employees` folder.
- `Path(filename).stem` strips the `.md` extension from the file name.
- `.split(' ')[-1]` takes the last word from the file stem and uses it as the dictionary key.
- `knowledge[name.lower()] = f.read()` stores the full file content under a lowercase employee identifier.

This block prepares the first part of the knowledge base from employee markdown files.

### Inspecting a loaded item

```python
knowledge["lancaster"]
```

- This line shows the entry stored for `lancaster`.
- It confirms that the `knowledge` dictionary contains parsed employee data.

### Loading product knowledge

```python
filenames = glob.glob("knowledge-base/products/*")

for filename in filenames:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()
```

- This block repeats the process for product documents.
- `Path(filename).stem` is used directly because product files do not require the same last-word extraction rule.
- Each product document is stored in `knowledge` keyed by the filename.

### Confirming loaded keys

```python
knowledge.keys()
```

- Displays the dictionary keys and verifies what content is available for retrieval.

### System prompt prefix for RAG

```python
SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""
```

- This multi-line string defines the system role and the RAG prompt structure.
- It tells the model to treat the business as `Insurellm` and to use extra context when answering.
- The final line `Relevant context:` is important because the code appends retrieved document text directly below it.

### Simple context retrieval function

```python
def get_relevant_context_simple(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    print(text)
    words = text.lower().split()
    print(words)
    relevant_context = []
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    return relevant_context
```

- `message` is the user query.
- `text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())` removes punctuation and non-letter characters.
- `words = text.lower().split()` tokenizes the cleaned text.
- The loop checks each word against the `knowledge` dictionary.
- Any matching knowledge items are appended to `relevant_context`.

This is a very simple keyword-based retrieval function used for proof of concept.

### Example retrieval

```python
get_relevant_context_simple("How old is Mr Lancaster")
```

- This call exercises the retrieval function and demonstrates that the notebook can find employee context.

### Cleaner retrieval function

```python
def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    print(text)
    words = text.lower().split()
    print(words)
    return [knowledge[word] for word in words if word in knowledge]
```

- This version is shorter and returns a list comprehension.
- It still cleans the message and matches words against the knowledge dictionary.
- It is functionally equivalent to the earlier version but more concise.

### Printing retrieved context

```python
texto = get_relevant_context("Who is lancaster?")
print(len(texto))
for item in texto:
  print(item)
```

- These lines show the returned context and prove that retrieval works.
- `print(len(texto))` shows how many matching documents were found.

### Combined query retrieval

```python
get_relevant_context("Who is Lancaster and what is carllm?")
```

- This call tests retrieval for multiple matching keywords in one query.

### Formatting additional context

```python
def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result
```

- `additional_context` wraps the retrieved documents into a single human-readable string.
- If no context is found, it returns a fallback message.
- `"\n\n".join(relevant_context)` preserves separation between multiple documents.

### Example additional context output

```python
print(additional_context("Who is Alex Lancaster?"))
```

- Shows how the appended context is formatted and what will be passed to the LLM.

### System prompt alias and chat function

```python
system_prefix="""
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""
```

- This is the same prompt as `SYSTEM_PREFIX`, but stored in a second variable.
- It is used in the `chat` function for the final message composition.

```python
def chat(message, history):
    system_message = system_prefix + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content
```

- This is the key RAG integration point.
- `system_message` embeds the retrieved knowledge into the system prompt.
- `messages` builds the chat conversation with system, history, and user roles.
- `openai.chat.completions.create(...)` calls the model.
- `response.choices[0].message.content` returns the generated answer.

### Launching the Gradio chat interface

```python
gr.ChatInterface(fn=chat).launch()
```

- This creates a Gradio app from the `chat` function.
- It launches an interactive chat UI directly from the notebook.
- Important: this is how the notebook turns the example into a runnable demo.

---

## Day 2: Vector Embeddings and Visualization

### Imports for embeddings and visualization

```python
import os
import glob
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
import plotly.graph_objects as go
```

- `tiktoken` is used to count tokens for the selected model.
- `numpy` handles vector arrays.
- `OpenAIEmbeddings` is imported for optional OpenAI embedding use.
- `Chroma` stores vector embeddings on disk.
- `HuggingFaceEmbeddings` gets embeddings from a transformer model.
- `DirectoryLoader` and `TextLoader` read markdown documents from folders.
- `RecursiveCharacterTextSplitter` divides long texts into manageable chunks.
- `TSNE` reduces embedding dimensions to 2D or 3D for plotting.
- `plotly.graph_objects` creates interactive visualizations.

### Model and environment setup

```python
MODEL = "gpt-4.1-nano"
db_name = "vector_db"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
```

- `MODEL` is reused for consistency.
- `db_name` sets the folder name for the Chroma vector store persistence.
- `load_dotenv(override=True)` loads environment variables if a `.env` file exists.
- `os.getenv('OPENAI_API_KEY')` reads the API key.
- The `print` statement verifies that the key is available.

### Counting characters in the knowledge base

```python
knowledge_base_path = "knowledge-base/**/*.md"
files = glob.glob(knowledge_base_path, recursive=True)
print(f"Found {len(files)} files in the knowledge base")

entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        entire_knowledge_base += f.read()
        entire_knowledge_base += "\n\n"

print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")
```

- Searches every markdown file under `knowledge-base` recursively.
- Reads all text into one string.
- Reports total character count to show corpus size.

### Counting tokens for the model

```python
encoding = tiktoken.encoding_for_model(MODEL)
tokens = encoding.encode(entire_knowledge_base)
token_count = len(tokens)
print(f"Total tokens for {MODEL}: {token_count:,}")
```

- `tiktoken.encoding_for_model(MODEL)` chooses the tokenizer for `gpt-4.1-nano`.
- `encoding.encode(...)` converts the entire knowledge base into tokens.
- `len(tokens)` reveals how many model tokens the data requires.

This is essential in RAG because token budget affects prompt size and retrieval strategy.

### Loading documents with LangChain

```python
folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Loaded {len(documents)} documents")
```

- `glob.glob("knowledge-base/*")` lists first-level subfolders such as `employees`, `products`, `contracts`, etc.
- `DirectoryLoader(..., glob="**/*.md", loader_cls=TextLoader)` loads every markdown file in each folder.
- `doc.metadata["doc_type"] = doc_type` tags each document with its source category.
- `documents.append(doc)` collects all loaded documents in one list.

This block turns raw files into LangChain documents with metadata.

### Inspecting a sample document

```python
documents[1]
```

- Displays the second document object.
- Useful for verifying that loaded documents contain text and metadata.

### Splitting documents into chunks

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Divided into {len(chunks)} chunks")
print(f"First chunk:\n\n{chunks[0]}")
```

- `chunk_size=1000` limits chunks to 1000 characters.
- `chunk_overlap=200` preserves overlap, which improves retrieval accuracy across chunk boundaries.
- `split_documents(documents)` creates a chunked corpus ready for embedding.

### Examining a specific chunk

```python
chunks[100]
```

- Displays the 101st chunk in the chunked corpus.
- Helps verify the splitting logic and inspect stored text content.

### Embedding documents and building the Chroma store

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")
```

- `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` creates an embedding function from a lightweight, local transformer.
- The commented-out `OpenAIEmbeddings` line shows an alternative using OpenAI embeddings.
- `if os.path.exists(db_name): ... delete_collection()` removes any existing Chroma collection to ensure a fresh build.
- `Chroma.from_documents(...)` converts document chunks into vector embeddings and persists them in `vector_db`.
- `vectorstore._collection.count()` prints the number of stored vectors.

### Inspecting vector metadata

```python
collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"]) ["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
```

- `collection = vectorstore._collection` gets direct access to the underlying Chroma collection.
- `collection.get(limit=1, include=["embeddings"])` retrieves one vector embedding.
- `len(sample_embedding)` tells how many embedding dimensions each vector has.
- This block verifies the vector store structure and dimensionality.

---

## Visualization

### Preparing visualization arrays

```python
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
metadatas = result['metadatas']
doc_types = [metadata['doc_type'] for metadata in metadatas]
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]
```

- Retrieves all stored embeddings and document metadata.
- Converts embeddings into a NumPy array for TSNE.
- `doc_types` uses metadata tags to classify each vector.
- `colors` maps each doc type to a color for plotting.

### 2D visualization with t-SNE

```python
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
```

- `TSNE(n_components=2, random_state=42)` reduces vectors into two dimensions for plotting.
- `fit_transform(vectors)` projects the embeddings.
- `go.Scatter(...)` creates a scatter plot where each point is a document chunk.
- The `text` field provides hover metadata so users can inspect document type and content preview.
- `fig.show()` opens the interactive 2D plot.

### 3D visualization with t-SNE

```python
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=10, b=10, l=10, t=40)
)

fig.show()
```

- The notebook repeats the visualization in 3D.
- This gives a more spatial view of vector relationships.
- `Scatter3d` is used to render an interactive 3D scatter plot.

---

## Empty or placeholder cells

The notebook contains several empty code and markdown cells for Week 5 Days 3-5. These are currently placeholders and can be filled later with additional RAG examples, evaluation, or improvements.

---

## Important conceptual notes

- The notebook demonstrates two different retrieval approaches:
  1. keyword-based retrieval from a small dictionary (`knowledge`) for Day 1,
  2. vector-based retrieval preparation for Day 2.
- The Day 1 RAG prompt uses `system_prefix + additional_context(message)` to inject retrieved text into the assistant's system prompt. This is the core RAG pattern.
- The Day 2 workflow is mostly preparation: loading, token-counting, chunk splitting, embedding, storing, and visualizing.
- `load_dotenv(override=True)` and `openai_api_key = os.getenv('OPENAI_API_KEY')` are important when OpenAI services are required.

---

## Recommendations for future work

- Add a retrieval query step for the vector store using `vectorstore.similarity_search(...)` or a LangChain retriever.
- Replace the simple keyword matcher in Day 1 with an embedding-based search for stronger retrieval.
- Add error handling around missing files and missing API keys.
- Ensure `load_dotenv()` is used consistently or remove it if environment variables are handled elsewhere.
- Fill the Day 3-5 placeholders with concrete RAG experiments.
