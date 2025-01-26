# RAG App

## Overview

🚀📄✨ The **RAG App** (Retrieval-Augmented Generation) is a simple application that leverages the `sentence-transformers/all-MiniLM-l6-v2` model to provide efficient text-based retrieval and augmentation. The app processes queries and retrieves relevant information from pre-embedded data. 🚀📄✨

## Project Structure

```
RAG-App/
│-- app2.py              # Main application script
│-- embedding.pkl       # Precomputed embeddings for retrieval
│-- requirements.txt    # List of dependencies
│-- README.md           # Project documentation
```

## Features

⚡🔍 - Uses the **`sentence-transformers/all-MiniLM-l6-v2`** model for embedding generation.

- Provides efficient query retrieval based on precomputed embeddings.
- Simple and easy-to-use Python-based implementation.

## Installation

🖥️⚙️1. **Clone the repository:** 

```bash
git clone https://github.com/yourusername/rag-app.git
cd rag-app
```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the sentence-transformer model:**\
   The model `sentence-transformers/all-MiniLM-l6-v2` will be automatically downloaded when running the app for the first time. 

## Usage

📝⚡ 1. **Run the application:**

```bash
python app.py
```

2. **Modify the script:**
   - Update `app2.py` to customize the query input and retrieval logic.
   - Ensure `embedding.pkl` contains the necessary precomputed embeddings. 

## Dependencies

Ensure the following dependencies are installed (listed in `requirements.txt`):

```
sentence-transformers
numpy
pickle
flask
```



## Model Information

The app uses the model, which is a lightweight and efficient transformer model optimized for sentence embeddings. 🤖

