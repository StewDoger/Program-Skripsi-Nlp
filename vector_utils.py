import os
import re
import numpy as np
import faiss
import torch
import pickle
from transformers import AutoTokenizer, RagTokenForGeneration, DPRContextEncoder, DPRContextEncoderTokenizer, RagConfig
from difflib import SequenceMatcher

# Inisialisasi FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Inisialisasi tokenizer dan model DPR untuk chunking
dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
dpr_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", ignore_mismatched_sizes=True).to('cpu')

# Inisialisasi tokenizer dan model RAG dengan konfigurasi
rag_config = RagConfig.from_pretrained("facebook/rag-token-nq")
rag_config.forced_bos_token_id = 0
rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq", clean_up_tokenization_spaces=False)
rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", config=rag_config).to('cpu')

def embed_text_dpr(text):
    """Menghasilkan embedding untuk teks menggunakan model DPR."""
    inputs = dpr_tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to('cpu')
    outputs = dpr_model(**inputs)
    return outputs.pooler_output.detach().numpy().astype(np.float32)

def chunk_text_dpr(text, max_length=512):
    """Memecah teks panjang menjadi beberapa potongan berdasarkan kalimat menggunakan DPR tokenizer."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Memecah berdasarkan kalimat
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(dpr_tokenizer.tokenize(current_chunk + sentence)) < max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:  # Jika sudah ada isi, simpan chunk saat ini
                chunks.append(current_chunk.strip())
                current_chunk = sentence  # Memulai chunk baru dengan kalimat ini

    # Menyimpan sisa chunk yang mungkin masih ada
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def embed_text_rag(contexts):
    """Menghasilkan embedding untuk konteks menggunakan model RAG."""
    inputs = rag_tokenizer(contexts, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cpu')
    outputs = rag_model.question_encoder(**inputs)
    return outputs[0].detach().numpy().astype(np.float32)

def process_document(doc):
    """Memproses satu dokumen dan menghasilkan embedding setiap chunk dengan DPR."""
    doc_embeddings = []
    doc_texts = []
    chunks = chunk_text_dpr(doc)

    for chunk in chunks:
        embedding = embed_text_dpr(chunk)
        doc_embeddings.append(embedding)
        doc_texts.append(chunk)

    return doc_texts, doc_embeddings

def prepare_vector_database(dataset, cache_path='embedding_cache.pkl', batch_size=1000):
    """Menyiapkan basis data vektor dengan batch kecil dan cache."""
    # Cek jika cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                all_texts, all_embeddings = pickle.load(f)
                index.add(np.vstack(all_embeddings))
                return all_texts
        except (pickle.UnpicklingError, EOFError, IOError) as e:
            print(f"Cache file corrupted: {e}. Deleting the cache file.")
            os.remove(cache_path)  # Hapus file cache yang rusak

    all_texts = []
    all_embeddings = []

    num_batches = (len(dataset['Text']) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = dataset['Text'][i * batch_size:(i + 1) * batch_size]

        for doc in batch:
            doc_texts, doc_embeddings = process_document(doc)
            all_texts.extend(doc_texts)
            all_embeddings.extend(doc_embeddings)

    embeddings_array = np.vstack(all_embeddings)
    index.add(embeddings_array)

    with open(cache_path, 'wb') as f:
        pickle.dump((all_texts, all_embeddings), f)

    return all_texts

def find_most_relevant_text(query_text, dataset_texts, top_k=5, similarity_threshold=0.8):
    """Menemukan teks yang paling relevan dalam database dengan memfilter kemiripan."""
    contexts = [query_text]
    query_embedding = embed_text_rag(contexts).astype(np.float32)

    # Ambil hasil top_k
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    retrieved_texts = [dataset_texts[idx] for idx in I[0] if 0 <= idx < len(dataset_texts)]

    # Filter kemiripan pada hasil
    filtered_texts = []
    for text in retrieved_texts:
        is_similar = any(
            SequenceMatcher(None, text, existing_text).ratio() > similarity_threshold
            for existing_text in filtered_texts
        )
        if not is_similar:
            filtered_texts.append(text)

    return filtered_texts