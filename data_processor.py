import pandas as pd
import numpy as np
import tempfile
import os
from transformers import RagTokenizer, RagSequenceForGeneration
from docx import Document
from datasets import Dataset, load_from_disk
import faiss

# Fungsi untuk memuat data dari file Excel
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    data = df.to_dict(orient='records')
    return Dataset.from_dict({
        'title': [item.get('title', '') for item in data],
        'text': [item['Text'] for item in data],
        'embeddings': [np.zeros(768).tolist() for _ in data]  # Placeholder untuk embeddings
    })

# Simpan dataset ke disk
def save_dataset(dataset, path):
    dataset.save_to_disk(path)
    print(f"Dataset saved to {path}")

# Muat dataset dari disk
def load_dataset(path):
    dataset = load_from_disk(path)
    print(f"Dataset loaded from {path}")
    return dataset

# Inisialisasi tokenizer dan model RAG
rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq')

# Memuat dataset dari file Excel
data_path = "E:/Program Skripsi Nlp/data/file_berita_dengan_teks.xlsx" 
dataset_path = "E:/Program Skripsi Nlp/dataset"
index_path = "E:/Program Skripsi Nlp/faiss/index.faiss"

# Fungsi untuk membuat embedding dari teks menggunakan RAG
def embed_text(text):
    inputs = rag_tokenizer(text, return_tensors='pt', truncation=True)
    outputs = rag_model.question_encoder(**inputs)
    if isinstance(outputs, tuple):
        last_hidden_state = outputs[0]
    else:
        last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.mean(dim=1).detach().numpy()  # Mengambil mean dari dimensi waktu

# Memuat dataset dan menyimpan ke disk
dataset = load_excel_data(data_path)
dataset.save_to_disk(dataset_path)

# Membuat embedding untuk setiap teks dalam dataset
embeddings = np.array([embed_text(item) for item in dataset['text']])
# Pastikan embeddings adalah 2D
if embeddings.ndim == 3:
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

# Simpan embeddings ke disk menggunakan FAISS
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # Sesuaikan dengan dimensi embedding
faiss_index.add(embeddings.astype(np.float32))
faiss.write_index(faiss_index, index_path)

# Memuat kembali dataset dan index dari disk
dataset = load_from_disk(dataset_path)
faiss_index = faiss.read_index(index_path)

def process_word_file(file):
    # Simpan file yang diunggah ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    # Buka dan proses file DOCX
    doc = Document(temp_file_path)
    text_content = '\n'.join([para.text for para in doc.paragraphs])

    # Buat query atau teks untuk diringkas
    query = text_content
    
    # Pencarian Dokumen menggunakan RAG
    question_encodings = rag_tokenizer(query, return_tensors='pt')
    question_embeddings = rag_model.question_encoder(**question_encodings)
    if isinstance(question_embeddings, tuple):
        question_embeddings_np = question_embeddings[0].detach().numpy().astype(np.float32)
    else:
        question_embeddings_np = question_embeddings.last_hidden_state.detach().numpy().astype(np.float32)
    
    # Pencarian FAISS
    _, indices = faiss_index.search(question_embeddings_np, k=5)
    
    # Gabungkan teks relevan
    relevant_texts = [dataset['text'][i] for i in indices[0]]
    combined_text = " ".join(relevant_texts) + " " + query

    # Hasilkan ringkasan menggunakan model RAG
    inputs = rag_tokenizer(combined_text, return_tensors='pt', truncation=True)
    summary_ids = rag_model.generate(inputs['input_ids'], max_length=200, num_beams=4, early_stopping=True)
    summary_text = rag_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Buat file DOCX untuk ringkasan
    output_path = tempfile.mktemp(suffix='.docx')
    doc_out = Document()
    doc_out.add_paragraph(summary_text)
    doc_out.save(output_path)

    # Hapus file sementara
    os.remove(temp_file_path)

    return output_path
