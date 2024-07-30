from flask import Flask, request, render_template, send_file
from data_processor import process_word_file
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import faiss
from datasets import load_from_disk

app = Flask(__name__)

# Path ke dataset dan FAISS index
dataset_path = "E:/Program Skripsi Nlp/dataset"
index_path = "E:/Program Skripsi Nlp/faiss/index.faiss"

# Memuat dataset dari disk
try:
    dataset = load_from_disk(dataset_path)
    print("Dataset berhasil dimuat.")
except Exception as e:
    print(f"Terjadi kesalahan saat memuat dataset: {e}")

# Memuat FAISS index dari disk
try:
    faiss_index = faiss.read_index(index_path)
    print("FAISS index berhasil dimuat.")
except Exception as e:
    print(f"Terjadi kesalahan saat memuat FAISS index: {e}")

# Inisialisasi RAG
rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
rag_retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name="custom", passages_path=dataset_path, index_path=index_path)
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.docx'):
            try:
                output_file = process_word_file(file)
                return send_file(output_file, as_attachment=True, download_name='summary_output.docx')
            except Exception as e:
                return str(e)  # Menampilkan kesalahan jika terjadi
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)