from rouge_score import rouge_scorer
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import os
from datetime import datetime
from text_processing import clean_extracted_text_from_url
from summarization import generate_summary
from utils import load_dataset, scrape_news_content, is_valid_news_url, sanitize_filename
from vector_utils import prepare_vector_database, find_most_relevant_text, embed_text_rag  # Import fungsi embed_text_rag
import time
import logging
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SUMMARIZED_FOLDER'] = 'summaries'
app.config['LOG_FOLDER'] = 'logs'

os.makedirs(app.config['SUMMARIZED_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

# Load dataset dan siapkan basis data vektor
dataset = load_dataset()
all_chunk_texts = prepare_vector_database(dataset)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/ringkas', methods=['POST'])
def ringkas():
    url = request.form.get('url')

    # Cek apakah input berupa URL yang valid
    if not url or not is_valid_news_url(url):
        flash("URL tidak valid. Pastikan URL berasal dari CNN Indonesia atau CNBC Indonesia.", "error")
        return redirect(url_for('home'))

    try:
        start_time = time.time()

        # Scraping konten berita dari URL yang valid
        berita_content = scrape_news_content(url)
        cleaned_text = clean_extracted_text_from_url(berita_content)

        # Temukan teks relevan dari dataset
        relevant_data = find_most_relevant_text(cleaned_text, all_chunk_texts)
        final_summary = generate_summary(cleaned_text, relevant_data)

       # Hitung kemiripan (contoh menggunakan cosine similarity)
        cleaned_embedding = embed_text_rag([cleaned_text])
        summary_embedding = embed_text_rag([final_summary])

        # Ambil skor kemiripan dari array numpy dan ubah ke float
        similarity_score = np.dot(cleaned_embedding, summary_embedding.T) / (np.linalg.norm(cleaned_embedding) * np.linalg.norm(summary_embedding))
        similarity_score_value = float(similarity_score)  # Ubah menjadi float

        # Akhiri pengukuran waktu
        end_time = time.time()
        processing_time = end_time - start_time
        logging.info("Waktu proses ringkasan: {:.2f} detik".format(processing_time))

        # Evaluasi menggunakan ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(cleaned_text, final_summary)

        # Log hasil evaluasi ROUGE dan kemiripan
        log_filename = os.path.join(app.config['LOG_FOLDER'], 'processing_log.txt')
        with open(log_filename, 'a') as log_file:
            log_file.write("URL Berita: {}\n".format(url))
            log_file.write("Waktu Proses: {:.2f} detik\n".format(processing_time))
            log_file.write("Timestamp: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            log_file.write("ROUGE-1: {}\n".format(rouge_scores['rouge1']))
            log_file.write("ROUGE-2: {}\n".format(rouge_scores['rouge2']))
            log_file.write("ROUGE-L: {}\n".format(rouge_scores['rougeL']))
            log_file.write("Kemiripan: {:.4f}\n".format(similarity_score_value))  # Gunakan similarity_score_value
            log_file.write("-" * 40 + "\n")

        # Sanitasi URL untuk dijadikan nama file
        sanitized_url = sanitize_filename(url)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = "summary_{}_{}.txt".format(timestamp, sanitized_url)
        output_path = os.path.join(app.config['SUMMARIZED_FOLDER'], output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Isi Berita:\n" + cleaned_text + "\n\n")
            f.write("Ringkasan:\n" + final_summary)

        return render_template('result.html', filename=output_filename)

    except Exception as e:
        flash("Terjadi kesalahan saat memproses URL.", "error")
        print(e)
        return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['SUMMARIZED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
