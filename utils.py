import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

def load_dataset():
    """Load dataset dari file Excel dan ambil konten artikel."""
    dataset_path = 'E:\\Program Skripsi Nlp\\data\\Data Link Berita.xlsx'
    try:
        df = pd.read_excel(dataset_path)
        articles = []

        for _, row in df.iterrows():
            link = row['Url']
            title = row['Judul']
            category = row['Kategori']
            article_content = scrape_article_content(link)
            articles.append({'Title': title, 'Category': category, 'Content': article_content})

        dataset = pd.DataFrame(articles)
        dataset['Text'] = dataset['Content']
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def scrape_article_content(link):
    """Ambil konten artikel dari link yang diberikan."""
    try:
        response = requests.get(link)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])
    except Exception as e:
        print(f"Error scraping {link}: {e}")
        return ""

def scrape_news_content(link):
    """Alias untuk fungsi scrape_article_content agar lebih jelas dalam konteks aplikasi."""
    return scrape_article_content(link)

def is_valid_news_url(url):
    """Validasi URL dari CNN atau CNBC Indonesia."""
    return re.match(r'https?://(www\.)?(cnnindonesia\.com|cnbcindonesia\.com)', url) is not None

import re

def sanitize_filename(url):
    """Hapus karakter tidak valid dari nama file, termasuk karakter non-ASCII."""
    # Mengambil bagian terakhir dari URL untuk digunakan sebagai nama file
    filename = url.split('/')[-1]  # Ambil bagian setelah '/'
    # Hapus parameter query jika ada (bagian setelah '?')
    filename = filename.split('?')[0]
    # Ganti karakter yang tidak valid dengan '_'
    # Menghapus karakter yang diinginkan, termasuk karakter tidak valid dan non-ASCII
    sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', filename)  # Menghapus karakter tidak valid
    sanitized_name = re.sub(r'[^\x00-\x7F]', '', sanitized_name)  # Menghapus karakter non-ASCII
    return sanitized_name