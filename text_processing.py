def clean_extracted_text_from_url(uploaded_text):
    """Bersihkan teks yang diambil dari URL."""
    unwanted_texts = [
        r"ADVERTISEMENT",
        r"SCROLL TO CONTINUE WITH CONTENT",
        r"[Gambas:Video CNN]",
        r"Ikuti kami di",
        r"Follow us on",
        r"Copyright",
        r"Terms of Use",
        r"Privacy Policy",
        r"Untuk informasi lebih lanjut",
        r"Klik di sini",
        r"Untuk berita selengkapnya",
        r"Read more",
        r"View this post on Instagram",
        r"Share this article",
        r"Jakarta, CNBC Indonesia",
        r"Jakarta, CNN Indonesia"
    ]
    cleaned_text = uploaded_text
    for unwanted in unwanted_texts:
        cleaned_text = cleaned_text.replace(unwanted, '')
    return cleaned_text.strip()