import validators

def is_valid_news_url(url):
    # Cek apakah URL valid
    if not validators.url(url):
        return False
    # Cek apakah URL berasal dari CNN Indonesia atau CNBC Indonesia
    return "cnnindonesia.com" in url or "cnbcindonesia.com" in url