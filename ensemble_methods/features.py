"""
features.py
-----------

Simple URL feature extraction used by the classification models.
"""

def extract_features(url):
    return [
        len(url) / 100,
        url.count('.') / 10,
        url.count('/') / 10,
        url.count('@'),
        1 if "https" in url else 0,
        1 if "-" in url else 0
    ]