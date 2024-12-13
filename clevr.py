import os
import requests
import zipfile


def download_clevr(download_url):
    path = os.path.basename(url)
    if not os.path.exists(path):
        # Download the dataset
        print(f"Downloading CLEVR dataset from {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()