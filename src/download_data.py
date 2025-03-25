import os
import gdown

def download_data():    
    file_id = "1_oqcPdcdM2Rf0F5VasNgxqmZBP-c50Hd"
    output_path = "data/people_wiki.csv"

    os.makedirs("data", exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading dataset from Google Drive...")
    gdown.download(url, output_path, quiet=False)

    print(f"Dataset downloaded successfully and saved to {output_path}")
