import requests
import gzip
import shutil
from tqdm import tqdm
from pathlib import Path

URL = "https://huggingface.co/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz"
OUT_GZ = Path("glove.6B.100d.txt.gz")
OUT_TXT = Path("glove.6B.100d.txt")

def download(url, out_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            unit="B", unit_scale=True, unit_divisor=1024, total=total, desc=out_path.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def decompress_gz(gz_path, txt_path):
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    print("Downloading...")
    download(URL, OUT_GZ)
    print("Decompressing...")
    decompress_gz(OUT_GZ, OUT_TXT)
    print(f"Done: {OUT_TXT.resolve()}")