import os
import urllib.request

def download_dataset():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    dest_path = os.path.join(raw_dir, "insurance.csv")
    print(f"Downloading dataset from {url} to {dest_path}...")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Dataset downloaded successfully!")
        print(f"File size: {os.path.getsize(dest_path)} bytes")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()
