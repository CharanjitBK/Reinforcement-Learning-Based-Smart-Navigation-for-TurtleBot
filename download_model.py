import os
import requests
import zipfile

# URL of your model zip file
url = "https://drive.google.com/uc?export=download&id=1YF2jDoFsI46XT7UlceDNF-M8WVPrNh8W"

output_path = "model.zip"

# Download the file
print("Downloading model...")
r = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download complete!")

# Optional: unzip
import zipfile
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("model")

print("Model extracted to 'model/' folder")
