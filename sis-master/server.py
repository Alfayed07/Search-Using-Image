import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:4]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        
        # Mengambil 30 gambar terdekat berdasarkan jarak
        ids = np.argsort(dists)[:4]
        filtered_ids = [id for id in ids if dists[id] <= 1.5]  # Memfilter gambar dengan skor di bawah atau sama dengan 1.23
        filtered_scores = [(dists[id], img_paths[id]) for id in filtered_ids]

        return render_template("index.html", query_path=uploaded_img_path, scores=filtered_scores)

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
