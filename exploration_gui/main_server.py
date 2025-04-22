from flask import Flask, jsonify, render_template, send_from_directory
from glob import glob
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Serve the index.html at the root URL
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_image(filename):
    return send_from_directory("images", filename)

@app.route('/<path:filename>')
def serve_audio(filename):
    return send_from_directory('audio_clips', filename)

# Fetch image and audio data
@app.route('/get_data', methods=['GET'])
def get_data():
    EXPECTED_OUTER_COLUMNS = ['x', 'y', 'artists', 'track_name', 'popularity', 'danceability', 'energy', 'loudness', 'valence', 'track_genre']
    EXPECTED_INNER_COLUMNS = ["prompt", "track_genre", "audio_path", "mean_color"]
    paths = glob("./static/jsons/*")
    paths = [p.replace("\\", "/") for p in paths]

    json_files_content = []
    unique_genres = set()
    for path in paths:
        with open(path, 'r') as f:
            data_block = json.load(f)
        assert all(col in data_block for col in EXPECTED_OUTER_COLUMNS), "Column mismatch"

        for row in data_block["data"]:
            assert all(col in row for col in EXPECTED_INNER_COLUMNS), "Column mismatch"
            assert os.path.exists(row["image_path"])
            assert os.path.exists(row["audio_path"])
            try:
                for color in "rgb":
                    assert 0 <= row["mean_color"][color] <= 255
            except Exception:
                raise ValueError(f"`{row['mean_color']=}` is not valid. Expected something like 'r': 79, 'g': 23, 'b': 19" )

        json_files_content.append(data_block)
        unique_genres.add(data_block["track_genre"])
    unique_genres = list(unique_genres)

    print(unique_genres)
    return jsonify({"data":json_files_content, "genres":unique_genres})

if __name__ == '__main__':
    app.run(debug=False)
