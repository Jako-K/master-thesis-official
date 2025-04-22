# Dot Explorer – Valence × Arousal Audio‑Image Map

## What this program lets you do

- Hover over any point to view metadata and click to play its audio clip.
- Filter tracks by genre using dynamically generated check‑boxes.
- Toggle between raw images and a colour‑averaged grid view.
- Cycle through multiple images per track with keys **1**, **2** and **3**.
---

## Quick‑start
```bash
pip install flask
python main_server.py        # → http://127.0.0.1:5000
```

## Deploy with Gunicorn + Nginx (example)

```bash
gunicorn --bind 127.0.0.1:8000 main_server:app \
         --daemon --access-logfile gunicorn_access.log \
         --error-logfile gunicorn_error.log
sudo systemctl start nginx
```

---

# How to use

### Keyboard shortcuts

| key | action |
|-----|--------|
| **T** | Toggle between *image view* and a color grid that shows one averaged RGB block per 6 × 4 cell. |
| **1 / 2 / 3** | Switch the displayed image‑audio pair (if multiple variants exist). |

Other interactions:

* Hover a dot to preview the song and read its metadata.
* Genre check‑boxes below the plot let you filter tracks; **Toggle All Genres** flips the selection.

---

# Bring your own data

Place three kinds of files (same *NAME*):

1. `static/audio_clips/NAME.mp3`
2. `static/images/NAME_001.jpg … NAME_N.jpg`
3. `static/jsons/NAME.json`

Dummy files shipped in the repo show the exact layout—open them for reference.

## JSON schema

Each object in `static/jsons/` must look like the snippet below. The server validates fields on startup, so mistakes fail fast.

```jsonc
{
  "x": 0.54,                    // 0.0 ≤ x ≤ 1.0  (valence)
  "y": 0.97,                    // 0.0 ≤ y ≤ 1.0  (arousal)
  "artists": ["Ikimonogakari"], // list of strings
  "track_name": "ホタルノヒカリ",    
  "popularity": 63,             // 0‑100
  "danceability": 0.566,        // 0.0‑1.0
  "energy": 0.92,               // 0.0‑1.0
  "loudness": -1.896,           // dBFS
  "valence": 0.562,             // 0.0‑1.0 (optional duplicate of x)
  "track_genre": "anime",
  "data": [
    {
      "image_path": "static/images/NAME_001.jpg",
      "prompt": "Short text prompt that generated the image",
      "track_genre": "anime",
      "audio_path": "static/audio_clips/NAME.mp3",
      "mean_color": { "r": 33, "g": 45, "b": 52 }  // each 0‑255
    },
    ...
  ]
}
```

> **Important:** `track_genre` inside each `data` row must match the top‑level `track_genre`.