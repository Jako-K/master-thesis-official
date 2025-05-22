# TL;DR

```bash
git clone <this-repo>
cd <this-repo>

# 1. Core model
pip install -r A2I_model/requirements.txt
jupyter lab tutorials/inference_examples.ipynb              # generate images

# 2. Optional extras
pip install -r dataset_creation/requirements.txt
jupyter lab dataset_creation/1_clean_spotify_dataset.ipynb  # recreate dataset

jupyter lab A2I_model/align_audio_with_text_encoder.ipynb   # train

pip install flask
python experiment_website/main.py                           # localhost:5000

python exploration_gui/main_server.py                       # valence–arousal map
```

# AI Audio-to-Image Pipeline

**Master’s Thesis**: *Music-to-Image Synthesis: Controlling Image Generation through Audio Modality*<br>
Thesis available at `./master_thesis.pdf`

### Overview

![thesis-master-overview-small](https://github.com/user-attachments/assets/48eba9e0-4c18-468e-a879-2a0c018ff5d6)
> *Figure: End-to-end pipeline mapping music to emotionally aligned images using CLIP embeddings and Stable Diffusion XL, generating visuals that reflect the overall **feel and mood** of the audio.*

### Contribution

* Designed and validated a human-centered method linking music and images
* Created datasets with 400,000 song–prompt pairs and 15,000 song–image pairs
* Built an audio-to-image model that generates emotionally resonant visuals from music
* Developed a self-hosted labeling platform for song-to-image selection and feedback
* Created an interactive multi-module GUI for data exploration

### Results
![thesis-master-results](https://github.com/user-attachments/assets/0720aebd-3ae9-4c67-8b75-4c5747d5198f)

# Video Demo

https://github.com/user-attachments/assets/4b76f541-cd0c-42d3-9136-8d89ecf05c3d

---

## Repo Map

| Folder                  | Description                                                                                                     |
| ----------------------- |-----------------------------------------------------------------------------------------------------------------|
| **A2I\_model**          | Training and inference code for the audio encoder. Logs, configs, and outputs in `training_logs/<run-id>/`.     |
| **dataset\_creation**   | 5-step pipeline to build the 40K-track dataset from Spotify, YouTube, and ChatGPT prompts.                      |
| **weights**             | Pre-trained audio encoder (`pretrained.pth`).                                                                   |
| **tutorials**           | Minimal notebooks to test the model and inspect SDXL outputs.                                                   |
| **experiment\_website** | Flask web app for large-scale human labeling. Includes login, invite codes, local DB, ...                       |
| **exploration\_gui**    | Interactive valence–arousal dot-plot. Hover to play audio, press **T** to toggle views, **1/2/3** for variants. |
| **dataset**             | Raw/processed audio, metadata, and toy data. Final output of `dataset_creation/`. (\~60 GB total)               |

---

## Training Hardware (for reference)

* CPU  : i9-14900KF @ 3.2 GHz
* GPU  : RTX 4090, 24 GB
* RAM  : 64 GB DDR5
* Disk : ≈ 1 TB cache (Faster I/O)

Training took weeks. You can find my weights in `weights/pretrained.pth`.
> *The full 40K-track dataset isn’t public, but everything you need to reconstruct it is included. Reach out if you need access.*

---

# Music Videos
It’s relatively straightforward to create music videos: split the song into 10-second clips, generate a sequence of images, then AI-interpolate between them to form a video.

https://github.com/user-attachments/assets/0ba4de99-ebf3-430f-ba60-422e1aea7d6e
