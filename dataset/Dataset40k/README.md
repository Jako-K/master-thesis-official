
# Dataset
The dataset contains ~40K unique tracks, 400K song-prompt pairs, and 15K song-image pairs. It includes rich metadata from both YouTube and Spotify, along with carefully designed stratified splits based on genre, valence, and arousal. You can reproduce the dataset by running the code in `dataset_creation`.
> *The full 40K-track dataset isn’t public, but everything needed to reconstruct it is included. Reach out if you need access.*

---
# `dataset.csv`

| Column                         | Type / Range       | What it means (short & sweet)                       |
| ------------------------------ | ------------------ |-----------------------------------------------------|
| **spotify\_track\_id**         | `str`              | Spotify ID (unique key).                            |
| **spotify\_track\_name**       | `str`              | Track title exactly as shown on Spotify.            |
| **spotify\_artists**           | `list[str]` (stringified) | All artist names for the track.                     |
| **spotify\_album\_name**       | `str`              | Name of the album the track belongs to.             |
| **spotify\_genre**             | `str`              | Primary genre tag.                                  |
| **spotify\_popularity**        | `int` 0-100        | Spotify popularity score.                           |
| **spotify\_energy**            | `float` 0-1        | Perceived intensity / activity level.               |
| **spotify\_valence**           | `float` 0-1        | Musical “positivity” (1=happy, 0=sad).              |
| **spotify\_danceability**      | `float` 0-1        | How suitable a track is for dancing.                |
| **spotify\_tempo**             | `float` BPM        | Tempo in beats-per-minute.                          |
| **spotify\_duration\_ms**      | `int`              | Track length in milliseconds.                       |
| **spotify\_key**               | `int` 0-11         | Musical key (0 =C, 1 =C♯/D♭ … 11 =B).               |
| **spotify\_mode**              | `int`              | 1 = major, 0 = minor, ...                           |
| **spotify\_loudness**          | `float` dB         | Average loudness (dBFS).                            |
| **spotify\_time\_signature**   | `int` 3-7          | Beats per bar (time signature numerator).           |
| **spotify\_acousticness**      | `float` 0-1        | Confidence the track is acoustic.                   |
| **spotify\_instrumentalness**  | `float` 0-1        | Likelihood the track has *no* vocals.               |
| **spotify\_speechiness**       | `float` 0-1        | Presence of spoken words.                           |
| **spotify\_liveness**          | `float` 0-1        | Probability the recording is live.                  |
| **spotify\_explicit\_content** | `bool`             | `True` if Spotify marks the track “explicit”.       |
| **spotify\_original\_index**   | `int`              | Row index in the original Spotify CSV.              |
| **youtube\_url**               | `str (URL)`        | Full YouTube link for the chosen video.             |
| **youtube\_title**             | `str`              | Video title as returned by the API.                 |
| **youtube\_views**             | `float` / `int`    | View count at download time.                        |
| **youtube\_duration\_ms**      | `float`            | Video length in milliseconds.                       |
| **youtube\_video\_skipped**    | `bool`             | `True` if the video was later excluded/flagged.     |
| **youtube\_search\_query**     | `str`              | Exact query string sent to YouTube Search.          |
| **youtube\_title\_download**   | `str`              | Cleaned title.                                      |
| **youtube\_english\_captions** | `str` (JSON)       | Raw English caption payload (empty string if none). |
| **path\_audio\_full**          | `str (path)`       | Relative path to the downloaded audio file.         |
| **openai\_prompts**            | `list[str]` (stringified) | Track matching prompts generated with ChatGPT.      |
| **train\_valence\_bin**        | `int` 0-3          | Discretised valence bucket for stratified sampling. |
| **train\_energy\_bin**         | `int` 0-3          | Discretised energy bucket.                          |
| **train\_stratify\_key**       | `str`              | Composite key `"<genre>_<val_bin>_<energy_bin>"`.   |
| **train\_split**               | `str`              | One of `train`, `val`, or `test`.                   |
