# Dataset Creation
The dataset is created in five steps, each in a separate Jupyter notebook. Intermediate files will be saved to disk and not deleted, but ultimately only `../dataset/Dataset40K/dataset.csv` and the data in `../dataset/Dataset40K/*` matter.

---
### Good to Know

1. See `../dataset/Dataset40K/README.md` for more details about the final dataset.
2. Downloading the YouTube data can take a significant amount of time—potentially several days.
3. You will likely need to update `pytube`, but you can try with the version specified in `requirements.txt`
4. A valid `chromedriver.exe` is required for the YouTube scraping to work.
5. Generating prompts with ChatGPT is time-consuming and requires an OpenAI API key with approximately \$20 in available credit. 
6. When I ran the code, the final, combined dataset required ~60GB storage.


---
### Code Breakdown
1. `1_clean_spotify_dataset.ipynb`: Uses Maharshi Pandya’s excellent [Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and performs initial preprocessing to suit the needs of this project.
2. `2_youtube_download.ipynb`: First scrapes YouTube URLs and titles matching the Spotify data from step 1 using pre-made search queries. Then it downloads the corresponding YouTube audio, metadata, and thumbnails.
3. `3_prepare_dataset_for_prompts.ipynb`: Combines everything from steps 1 and 2, and performs light cleaning and reformatting to prepare for prompt generation.
4. `4_openai_prompt_generation.ipynb`: Uses ChatGPT to generate 10 hand-tailored prompts for each song. For each song, we now have: (1) Spotify metadata, (2) YouTube audio, thumbnail, and metadata, and (3) 10 prompts.
5. `5_create_train_valid_and_test_splits.ipynb`: Performs a comprehensive, stratified train/validation/test split based on the Spotify metadata.