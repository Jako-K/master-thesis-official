"""
Dataset/DataLoader classes for audio-text/image pipelines.
"""

import copy
import os
import torch
import pickle
import pandas as pd
import random
from tqdm.notebook import tqdm
import numpy as np
from sound_augmentation import random_crop, add_noise, random_volume_scale, center_crop
from torch.utils.data import DataLoader
import gc


class AudioImagePairDataset:
    def __init__(self, df: pd.DataFrame, cache_path:str, verbose:bool=True, do_info_nce_loss:bool=False, do_augmentation:bool=True):
        self.expected_sample_length = 48_000 * 10
        self.do_info_nce_loss = do_info_nce_loss
        self.do_augmentation = do_augmentation
        cache_paths = {
            "audio": f"{cache_path}_audio.pkl",
            "text":  f"{cache_path}_text.pkl",
            "data":  f"{cache_path}_data.pkl"
        }

        # Create caches
        if not os.path.exists(cache_paths["audio"]):
            self.cache_audio(df, cache_paths["audio"], verbose); gc.collect()
        if not os.path.exists(cache_paths["text"]):
            self.cache_text(df,  cache_paths["text"],  verbose); gc.collect()
        if not os.path.exists(cache_paths["data"]):
            self.cache_data(df,  cache_paths["data"],  verbose); gc.collect()


        # Load caches
        self.audio_path_2_waveform = self.load_cached_data(df, "audio_path_2_waveform", cache_paths["audio"])
        self.prompt_2_embedding = self.load_cached_data(df, "prompt_2_embedding", cache_paths["text"])
        self.combined_data = self.load_cached_data(df, "combined_data", cache_paths["data"])
        self.df = df


    @staticmethod
    def cache_audio(df, save_cache_path, verbose):
        # Construct
        audio_path_2_waveform = {}
        iterator = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        for df_index, row in iterator:
            audio_path = row["path_audio_full"]
            cache_song_path = row["cache_song_path"]
            assert os.path.exists(cache_song_path), cache_song_path
            audio_path_2_waveform[audio_path] = np.load(cache_song_path)

        # Cache
        with open(save_cache_path, "wb") as f:
            pickle.dump({
                "df": df[["spotify_track_id"]],
                "audio_path_2_waveform": audio_path_2_waveform,
            }, f)
        assert os.path.exists(save_cache_path), save_cache_path


    @staticmethod
    def cache_text(df, save_cache_path, verbose):
        # Construct
        prompt_2_embedding = {}
        iterator = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        for df_index, row in iterator:
            prompt = row["openai_prompts"]
            cache_clip_path = row["cache_clip_path"]
            assert os.path.exists(cache_clip_path)
            prompt_index = row["openai_prompt_index"]
            prompt_2_embedding[prompt] = np.load(cache_clip_path)[prompt_index]

        # Cache
        with open(save_cache_path, "wb") as f:
            pickle.dump({
                "df": df[["spotify_track_id"]],
                "prompt_2_embedding": prompt_2_embedding,
            }, f)
        assert os.path.exists(save_cache_path), save_cache_path


    @staticmethod
    def load_cached_data(df, key_name, load_cache_path):
        # Load cache if it is there
        assert os.path.exists(load_cache_path)
        with open(load_cache_path, "rb") as f:
            cache = pickle.load(f)
        data = cache[key_name]
        found_ids = cache["df"]["spotify_track_id"]
        expected_ids = df["spotify_track_id"]
        assert (found_ids == expected_ids).all(), f"cache mismatch! Delete the old cache at {load_cache_path}"
        return data


    @staticmethod
    def cache_data(df, save_cache_path, verbose):
        # Compute VA-distances and genre-differences
        distances_all = []
        genre_matches = []
        coordinates = df[["spotify_valence", "spotify_energy"]]
        for i, (row_index, row) in enumerate(df.iterrows()):
            assert row_index == i
            coordinate = coordinates.loc[row_index].to_numpy()
            distances = np.linalg.norm(coordinate-coordinates, axis=1)
            genre_match = (df["spotify_genre"] == row["spotify_genre"]).to_numpy()
            assert distances[i] == 0
            assert genre_match[i] == True
            assert len(distances) == len(genre_match) == df.shape[0]
            distances_all.append(distances)
            genre_matches.append(genre_match)
        assert len(genre_matches) == len(distances_all) == len(df)
        df["train_distances"] = distances_all
        df["train_genre_match"] = genre_matches


        # Precompute valid negative indices by genre
        genre_to_indices = {genre: df[df["spotify_genre"] == genre].index for genre in df["spotify_genre"].unique()}
        df["train_distances_argsorted"] = df["train_distances"].apply(np.argsort)


        # Create data used for
        iterator = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        combined_data = []
        for i, (df_index, row) in enumerate(iterator):
            assert i == df_index
            sorted_indices = row["train_distances_argsorted"]
            valid_negative_genre_indices = df.index.difference(genre_to_indices[row["spotify_genre"]]) # Get valid negative indices for the row's genre
            valid_indices = sorted_indices[np.isin(sorted_indices, valid_negative_genre_indices) & (row["train_distances"][sorted_indices] > 0.1)]

            assert len(valid_indices) > 5
            negative_indexes = valid_indices[-20:]
            negative_prompts = df.loc[negative_indexes, "openai_prompts"]
            negative_pool = list(zip(negative_indexes, negative_prompts))

            combined_data.append({
                "positive_row_index": row.name,
                "audio_path": row["path_audio_full"],
                "prompt": row["openai_prompts"],
                "negative_pool": negative_pool
            })


        # Cache
        with open(save_cache_path, "wb") as f:
            pickle.dump({
                "df": df,
                "combined_data": combined_data,
            }, f)
        assert os.path.exists(save_cache_path), save_cache_path


    def __len__(self):
        return len(self.combined_data)


    def __getitem__(self, i):
        data = self.combined_data[i]

        # Audio augmentation
        waveform = self.audio_path_2_waveform[data["audio_path"]]
        if self.do_augmentation:
            waveform = random_crop(waveform, self.expected_sample_length)
            if random.random() < 0.5:
                waveform = add_noise(waveform)
            if random.random() < 0.5:
                waveform = random_volume_scale(waveform)
        else:
            waveform = center_crop(waveform, self.expected_sample_length)

        # Prompt embeddings
        prompt_embeddings = self.prompt_2_embedding[data["prompt"]]
        if self.do_info_nce_loss:
            negative_row_index = torch.cat(
                [torch.tensor(d[0]).unsqueeze(0) for d in data["negative_pool"]]
            )
            negative_prompt_embeddings = torch.cat(
                [torch.tensor(self.prompt_2_embedding[d[1]]).unsqueeze(0) for d in data["negative_pool"]]
            )
        else:
            negative_row_index, negative_prompt = random.choice(data["negative_pool"])
            negative_prompt_embeddings = self.prompt_2_embedding[negative_prompt]

        # Extra info
        columns_of_interest = ["spotify_track_id", "spotify_valence", "spotify_energy", "spotify_genre"]
        song_info = self.df.iloc[data["positive_row_index"]][columns_of_interest].to_dict()

        return_data = {
            "song_info": song_info,
            "positive_row_index": data["positive_row_index"],
            "negative_row_index": negative_row_index,
            "waveform": waveform,
            "prompt_embeddings": prompt_embeddings,
            "negative_prompt_embeddings": negative_prompt_embeddings,
        }
        return return_data


class AudioTextPairDataLoader:
    def __init__(self, df_all, batch_size, max_dataset_length, shuffle:bool, cache_folder:str, dataset_type:str,
                 seed:int, device:str="cuda", number_of_prompts_per_song:int=10, do_info_nce_loss:bool=False, do_augmentation:bool=True):
        # Setup
        self.batch_size = batch_size
        self.max_dataset_length = max_dataset_length
        self.number_of_prompts_per_song = number_of_prompts_per_song
        self.max_dataset_length_exploded = max_dataset_length * self.number_of_prompts_per_song
        self.seed = seed
        self.device = device
        self.shuffle = shuffle
        self.cache_folder = cache_folder
        self.current_dl = None
        self.current_dl_counter = -1
        self.dataset_type = dataset_type
        self.cache_base_path = f"{self.cache_folder}/{self.dataset_type}_{self.max_dataset_length}_{self.number_of_prompts_per_song}_{self.seed}_{self.shuffle}"
        self.do_info_nce_loss = do_info_nce_loss
        self.do_augmentation= do_augmentation


        assert os.path.exists(cache_folder) and os.path.isdir(cache_folder)
        assert 0 < self.number_of_prompts_per_song <= 10, self.number_of_prompts_per_song

        # Initialization
        df_all = df_all.reset_index(drop=True)
        if self.number_of_prompts_per_song != 10:
            df_all["openai_prompts"] = df_all["openai_prompts"].apply(lambda prompts: prompts[:self.number_of_prompts_per_song])
        self.df_chucks = self._make_dataset_dataframes(df_all)
        self.sample_count = sum(len(dataset) for dataset in self.df_chucks)
        self.num_datasets = len(self.df_chucks)
        self.batch_count = int(sum(np.ceil(len(dataset)/self.batch_size) for dataset in self.df_chucks))
        self._init_datasets()
        assert self.sample_count == len(df_all)*self.number_of_prompts_per_song, self.sample_count


    def _make_dataset_dataframes(self, df_all):
        n_df = len(df_all)
        df_chucks = []
        num_datasets = int(np.ceil(n_df / self.max_dataset_length))
        df_all_subset = df_all.explode("openai_prompts")
        df_all_subset["openai_prompt_index"] = df_all_subset.groupby(level=0).cumcount()
        if self.shuffle:
            df_all_subset = df_all_subset.sample(frac=1.0, random_state=self.seed)
        for dataset_index in range(num_datasets):
            start_index = dataset_index * self.max_dataset_length_exploded
            end_index = min( (dataset_index+1) * self.max_dataset_length_exploded, len(df_all)*self.number_of_prompts_per_song )
            chunk = df_all_subset.iloc[start_index:end_index].reset_index(drop=True)
            chunk = copy.deepcopy(chunk) # TODO: This is hella expensive and should probably be removed. I put it here to avoid memory bugs initially.
            df_chucks.append(chunk)
            print(f"{dataset_index}: unique track_ids: {len(chunk['spotify_track_id'].unique())}")
        assert all([len(df)<=self.max_dataset_length_exploded for df in df_chucks])
        return df_chucks


    def _init_datasets(self):
        print("Initialize datasets")
        for dataset_index, df in enumerate(tqdm(self.df_chucks)):
            cache_path = f"{self.cache_base_path}_{dataset_index+1}-{self.num_datasets}"
            is_already_cached = all([os.path.exists(p) for p in [f"{cache_path}_audio.pkl", f"{cache_path}_text.pkl", f"{cache_path}_data.pkl"]])
            if not is_already_cached:
                dataset = AudioImagePairDataset(df, cache_path=cache_path, do_info_nce_loss=self.do_info_nce_loss, do_augmentation=self.do_augmentation)
                del dataset
                gc.collect()

    def __iter__(self):
        for dataset_index in range(self.num_datasets):
            cache_path = f"{self.cache_base_path}_{dataset_index+1}-{self.num_datasets}"
            df = self.df_chucks[dataset_index]
            dataset = AudioImagePairDataset(df, cache_path=cache_path, verbose=False, do_info_nce_loss=self.do_info_nce_loss, do_augmentation=self.do_augmentation)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
            for batch in dataloader:
                batch["prompt_embeddings"] = batch["prompt_embeddings"].to(self.device)
                batch["negative_prompt_embeddings"] = batch["negative_prompt_embeddings"].to(self.device)
                yield batch
            del dataset, dataloader


    def __len__(self):
        return self.batch_count