"""
Utilities classes for guiding Stable Diffusion XL with audio embeddings.
"""

import warnings
from diffusers import StableDiffusionXLPipeline
import librosa.feature
import numpy as np
from diffusers.utils import logging
import pandas as pd
from justshowit import show
import os
import torch
from helpers_general import create_image_with_prompt, combine_videos_ffmpeg, save_as_video_clip
import shutil
import simple_asserts as S
from PIL.Image import Image
from typing import List, Optional, Dict


class StableDiffusionHelper:
    """
    Helper for generating images and videos from audio with Stable Diffusion XL.

    # EXAMPLE:
    >> helper = StableDiffusionHelper(spotify_id_2_song_name)
    >> clips = helper.load_song_whole_in_clips("path/to/song.mp3")
    >> center_clip = helper.load_song_clip("path/to/song.mp3", crop_method="center")
    >> helper.load_pipeline()
    >> image = helper.generate_image(prompt_embedding, guidance_scale=15.0)
    >> helper.batch_2_video(batch, audio_embeddings, "out_folder", guidance_scale=15.0, prefix="epoch1")
    >> helper.clear_pipeline()

    :param spotify_id_2_song_name: Dict mapping Spotify track IDs to human-readable song names.
    """

    def __init__(self, spotify_id_2_song_name:Optional[Dict[str, str]]=None):
        self.scheduler = None
        self.unet = None
        self.vae = None
        self.image_processor = None
        self.EXPECTED_SAMPLE_LENGTH = 48_000 * 10
        self.spotify_id_2_song_name = {}
        if spotify_id_2_song_name is not None:
            self.spotify_id_2_song_name = spotify_id_2_song_name

    def generate_image(self, embedding: torch.Tensor, guidance_scale: float = 20.0) -> np.ndarray:
        """
        Generate an image from a single prompt embedding using SDXL-like diffusion.

        # EXAMPLE:
        >> embedding = torch.randn(1, 77, 2048)
        >> image = self.generate_image(embedding, guidance_scale=30.0)

        :param embedding: Tensor of shape (77, 2048) or (1, 77, 2048)
        :param guidance_scale: Float controlling classifier-free guidance strength
        :return: Image as numpy array (1024 x 1024 x 3)
        """
        # Input checks
        S.assert_type(embedding, torch.Tensor, "prompt_embedding")
        if embedding.shape == (77, 2048):
            embedding = embedding.unsqueeze(0)
        if embedding.shape != (1, 77, 2048):
            raise ValueError(f"Expected shape (1, 77, 2048), got {tuple(embedding.shape)}")
        S.assert_positive_float(guidance_scale, zero_allowed=True, variable_name="guidance_scale")

        # Create prompt embeddings (positive and negative)
        negative_prompt_embeds = torch.zeros_like(embedding)
        prompt_embeds = torch.cat([negative_prompt_embeds, embedding], dim=0).to(dtype=torch.float16)

        # Initialize latents
        num_inference_steps = 50
        self.scheduler.set_timesteps(num_inference_steps)
        latent_shape = (1, 4, 128, 128) # Changing this will alter the image output size e.g. (1, 4, 64, 64) --> 512x512
        latents = torch.randn(latent_shape, device="cuda", dtype=torch.float16)
        latents *= self.scheduler.init_noise_sigma

        # Denoising loop
        add_text_embeds = torch.zeros((2, 1280), dtype=torch.float16, device="cuda")
        add_time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]], dtype=torch.float16, device="cuda").repeat(2, 1)
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                latent_input = torch.cat([latents] * 2)
                latent_input = self.scheduler.scale_model_input(latent_input, t)
                noise_pred = self.unet(
                    latent_input, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                    return_dict=False
                )[0]
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, generator=None, return_dict=False)[0]

        # Decode latents to image
        latents = latents.to(dtype=torch.float32) / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0] # Latent --> pixel space
            image = self.image_processor.postprocess(image, output_type='pil')[0]
        torch.cuda.empty_cache()

        return np.asarray(image)


    def batch_2_video(self, batch: dict, audio_embeddings: torch.Tensor, save_folder: str, guidance_scale: float,
                      prefix: str = "", number_of_images: Optional[int] = None, add_prefix_folder: bool = True) -> None:
        """
        Generate an info image + SDXL image + audio clip per embedding, save each as MP4,
        then concatenate into one combined video.

        # EXAMPLE:
        >> self.batch_2_video(
        ...     valid_batch,             # must have 'song_info' list and 'waveform' Tensor
        ...     valid_audio_embeddings,  # Tensor of shape (N, seq_len, emb_dim)
        ...     "outputs",
        ...     guidance_scale=7.5
        ... )

        :param batch: {
            "song_info": list of dicts with at least "spotify_track_id",
            "waveform": Tensor of shape (N, samples)
        }
        :param audio_embeddings: Tensor of shape (N, sequence_length, embedding_dim)
        :param save_folder: existing directory to write outputs into
        :param guidance_scale: positive float
        :param prefix: optional filename prefix (no trailing underscore)
        :param number_of_images: how many items to process (defaults to N)
        :param add_prefix_folder: if True, a subfolder `save_folder/prefix` will be created
        :return: None
        """
        import torch

        # Input checks - batch
        S.assert_type(batch, dict, "batch")
        if "song_info" not in batch or "waveform" not in batch:
            raise KeyError("`batch` must contain 'song_info' and 'waveform'")
        S.assert_type(batch["waveform"], torch.Tensor, "batch['waveform']")

        S.assert_type(batch["song_info"], dict, "batch['song_info']")
        assert list(batch["song_info"].keys()) == ['spotify_track_id', 'spotify_valence', 'spotify_energy', 'spotify_genre']
        assert [type(v) for v in batch["song_info"].values()] == [list, torch.Tensor, torch.Tensor, list]
        assert len(set([len(v) for v in batch["song_info"].values()])) == 1, "Expected all entries in batch to have the same length."

        waveform = batch["waveform"]
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be 2D got shape `{tuple(waveform.shape)}`")
        n_items, n_samples = waveform.shape

        # Input check - audio_embeddings
        S.assert_type(audio_embeddings, torch.Tensor, "audio_embeddings")
        if audio_embeddings.ndim != 3:
            raise ValueError(f"audio_embeddings must be 3D got ndim=`{audio_embeddings.ndim}`")
        n_emb, seq_len, emb_dim = audio_embeddings.shape
        if n_emb != n_items:
            raise ValueError(f"audio_embeddings length `{n_emb}` must equal batch size `{n_items}`")

        # Input check - number_of_images
        if number_of_images is None:
            number_of_images = n_emb
        else:
            S.assert_positive_int(number_of_images, zero_allowed=False)
            if number_of_images > n_emb:
                raise ValueError(f"{number_of_images=} > number of embeddings ({n_emb})")

        # Input check - number_of_images, save_folder, prefix, add_prefix_folder, guidance_scale
        S.assert_folder_exists(save_folder, "save_folder")
        S.assert_type(prefix, str, "prefix")
        S.assert_type(add_prefix_folder, bool, "add_prefix_folder")
        S.assert_positive_float(guidance_scale, zero_allowed=False, variable_name="guidance_scale")

        # Prepare folders & prefix formatting
        save_folder_original = save_folder
        if add_prefix_folder:
            if not prefix:
                raise ValueError("prefix must be non-empty when add_prefix_folder=True")
            save_folder = os.path.join(save_folder, prefix)
            os.makedirs(save_folder, exist_ok=True)
        if prefix:
            prefix = prefix.rstrip("_") + "_"

        # Build DataFrame of song info
        df_song = pd.DataFrame(batch["song_info"]).iloc[:number_of_images]
        df_song["spotify_track_name"] = df_song["spotify_track_id"].map(
            lambda x: self.spotify_id_2_song_name.get(x, "")[:45]
        )

        # Generate everything for each video
        video_paths = []
        for i, row in df_song.iterrows():
            # 1) Create info image
            info_dict = row.to_dict()
            info_dict["guidance_scale"] = guidance_scale
            keys = [
                "spotify_track_name", "spotify_genre", "guidance_scale",
                "spotify_energy", "spotify_valence", "spotify_track_id"
            ]
            info_text = "\n".join(
                f"{k.replace('spotify_', '').upper()}:\n {info_dict.get(k, '')}\n"
                for k in keys
            ).strip()
            info_image = create_image_with_prompt(info_text, font_size=30, text_start_height=15)

            # 2) Generate SDXL image
            emb = audio_embeddings[i]  # shape: (seq_len, emb_dim)
            image = self.generate_image(emb, guidance_scale)
            show(
                image,
                save_image_path=os.path.join(save_folder, f"{prefix}{row.spotify_track_id}.jpg"),
                display_image=False
            )

            # 3) Combine info + generated image
            combined_jpg = os.path.join(save_folder, f"{prefix}combined_{row.spotify_track_id}.jpg")
            image_combined = show(
                [info_image, np.asarray(image)],
                display_image=False,
                return_image=True,
                save_image_path=combined_jpg
            )

            # 4) Save as video clip
            vid_path = os.path.join(save_folder, f"{prefix}{row.spotify_track_id}.mp4")
            save_as_video_clip(
                waveform[i].cpu().numpy(),
                image_combined,
                video_output_path=vid_path,
                expected_sample_length=n_samples
            )
            video_paths.append(vid_path)

        # Concatenate all generated videos
        combined_mp4 = os.path.join(save_folder, f"{prefix}combined.mp4")
        combine_videos_ffmpeg(video_paths, combined_mp4)

        # Copy final to original folder
        target = os.path.join(save_folder_original, f"{prefix}.mp4").replace("_.mp4", ".mp4")
        shutil.copy(combined_mp4, target)


    def load_pipeline(self) -> None:
        """
        Load and configure the Stable Diffusion XL pipeline.

        Loads scheduler, UNet, VAE, and image processor to GPU. Frees unused parts to reduce memory use.
        """
        # Load model
        logging.disable_progress_bar()
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        logging.enable_progress_bar()

        # Grab what I need and delete the rest
        torch.cuda.empty_cache()
        self.scheduler = pipeline.scheduler
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.vae.post_quant_conv.to(self.vae.dtype)
        self.vae.decoder.conv_in.to(self.vae.dtype)
        self.vae.decoder.mid_block.to(self.vae.dtype)
        self.vae.to(dtype=torch.float32)
        self.image_processor = pipeline.image_processor
        del pipeline
        torch.cuda.empty_cache()


    def clear_pipeline(self):
        """
        Clears the loaded model components from memory.

        Frees up GPU memory by deleting the scheduler, UNet, VAE, and image processor.
        """
        del self.scheduler
        self.scheduler = None
        del self.unet
        self.unet = None
        del self.vae
        self.vae = None
        del self.image_processor
        self.image_processor = None
        torch.cuda.empty_cache()


class StableDiffusionTester:
    """
    Generates and saves test images using Stable Diffusion XL.

    # EXAMPLE:
    >> sdxl_tester = StableDiffusionTester(path_test_data, text_encoder, guidance_scale=25.0)
    >> sdxl_tester.generate_images_from_audio_encoder(audio_encoder, 32)
    >> sdxl_tester.save_images_to_disk(C.path_logging_debug_image_folder, f"{epoch}_test")

    :param test_data_folder_path: Path to folder with:
        - `exp_info.csv`: must contain columns like `path_audio_clip`, `path_image`, and `flux_generation_prompt`
        - cropped audio clips (e.g. `cut_XXXXX.mp3`)
        - real target images (e.g. `XXXXX_2.jpg`)
        - optionally SDXL-generated images (e.g. `SDXL_XXXXX_2.jpg`)

    :param text_encoder: Model that turns prompts into (32, 77, 2048) embeddings
    :param guidance_scale: Classifier-free guidance scale (0–200 recommended)
    """

    def __init__(self, test_data_folder_path:str, text_encoder:torch.nn.Module, guidance_scale:float=30.0):
        # Input checks
        S.assert_types(
            [test_data_folder_path, text_encoder, guidance_scale],
            [str, torch.nn.Module, float],
            ["test_data_folder_path", "text_encoder", "guidance_scale"],
        )
        S.assert_folder_exists(test_data_folder_path, "test_data_folder_path")
        S.assert_positive_float(guidance_scale, max_value_allowed=200.0) # TODO: SDXL breaks with GS-values this high

        # Derived path checks
        csv_path = f"{test_data_folder_path}/exp_info.csv"
        S.assert_path_exists(csv_path, "csv_path")
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.df["path_audio_clip"] = self.df["path_audio_clip"].apply(lambda p:f"{test_data_folder_path}/{os.path.basename(p)}")
        assert self.df["path_audio_clip"].apply(os.path.exists).all(), "Found one or more bad paths"
        self.df["path_image"] = self.df["path_image"].apply(lambda p:f"{test_data_folder_path}/{os.path.basename(p)}")
        assert self.df["path_image"].apply(os.path.exists).all(), "Found one or more bad paths"

        # Initialization
        self.guidance_scale = guidance_scale
        self.prompts = self.df["flux_generation_prompt"].tolist()
        self.prompts_embeddings = text_encoder(self.prompts)
        assert self.prompts_embeddings.shape == (32, 77, 2048)
        self.audio_data = [librosa.load(p, sr=48000)[0][:48_000*10] for p in self.df["path_audio_clip"]]
        self.audio_data_torch_stacked = torch.stack([torch.tensor(audio) for audio in self.audio_data])
        self.scheduler = None
        self.unet = None
        self.vae = None
        self.image_processor = None
        self.song_generated_images = []

        # Extra stuff
        self.info_images = []
        for _, row in self.df.iterrows():
            # Info image
            row["guidance_scale"] = self.guidance_scale
            new_order = ['spotify_track_name', 'spotify_genre', 'guidance_scale', 'spotify_energy', 'spotify_valence', 'spotify_track_id']
            info = row[new_order].to_dict()
            info = "\n".join([f"{k.replace('spotify_', '').upper()}:\n {v}\n" for k,v in info.items()]).strip()
            info_image = create_image_with_prompt(info, font_size=30, text_start_height=15)
            self.info_images.append(info_image)


    def _load_pipeline(self) -> None:
        """
        Load and configure the Stable Diffusion XL pipeline.

        Loads scheduler, UNet, VAE, and image processor to GPU. Frees unused parts to reduce memory use.
        """

        # Load model
        logging.disable_progress_bar()
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
        logging.enable_progress_bar()

        # Grab what I need and delete the rest
        torch.cuda.empty_cache()
        self.scheduler = pipeline.scheduler
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.vae.post_quant_conv.to(self.vae.dtype)
        self.vae.decoder.conv_in.to(self.vae.dtype)
        self.vae.decoder.mid_block.to(self.vae.dtype)
        self.vae.to(dtype=torch.float32)
        self.image_processor = pipeline.image_processor
        del pipeline
        torch.cuda.empty_cache()


    def _clear_pipeline(self) -> None:
        """
        Clears the loaded model components from memory.

        Frees up GPU memory by deleting the scheduler, UNet, VAE, and image processor.
        """

        del self.scheduler
        self.scheduler = None
        del self.unet
        self.unet = None
        del self.vae
        self.vae = None
        del self.image_processor
        self.image_processor = None
        torch.cuda.empty_cache()


    def _generate_image(self, prompt_embedding:torch.Tensor) -> Image:
        """
        Generate a single image from a given prompt embedding using SDXL.

        :param prompt_embedding: Tensor of shape (1, 77, 2048)
        :return: A PIL image generated from the embedding
        """
        # Init
        S.assert_type(prompt_embedding, torch.Tensor, "prompt_embedding")
        assert prompt_embedding.shape == torch.Size([1, 77, 2048]), prompt_embedding.shape
        negative_prompt_embeds = torch.zeros_like(prompt_embedding)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embedding], dim=0).to(torch.float16)

        # Step 5: Initialize latents and set up noise scheduler
        num_inference_steps = 50
        self.scheduler.set_timesteps(num_inference_steps)
        latent_shape = (1, 4, 128, 128)
        latents = torch.randn(latent_shape, generator=None, device="cuda", dtype=torch.float16, layout=torch.strided).to("cuda")
        latents = latents * self.scheduler.init_noise_sigma

        # Step 6: Denoising loop with uNet
        extra_step_kwargs = {'generator': None}
        add_text_embeds = torch.zeros((2, 1280), dtype=torch.float16).to("cuda")
        add_time_ids = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]], dtype=torch.float16).to("cuda").repeat(2, 1)
        with torch.no_grad():
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # Step 7: Denormalize and decode latents to image with VAE
        latents = latents.to(torch.float32)
        latents = latents / self.vae.config.scaling_factor # Apply scaling and denormalize if mean and std are defined
        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0] # Decode the latents to get the image in pixel space
            image = self.image_processor.postprocess(image, output_type='pil')[0]
        torch.cuda.empty_cache()
        return image


    def generate_images_from_audio_encoder(self, audio_encoder:torch.nn.Module, amount:int=32) -> List[Image]:
        """
        Generate images by passing audio clips through the audio encoder.

        :param audio_encoder: Model that maps `amount` waveform -> embedding with shape (`amount`, 77, 2048)
        :param amount: Number of images to generate
        :return: List of generated PIL images
        """
        S.assert_type(audio_encoder, torch.nn.Module, "audio_encoder")
        S.assert_positive_int(amount, zero_allowed=False, max_value_allowed=len(self.df))

        with torch.no_grad():
            try: # TODO explain why this is necessary
                audio_embeddings = audio_encoder(self.audio_data)
            except AttributeError:
                audio_embeddings = audio_encoder(self.audio_data_torch_stacked.to(audio_encoder.device))
        self._load_pipeline()
        generated_images = [self._generate_image(e.unsqueeze(0)) for e in audio_embeddings[:amount]]
        self._clear_pipeline()
        self.song_generated_images = generated_images
        assert len(self.song_generated_images) == amount, "This should not be possible."
        return generated_images


    def save_images_to_disk(self, save_folder: str, file_prefix: str = "") -> None:
        """
        Save generated images (and matching info) as images and video files.
        NOTE: The function relies on the images produced by `self.generate_images_from_audio_encoder(...)`

        :param save_folder: Root output folder
        :param file_prefix: Filename prefix used for naming outputs. If non-empty, also creates a subfolder.
        :return: None
        """

        # Type check
        S.assert_types([save_folder, file_prefix], [str, str], ["save_folder", "file_prefix"])
        S.assert_folder_exists(save_folder, "save_folder")

        # Setup paths
        save_folder_original = save_folder
        if file_prefix:
            save_folder = os.path.join(save_folder, file_prefix)
            os.makedirs(save_folder, exist_ok=True)

        # Generate images and videos for all test samples
        video_paths = []
        amount = len(self.song_generated_images)
        if amount == 0:
            warnings.warn("You are attempting to save images to disk that have yet to be generated. "
                          "Run `StableDiffusionHelper.generate_images_from_audio_encoder(...)` first. Will return without saving anything.")
            return
        for i in range(amount):
            # Image only
            show(self.song_generated_images[i], save_image_path=f"{save_folder}/{file_prefix}_{i}.jpg", display_image=False)

            # Image with info
            images = [self.info_images[i], self.song_generated_images[i]]
            save_image_path = f"{save_folder}/{file_prefix}_combined_{i}.jpg"
            image_with_info = show(images, display_image=False, return_image=True, save_image_path=save_image_path)

            # Video
            audio_clip = self.audio_data[i]
            save_video_path = f"{save_folder}/{file_prefix}_{i}.mp4"
            save_as_video_clip(audio_clip, image_with_info, save_video_path)
            video_paths.append(save_video_path)

        # Save combined videos
        save_path = f"{save_folder}/{file_prefix}_combined.mp4"
        combine_videos_ffmpeg(video_paths, save_path)
        shutil.copy(save_path, f"{save_folder_original}/{file_prefix}.mp4")