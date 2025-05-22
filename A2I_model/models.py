"""
Defines NN-modules to encode audio (AudioEncoder) and text (ClipTextEncoder)
into fixed-length embeddings compatible with Stable Diffusion XL image generation.
"""

from typing import List, Union

import numpy as np
import torch.nn as nn
from transformers import ClapAudioModelWithProjection, ClapProcessor
from diffusers import StableDiffusionXLPipeline
import torch

class AudioEncoder(nn.Module):
    """
    Encode raw audio waveforms into fixed-length embeddings for SDXL.

    Uses a pretrained CLAP model to extract audio embeddings, projects them
    to match 2 text-encoder dimensions, adds positional embeddings, and
    processes through Transformer encoders to produce a sequence of length 77
    with embedding size 2048 as expected by SDXL.

    # EXAMPLE:
    >> encoder = AudioEncoder(device="cuda")
    >> audio_tensors = [torch.randn(480000) for _ in range(4)]
    >> embeds = encoder(audio_tensors)  # embeds.shape == (4, 77, 2048)

    :param device: Torch device (e.g. "cuda" or "cpu")
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Load the pretrained CLAP model
        self.clap_model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

        # Project CLAP embeddings to match the embedding sizes of text encoders
        self.projection_1 = nn.Linear(512, 768).to(self.device)   # To match text_encoder_1 output size
        self.projection_2 = nn.Linear(512, 1280).to(self.device)  # To match text_encoder_2 output size

        # Positional embeddings to create a sequence of length 77
        self.sequence_length = 77
        self.positional_embeddings_1 = nn.Parameter(torch.zeros(1, self.sequence_length, 768)).to(self.device)
        self.positional_embeddings_2 = nn.Parameter(torch.zeros(1, self.sequence_length, 1280)).to(self.device)

        # Transformer encoders to expand the embeddings to a sequence
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=4).to(self.device)

        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=1280, nhead=8)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=4).to(self.device)

    def forward(self, audio_data: List[np.ndarray]):
        """
        Convert a batch of raw audio tensors into a sequence of embeddings.

        # EXAMPLE:
        >> audio_encoder = AudioEncoder(device="cuda")
        >> audio_tensors = [np.random.randn(480000) for _ in range(4)]
        >> embeds = audio_encoder(audio_tensors)
        >> assert embeds.shape == (4, 77, 2048)

        :param audio_data: List of 1D np.ndarray audio waveforms (each up to 48000*10 samples)
        :return: Float16 tensor of shape (batch_size, 77, 2048)
        """

        # Process the audio data
        # TODO: Should I manually check the inputs here to catch errors early on?
        inputs = self.processor(audios=audio_data, sampling_rate=48000, return_tensors="pt", padding=True, max_length=48000*10)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get CLAP audio embeddings
        audio_embedding = self.clap_model(**inputs).audio_embeds  # Shape: [batch_size, 512]

        # Project to match embedding sizes
        projected_embedding_1 = self.projection_1(audio_embedding)  # Shape: [batch_size, 768]
        projected_embedding_2 = self.projection_2(audio_embedding)  # Shape: [batch_size, 1280]

        # Expand to sequences of length 77
        expanded_embedding_1 = projected_embedding_1.unsqueeze(1).repeat(1, self.sequence_length, 1)  # Shape: [batch_size, 77, 768]
        expanded_embedding_2 = projected_embedding_2.unsqueeze(1).repeat(1, self.sequence_length, 1)  # Shape: [batch_size, 77, 1280]

        # Add positional embeddings
        embeddings_with_position_1 = expanded_embedding_1 + self.positional_embeddings_1  # Shape: [batch_size, 77, 768]
        embeddings_with_position_2 = expanded_embedding_2 + self.positional_embeddings_2  # Shape: [batch_size, 77, 1280]

        # Pass through transformer encoders
        embeddings_with_position_1 = embeddings_with_position_1.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        embeddings_with_position_2 = embeddings_with_position_2.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]

        transformer_output_1 = self.transformer_encoder_1(embeddings_with_position_1)  # [seq_len, batch_size, 768]
        transformer_output_2 = self.transformer_encoder_2(embeddings_with_position_2)  # [seq_len, batch_size, 1280]

        transformer_output_1 = transformer_output_1.permute(1, 0, 2)  # [batch_size, seq_len, 768]
        transformer_output_2 = transformer_output_2.permute(1, 0, 2)  # [batch_size, seq_len, 1280]

        # Concatenate embeddings along the last dimension
        transformer_output = torch.cat([transformer_output_1, transformer_output_2], dim=-1)  # [batch_size, seq_len, 2048]

        # Output shape: [batch_size, 77, 2048]
        return transformer_output.to(torch.float16)


class ClipTextEncoder(torch.nn.Module):
    """
    Encode text prompts into sequence embeddings compatible with SDXL.

    Uses the StableDiffusionXLPipeline's tokenizers and text encoders to produce 2 (dual)
    hidden-state sequences, then concatenates them to form a tensor of shape (batch_size, 77, 2048).

    # EXAMPLE:
    >> encoder = ClipTextEncoder(device="cuda")
    >> embeds = encoder(["a cat", "a dog"])
    >> assert embeds.shape == (2, 77, 2048)

    :param device: Torch device (e.g. "cuda" or "cpu")
    """
    def __init__(self, device:str="cuda"):
        super(ClipTextEncoder, self).__init__()
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(device)

        self.device = device
        self.tokenizer_1 = pipeline.tokenizer
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_1 = pipeline.text_encoder.to(self.device)
        self.text_encoder_2 = pipeline.text_encoder_2.to(self.device)
        del pipeline
        torch.cuda.empty_cache()

    def forward(self, prompts:Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenize and encode text prompts into hidden-state embeddings.

        # EXAMPLE:
        >> out = encoder("sunrise")
        >> assert out.shape == (1, 77, 2048)

        :param prompts: Single string or list of strings to encode
        :return: Float tensor of shape (batch_size, 77, 2048)
        """
        with torch.no_grad():
            tokens_text_encoder_1 = self.tokenizer_1(prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(self.device)
            tokens_text_encoder_2 = self.tokenizer_2(prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(self.device)

            text_encoder_output_1 = self.text_encoder_1(tokens_text_encoder_1, output_hidden_states=True)
            prompt_embeds_1 = text_encoder_output_1.hidden_states[-2]

            text_encoder_output_2 = self.text_encoder_2(tokens_text_encoder_2, output_hidden_states=True)
            prompt_embeds_2 = text_encoder_output_2.hidden_states[-2]

            # Concatenate embeddings along the last dimension
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        # Output shape: [batch_size, 77, 2048]
        return prompt_embeds