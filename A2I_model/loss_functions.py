"""
Contrastive and InfoNCE loss implementations for training audio-text embedding alignments.
"""

import torch.nn.functional as F
import torch

def contrastive_loss(audio_embeddings:torch.Tensor, positive_embeddings:torch.Tensor, hard_negative_embeddings:torch.Tensor, temperature:float=0.07):
    """
    Compute contrastive loss between audio and text embeddings.

    # EXAMPLE:
    >> audio = torch.randn(70, 77, 2048)
    >> pos = torch.randn(70, 77, 2048)
    >> neg = torch.randn(70, 77, 2048)
    >> loss = contrastive_loss(audio, pos, neg)
    tensor(1.4477)

    :param audio_embeddings: Tensor of shape [B, 77, D]
    :param positive_embeddings: Tensor of shape [B, 77, D]
    :param hard_negative_embeddings: Tensor of shape [B, 77, D]
    :param temperature: Scaling factor for logits (default: 0.07)
    :return: Scalar tensor with contrastive loss
    """

    # Step 1: Reduce sequence length dimension (average pooling)
    audio_embeddings = audio_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
    positive_embeddings = positive_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
    hard_negative_embeddings = hard_negative_embeddings.mean(dim=1)  # [batch_size, embedding_dim]

    # Step 2: Normalize embeddings
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)  # [batch_size, embedding_dim]
    positive_embeddings = F.normalize(positive_embeddings, dim=-1)  # [batch_size, embedding_dim]
    hard_negative_embeddings = F.normalize(hard_negative_embeddings, dim=-1)  # [batch_size, embedding_dim]

    # Step 3: Compute positive similarity
    # Dot product between audio and positive embeddings, scaled by temperature
    pos_sim = torch.sum(audio_embeddings * positive_embeddings, dim=-1) / temperature  # [batch_size]

    # Step 4: Compute hard negative similarity
    # Dot product between audio and hard negative embeddings, scaled by temperature
    hard_neg_sim = torch.sum(audio_embeddings * hard_negative_embeddings, dim=-1) / temperature  # [batch_size]

    # Step 5: Compute loss for positive pairs
    loss_pos = -F.logsigmoid(pos_sim)  # Maximize similarity for positive pairs

    # Step 6: Compute loss for hard negative pairs
    loss_hard_neg = -F.logsigmoid(-hard_neg_sim)  # Minimize similarity for hard negatives

    # Step 7: Combine positive and hard negative losses
    loss = (loss_pos + loss_hard_neg).mean()

    return loss


def info_nce_loss(audio_embeddings:torch.Tensor, positive_embeddings:torch.Tensor, hard_negative_embeddings:torch.Tensor, temperature:float=0.07):
    """
    Compute InfoNCE loss using one positive and multiple hard negatives.

    # MATH & DETAILS:
    TODO: Add link to github / thesis

    # EXAMPLE:
    >> audio_embeddings = torch.randn(70, 77, 2048)
    >> prompt_embeddings = torch.randn(70, 77, 2048)
    >> negative_prompt_embeddings = torch.randn(70, 20, 77, 2048)
    >> loss = info_nce_loss(audio_embeddings, prompt_embeddings, negative_prompt_embeddings)
    tensor(3.0496)

    :param audio_embeddings: Tensor of shape [B, 77, D]
    :param positive_embeddings: Tensor of shape [B, 77, D]
    :param hard_negative_embeddings: Tensor of shape [B, N, 77, D] (N = number of negatives)
    :param temperature: Scaling factor for logits (default: 0.07)
    :return: Scalar tensor with InfoNCE loss
    """

    # Step 1: Reduce sequence length dimension (average pooling)
    audio_embeddings = audio_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
    positive_embeddings = positive_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
    hard_negative_embeddings = hard_negative_embeddings.mean(dim=2)  # [batch_size, num_negatives, embedding_dim]

    # Step 2: Normalize embeddings
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)  # [batch_size, embedding_dim]
    positive_embeddings = F.normalize(positive_embeddings, dim=-1)  # [batch_size, embedding_dim]
    hard_negative_embeddings = F.normalize(hard_negative_embeddings, dim=-1)  # [batch_size, num_negatives, embedding_dim]

    # Step 3: Compute similarity scores
    # Positive similarities
    pos_sim = torch.sum(audio_embeddings * positive_embeddings, dim=-1, keepdim=True)  # [batch_size, 1]

    # Negative similarities
    # Compute dot product between audio_embeddings and each hard negative embedding
    # Expand audio_embeddings to match the negative embeddings shape
    audio_embeddings_expanded = audio_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
    neg_sim = torch.bmm(hard_negative_embeddings, audio_embeddings_expanded.transpose(1, 2)).squeeze(-1)  # [batch_size, num_negatives]

    # Step 4: Concatenate positive and negative similarities
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 1 + num_negatives]

    # Step 5: Apply temperature scaling
    logits = logits / temperature

    # Step 6: Create labels (positives are at index 0)
    labels = torch.zeros(audio_embeddings.size(0), dtype=torch.long, device=audio_embeddings.device)  # [batch_size]

    # Step 7: Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss