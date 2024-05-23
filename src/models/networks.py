import torch
import torch.nn as nn

from configs.base import Config

from .modules import build_audio_encoder, build_text_encoder

class _4M_SER(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        """Summary: 4M-SER model version 2 for optimizing parameters.
        Args:
            cfg (Config): Config object
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super(_4M_SER, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_norm_type = cfg.audio_norm_type
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        if cfg.audio_norm_type == "layer_norm":
            self.audio_encoder_layer_norm = nn.LayerNorm(cfg.audio_encoder_dim)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.text_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.audio_encoder_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.audio_encoder_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.audio_encoder_dim, cfg.audio_encoder_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.audio_encoder_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type
        self.transfer_learning = False

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = input_text
        audio_embeddings = input_audio
        if not self.transfer_learning:
            text_embeddings = self.text_encoder(input_text).last_hidden_state
            audio_embeddings = self.audio_encoder(input_audio)

        if self.audio_norm_type == "layer_norm":
            audio_embeddings_norm = self.audio_encoder_layer_norm(audio_embeddings)
        elif self.audio_norm_type == "min_max":
            # Min-max normalization
            audio_embeddings_norm = (audio_embeddings - audio_embeddings.min()) / (
                audio_embeddings.max() - audio_embeddings.min()
            )
        else:
            audio_embeddings_norm = audio_embeddings

        ## Fusion Module
        # Self-attention to reduce the dimensionality of the text embeddings
        text_attention, text_attn_output_weights = self.text_attention(
            text_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_embeddings_norm), 1)

        # Selt-attention module
        fusion_attention, fusion_attn_output_weights = self.fusion_attention(
            fusion_embeddings,
            fusion_embeddings,
            fusion_embeddings,
            average_attn_weights=False,
        )
        fusion_linear = self.fusion_linear(fusion_attention)
        fusion_norm = self.fusion_layer_norm(fusion_linear)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [
                text_attn_output_weights,
                fusion_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_embeddings_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state