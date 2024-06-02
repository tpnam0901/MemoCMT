import torch
import torch.nn as nn

from configs.base import Config

from .modules import build_audio_encoder, build_text_encoder


class TestSER(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TestSER, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

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
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(input_audio.view(-1, *input_audio.shape[2:])).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(batch_size, num_samples, *audio_embeddings.shape[1:])
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        ## Fusion Module

        # Text cross attenttion text Q audio , K and V text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Audio cross attetntion Q text, K and V audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights=False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_norm), 1)

        # Selt-attention module
        fusion_attention, _ = self.fusion_attention(
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
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state


class TestSER_v2(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TestSER_v2, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

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
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim + cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        batch_size, num_samples = input_audio.size(0), input_audio.size(1)
        audio_embeddings = self.audio_encoder(input_audio.view(-1, *input_audio.shape[2:])).last_hidden_state
        audio_embeddings = audio_embeddings.mean(1)
        audio_embeddings = audio_embeddings.view(batch_size, num_samples, *audio_embeddings.shape[1:])

        ## Fusion Module

        # Text cross attenttion text Q audio , K and V text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)

        # Audio cross attetntion Q text, K and V audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights=False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)

        text_norm = text_norm[:, 0, :]
        audio_norm = audio_norm.mean(1)
        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_norm), 1)

        cls_token_final_fusion_norm = fusion_embeddings.view(fusion_embeddings.size(0), -1)

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
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state


class TestSER_v3(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TestSER_v3, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

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
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        
        batch_size, num_samples = input_audio.size(0), input_audio.size(1)
        print(input_audio.size())
        audio_embeddings = self.audio_encoder(input_audio.view(-1, *input_audio.shape[2:])).last_hidden_state
        audio_embeddings = audio_embeddings.mean(1)
        audio_embeddings = audio_embeddings.view(batch_size, num_samples, *audio_embeddings.shape[1:])


        ## Fusion Module

        # Text cross attenttion text Q audio , K and V text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights=False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)
        # Audio cross attetntion Q text, K and V audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights=False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)

        # Concatenate the text and audio embeddings
        fusion_embeddings = torch.cat((text_norm, audio_norm), 1)

        # Selt-attention module
        fusion_attention, _ = self.fusion_attention(
            fusion_embeddings,
            fusion_embeddings,
            fusion_embeddings,
            average_attn_weights=False,
        )

        fusion_linear = self.fusion_linear(fusion_attention)
        fusion_norm = self.fusion_layer_norm(fusion_linear)

        # skip connection
        cls_token_final_fusion_norm = fusion_norm.mean(dim=1) + audio_norm.mean(dim=1)

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
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state



