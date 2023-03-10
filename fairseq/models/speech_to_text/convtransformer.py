#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple, Any

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.distributed import fsdp_wrap
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    FairseqDropout,
)
from torch import Tensor

logger = logging.getLogger(__name__)

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


@register_model("convtransformer")
class ConvTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument("--dropout", type=float, metavar="D", help="dropout probability")
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument("--encoder-layers", type=int, metavar="N", help="num encoder layers")
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument("--decoder-layers", type=int, metavar="N", help="num decoder layers")
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="INT",
            help="the number of output channels of conv layer",
        )
        parser.add_argument(
            "--freeze-pretrained-encoder",
            action="store_true",
            help="freeze pretrained encoder during training",
        )
        parser.add_argument(
            "--freeze-pretrained-decoder",
            action="store_true",
            help="freeze pretrained decoder during training",
        )
        parser.add_argument(
            "--reset-parameter-state",
            action="store_true",
            help="unfreeze both encoder and decoder during training",
        )
        # lm args
        parser.add_argument('--add-language-model', action='store_true', default=False,
                            help='if True, spwan a Transformer decoder for an LM.')
        parser.add_argument("--share-transformer-lm-decoder-embed", action="store_true", 
                            default=False, help="share transformer decoder embedding and lm decoder embedding.")
        parser.add_argument("--share-lm-decoder-softmax-embed", action="store_true", 
                            default=False, help="share lm decoder embedding and softmax weight.")
        parser.add_argument("--share-transformer-lm-decoder-softmax-embed", action="store_true", 
                            default=False, help="share transformer decoder embedding and lm decoder embedding \
                            and softmax weights.")

    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info("Successfully loaded pretrained encoder from : {}".format(args.load_pretrained_encoder_from))
        
        if getattr(args, "freeze_pretrained_encoder", False):
            for k, p in encoder.named_parameters():
                p.requires_grad = False
            logger.info("Pretrained encoder is frozen. Encoder params will not update during training.")
        
        if getattr(args, "reset_parameter_state", False):
            for k, p in encoder.named_parameters():
                p.requires_grad = True
            logger.info("Encoder parameters are live.")

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        
        if getattr(args, "freeze_pretrained_decoder", False):
            for k, p in decoder.named_parameters():
                p.requires_grad = False
            logger.info("Pretrained decoder is frozen. Decoder params will not update during training.")
        
        if getattr(args, "reset_parameter_state", False):
            for k, p in decoder.named_parameters():
                p.requires_grad = True
            logger.info("Decoder parameters are live.")

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(task.target_dictionary, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(self, src_tokens, src_lengths, prev_output_tokens, train_threshold=None):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            train_threshold=train_threshold,
        )
        return decoder_out


class ConvTransformerEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args):
        """Construct an Encoder object."""
        super().__init__(None)

        self.dropout = args.dropout
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, args.conv_out_channels, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                args.conv_out_channels,
                args.conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(self.in_channels, self.input_dim, args.conv_out_channels)
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.uni_encoder = getattr(args, "unidirectional_encoder", False)
        self.uni_encoder = True

        self.pre_decision_ratio = getattr(args, "fixed_pre_decision_ratio", 7)

        seq_len = 6000
        if self.uni_encoder:
            tmp = (
                torch.arange(1, math.ceil(seq_len / self.pre_decision_ratio) + 1, device="cuda")
                .unsqueeze(1)
                .repeat(1, self.pre_decision_ratio)
                .contiguous()
                .view(-1, 1)[:seq_len]
            )
            block_mask = torch.arange(0, seq_len, device="cuda").unsqueeze(0).repeat(seq_len, 1).float()
            block_mask = block_mask.masked_fill((block_mask <= (tmp * self.pre_decision_ratio - 1)), 0.0)
            self.block_mask = block_mask.masked_fill((block_mask > (tmp * self.pre_decision_ratio - 1)), float("-inf"))

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend([TransformerEncoderLayer(args) for i in range(args.encoder_layers)])
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def pooling_ratio(self):
        return 4

    def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def forward(self, src_tokens, src_lengths):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim).transpose(1, 2).contiguous()

        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(input_len_0.device)
        input_lengths = torch.min(input_len_0, input_len_1)

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.uni_encoder:
            seq_len, _, _ = x.size()
            attn_mask = self.block_mask[:seq_len, :seq_len].to(x.device)

        for layer in self.transformer_layers:
            x = layer(
                x,
                encoder_padding_mask,
                attn_mask=attn_mask if self.uni_encoder else None,
            )

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask

        return {
            "encoder_out": [x],
            "encoder_padding_mask": [maybe_encoder_padding_mask] if maybe_encoder_padding_mask is not None else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [(encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [(encoder_out["encoder_embedding"][0]).index_select(0, new_order)]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


################### rewriting old decoder here ###################
# class TransformerDecoder(FairseqIncrementalDecoder):
#     """
#     Transformer decoder consisting of *args.decoder_layers* layers. Each layer
#     is a :class:`TransformerDecoderLayer`.

#     Args:
#         args (argparse.Namespace): parsed command-line arguments
#         dictionary (~fairseq.data.Dictionary): decoding dictionary
#         embed_tokens (torch.nn.Embedding): output embedding
#         no_encoder_attn (bool, optional): whether to attend to encoder outputs
#             (default: False).
#     """

#     def __init__(
#         self,
#         args,
#         dictionary,
#         embed_tokens,
#         no_encoder_attn=False,
#         output_projection=None,
#     ):
#         self.args = args
#         super().__init__(dictionary)
#         self.register_buffer("version", torch.Tensor([3]))
#         self._future_mask = torch.empty(0)

#         self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
#         self.decoder_layerdrop = args.decoder_layerdrop
#         self.share_input_output_embed = args.share_decoder_input_output_embed

#         input_embed_dim = embed_tokens.embedding_dim
#         embed_dim = args.decoder_embed_dim
#         self.embed_dim = embed_dim
#         self.output_embed_dim = args.decoder_output_dim

#         self.padding_idx = embed_tokens.padding_idx
#         self.max_target_positions = args.max_target_positions

#         self.embed_tokens = embed_tokens

#         self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

#         if not args.adaptive_input and args.quant_noise_pq > 0:
#             self.quant_noise = apply_quant_noise_(
#                 nn.Linear(embed_dim, embed_dim, bias=False),
#                 args.quant_noise_pq,
#                 args.quant_noise_pq_block_size,
#             )
#         else:
#             self.quant_noise = None

#         self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
#         self.embed_positions = (
#             PositionalEmbedding(
#                 self.max_target_positions,
#                 embed_dim,
#                 self.padding_idx,
#                 learned=args.decoder_learned_pos,
#             )
#             if not args.no_token_positional_embeddings
#             else None
#         )

#         if getattr(args, "layernorm_embedding", False):
#             self.layernorm_embedding = LayerNorm(embed_dim)
#         else:
#             self.layernorm_embedding = None

#         self.cross_self_attention = getattr(args, "cross_self_attention", False)

#         if self.decoder_layerdrop > 0.0:
#             self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
#         else:
#             self.layers = nn.ModuleList([])
#         self.layers.extend([self.build_decoder_layer(args, no_encoder_attn) for _ in range(args.decoder_layers)])
#         self.num_layers = len(self.layers)

#         if args.decoder_normalize_before and not getattr(args, "no_decoder_final_norm", False):
#             self.layer_norm = LayerNorm(embed_dim)
#         else:
#             self.layer_norm = None

#         self.project_out_dim = (
#             Linear(embed_dim, self.output_embed_dim, bias=False)
#             if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
#             else None
#         )

#         self.adaptive_softmax = None
#         self.output_projection = output_projection
#         if self.output_projection is None:
#             self.build_output_projection(args, dictionary, embed_tokens)

#     def build_output_projection(self, args, dictionary, embed_tokens):
#         if args.adaptive_softmax_cutoff is not None:
#             self.adaptive_softmax = AdaptiveSoftmax(
#                 len(dictionary),
#                 self.output_embed_dim,
#                 utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
#                 dropout=args.adaptive_softmax_dropout,
#                 adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
#                 factor=args.adaptive_softmax_factor,
#                 tie_proj=args.tie_adaptive_proj,
#             )
#         elif self.share_input_output_embed:
#             self.output_projection = nn.Linear(
#                 self.embed_tokens.weight.shape[1],
#                 self.embed_tokens.weight.shape[0],
#                 bias=False,
#             )
#             self.output_projection.weight = self.embed_tokens.weight
#         else:
#             self.output_projection = nn.Linear(self.output_embed_dim, len(dictionary), bias=False)
#             nn.init.normal_(self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5)
#         num_base_layers = getattr(args, "base_layers", 0)
#         for i in range(num_base_layers):
#             self.layers.insert(
#                 ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
#                 BaseLayer(args),
#             )

#     def build_decoder_layer(self, args, no_encoder_attn=False):
#         layer = TransformerDecoderLayer(args, no_encoder_attn)
#         checkpoint = getattr(args, "checkpoint_activations", False)
#         if checkpoint:
#             offload_to_cpu = getattr(args, "offload_activations", False)
#             layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
#         # if we are checkpointing, enforce that FSDP always wraps the
#         # checkpointed layer, regardless of layer size
#         min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP) if not checkpoint else 0
#         layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
#         return layer

#     def forward(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Dict[str, List[Tensor]]] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         features_only: bool = False,
#         full_context_alignment: bool = False,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#         src_lengths: Optional[Any] = None,
#         return_all_hiddens: bool = False,
#         step=None,
#         train_threshold=None,
#     ):
#         """
#         Args:
#             prev_output_tokens (LongTensor): previous decoder outputs of shape
#                 `(batch, tgt_len)`, for teacher forcing
#             encoder_out (optional): output from the encoder, used for
#                 encoder-side attention, should be of size T x B x C
#             incremental_state (dict): dictionary used for storing state during
#                 :ref:`Incremental decoding`
#             features_only (bool, optional): only return features without
#                 applying output layer (default: False).
#             full_context_alignment (bool, optional): don't apply
#                 auto-regressive mask to self-attention (default: False).

#         Returns:
#             tuple:
#                 - the decoder's output of shape `(batch, tgt_len, vocab)`
#                 - a dictionary with any model-specific outputs
#         """

#         x, extra = self.extract_features(
#             prev_output_tokens,
#             encoder_out=encoder_out,
#             incremental_state=incremental_state,
#             full_context_alignment=full_context_alignment,
#             alignment_layer=alignment_layer,
#             alignment_heads=alignment_heads,
#             train_threshold=train_threshold,
#             step=step,
#         )

#         if not features_only:
#             x = self.output_layer(x)
#         return x, extra

#     def extract_features(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Dict[str, List[Tensor]]],
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         full_context_alignment: bool = False,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#         train_threshold=None,
#         step=None,
#     ):
#         return self.extract_features_scriptable(
#             prev_output_tokens,
#             encoder_out,
#             incremental_state,
#             full_context_alignment,
#             alignment_layer,
#             alignment_heads,
#             train_threshold,
#         )

#     """
#     A scriptable subclass of this class has an extract_features method and calls
#     super().extract_features, but super() is not supported in torchscript. A copy of
#     this function is made to be used in the subclass instead.
#     """

#     def extract_features_scriptable(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Dict[str, List[Tensor]]],
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         full_context_alignment: bool = False,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#         train_threshold=None,
#     ):
#         """
#         Similar to *forward* but only return features.

#         Includes several features from "Jointly Learning to Align and
#         Translate with Transformer Models" (Garg et al., EMNLP 2019).

#         Args:
#             full_context_alignment (bool, optional): don't apply
#                 auto-regressive mask to self-attention (default: False).
#             alignment_layer (int, optional): return mean alignment over
#                 heads at this layer (default: last layer).
#             alignment_heads (int, optional): only average alignment over
#                 this many heads (default: all heads).

#         Returns:
#             tuple:
#                 - the decoder's features of shape `(batch, tgt_len, embed_dim)`
#                 - a dictionary with any model-specific outputs
#         """
#         bs, slen = prev_output_tokens.size()
#         if alignment_layer is None:
#             alignment_layer = self.num_layers - 1

#         enc: Optional[Tensor] = None
#         padding_mask: Optional[Tensor] = None
#         if encoder_out is not None:
#             enc = encoder_out["encoder_out"][0]
#             padding_mask = encoder_out["encoder_padding_mask"][0]
#             assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"

#         # embed positions
#         positions = None
#         if self.embed_positions is not None:
#             positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state)

#         if incremental_state is not None:
#             prev_output_tokens = prev_output_tokens[:, -1:]
#             if positions is not None:
#                 positions = positions[:, -1:]

#         # embed tokens and positions
#         x = self.embed_scale * self.embed_tokens(prev_output_tokens)

#         if self.quant_noise is not None:
#             x = self.quant_noise(x)

#         if self.project_in_dim is not None:
#             x = self.project_in_dim(x)

#         if positions is not None:
#             x += positions

#         if self.layernorm_embedding is not None:
#             x = self.layernorm_embedding(x)

#         x = self.dropout_module(x)

#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)

#         self_attn_padding_mask: Optional[Tensor] = None
#         if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
#             self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

#         # decoder layers
#         attn: Optional[Tensor] = None
#         inner_states: List[Optional[Tensor]] = [x]
#         for idx, layer in enumerate(self.layers):
#             if incremental_state is None and not full_context_alignment:
#                 self_attn_mask = self.buffered_future_mask(x)
#             else:
#                 self_attn_mask = None

#             x, layer_attn, _ = layer(
#                 x,
#                 enc,
#                 padding_mask,
#                 incremental_state,
#                 self_attn_mask=self_attn_mask,
#                 self_attn_padding_mask=self_attn_padding_mask,
#                 need_attn=bool((idx == alignment_layer)),
#                 need_head_weights=bool((idx == alignment_layer)),
#             )
#             inner_states.append(x)
#             if layer_attn is not None and idx == alignment_layer:
#                 attn = layer_attn.float().to(x)

#         if attn is not None:
#             if alignment_heads is not None:
#                 attn = attn[:alignment_heads]

#             # average probabilities over heads
#             attn = attn.mean(dim=0)

#         if self.layer_norm is not None:
#             x = self.layer_norm(x)

#         # T x B x C -> B x T x C
#         x = x.transpose(0, 1)

#         if self.project_out_dim is not None:
#             x = self.project_out_dim(x)

#         return x, {"attn": [attn], "inner_states": inner_states}

#     def output_layer(self, features):
#         """Project features to the vocabulary size."""
#         if self.adaptive_softmax is None:
#             # project back to size of vocabulary
#             return self.output_projection(features)
#         else:
#             return features

#     def max_positions(self):
#         """Maximum output length supported by the decoder."""
#         if self.embed_positions is None:
#             return self.max_target_positions
#         return min(self.max_target_positions, self.embed_positions.max_positions)

#     def buffered_future_mask(self, tensor):
#         dim = tensor.size(0)
#         # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
#         if (
#             self._future_mask.size(0) == 0
#             or (not self._future_mask.device == tensor.device)
#             or self._future_mask.size(0) < dim
#         ):
#             self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1)
#         self._future_mask = self._future_mask.to(tensor)
#         return self._future_mask[:dim, :dim]

#     def upgrade_state_dict_named(self, state_dict, name):
#         """Upgrade a (possibly old) state dict for new versions of fairseq."""
#         if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
#             weights_key = "{}.embed_positions.weights".format(name)
#             if weights_key in state_dict:
#                 del state_dict[weights_key]
#             state_dict["{}.embed_positions._float_tensor".format(name)] = torch.FloatTensor(1)

#         if f"{name}.output_projection.weight" not in state_dict:
#             if self.share_input_output_embed:
#                 embed_out_key = f"{name}.embed_tokens.weight"
#             else:
#                 embed_out_key = f"{name}.embed_out"
#             if embed_out_key in state_dict:
#                 state_dict[f"{name}.output_projection.weight"] = state_dict[embed_out_key]
#                 if not self.share_input_output_embed:
#                     del state_dict[embed_out_key]

#         for i in range(self.num_layers):
#             # update layer norms
#             layer_norm_map = {
#                 "0": "self_attn_layer_norm",
#                 "1": "encoder_attn_layer_norm",
#                 "2": "final_layer_norm",
#             }
#             for old, new in layer_norm_map.items():
#                 for m in ("weight", "bias"):
#                     k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
#                     if k in state_dict:
#                         state_dict["{}.layers.{}.{}.{}".format(name, i, new, m)] = state_dict[k]
#                         del state_dict[k]

#         version_key = "{}.version".format(name)
#         if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
#             # earlier checkpoints did not normalize after the stack of layers
#             self.layer_norm = None
#             self.normalize = False
#             state_dict[version_key] = torch.Tensor([1])

#         return state_dict


##################################################################


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        step=None,
        train_threshold=None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


@register_model_architecture("convtransformer", "convtransformer_espnet")
def convtransformer_espnet(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
