import torch
import itertools
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable


class TransformerEncoderLayerInterpret(nn.TransformerEncoderLayer):
    '''
    Overloaded version of the encoder layer s.t. we can extract self-attention
    '''
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerInterpret, self).__init__(*args, **kwargs)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            src_mask is None and
                not (src.is_nested and src_key_padding_mask is not None)):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,  # TODO: split into two args
                )
        x = src
        attn_list = []
        if self.norm_first:
            sa_add, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            attn_list.append(attn)
            x = x + sa_add
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_add, attn = self._sa_block(x, src_mask, src_key_padding_mask)
            attn_list.append(attn)
            x = self.norm1(x + sa_add)
            x = self.norm2(x + self._ff_block(x))

        return x, attn_list

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # Modified to output attention weights
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attn_weights

class TransformerEncoderInterpret(nn.TransformerEncoder):
    r'''
    Modified version of Transformer Encoder s.t. we can extract self-attention
    '''

    def __init__(self, *args, **kwargs):
        super(TransformerEncoderInterpret, self).__init__(*args, **kwargs)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        # Add check for custom Transformer Encoder layer:
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer) or isinstance(first_layer, TransformerEncoderLayerInterpret):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        attn_per_layer = []

        for mod in self.layers:
            if convert_to_nested:
                output, attn = mod(output, src_mask=mask)
            else:
                output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_per_layer.append(attn)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_per_layer