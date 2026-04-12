import warnings
from typing import Optional, List, Dict, Any

import torch
from torch import nn

from torchdrug import core, models
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from evoif import gvp

@R.register("models.MyESM")
class MyESM(models.EvolutionaryScaleModeling):
    """
    Custom wrapper for Evolutionary Scale Modeling (ESM).
    Extracts residue-level and graph-level features from protein sequences.
    """

    def forward(self, graph, input, all_loss=None, metric=None) -> Dict[str, torch.Tensor]:
        """
        Compute the residue representations and the graph representation(s).

        Args:
            graph (Protein): A batch of `n` protein graphs.
            input (torch.Tensor): Input node representations.
            all_loss (torch.Tensor, optional): Tensor to accumulate loss.
            metric (dict, optional): Dictionary to record metrics.

        Returns:
            dict: Containing 'graph_feature', 'residue_feature', and 'logits'.
        """
        input = graph.residue_type
        input = self.mapping[input]
        input[input == -1] = graph.residue_type[input == -1]
        size = graph.num_residues
        
        if (size > self.max_input_length).any():
            warnings.warn(
                f"ESM can only encode proteins within {self.max_input_length} residues. "
                "Truncating the input to fit into ESM memory limits."
            )
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
            
        extended_size = size
        if self.alphabet.prepend_bos:
            bos_token = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, extended_size = functional._extend(bos_token, torch.ones_like(extended_size), input, extended_size)
        if self.alphabet.append_eos:
            eos_token = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, extended_size = functional._extend(input, extended_size, eos_token, torch.ones_like(extended_size))
            
        input = functional.variadic_to_padded(input, extended_size, value=self.alphabet.padding_idx)[0]
        output = self.model(input, repr_layers=[self.repr_layer])
        residue_feature = output["representations"][self.repr_layer]
        logits = output["logits"]
        residue_feature = functional.padded_to_variadic(residue_feature, extended_size)
        logits = functional.padded_to_variadic(logits, extended_size)
        starts = extended_size.cumsum(0) - extended_size
        
        if self.alphabet.prepend_bos:
            starts = starts + 1   
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        logits = logits[mask]
        residue_type_index = torch.arange(20, dtype=torch.long, device=logits.device)
        logits = logits[:, self.mapping[residue_type_index]]
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
            "logits": logits
        }


class Transition(nn.Module):
    """
    A two-layer Multi-Layer Perceptron (MLP) with SiLU activation 
    and LayerNorm, used for feature transitions and refinements.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim)
        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim)
        """
        x = self.norm(x)
        x = self.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


  

@R.register("models.LightProfileModule")
class LightProfileModule(nn.Module, core.Configurable):
    """Light profile encoding module without pair representation."""
    
    def __init__(
        self,
        profile_type: str,
        profile_out_dim: int = 1280,
        num_amino_acids: int = 20,
        profile_dropout: float = 0.1,
        **kwargs,

    ):
        super().__init__()
        self.num_amino_acids = num_amino_acids
        self.profile_out_dim = profile_out_dim
        self.profile_type = profile_type
        self.hidden= Transition(dim=profile_out_dim, hidden=profile_out_dim)
        self.context_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=profile_out_dim,
                        nhead=8,
                        dim_feedforward=2*profile_out_dim,
                        batch_first=True,
                        dropout=profile_dropout
                    ),
                    num_layers=2
                    )
        self.transition_logits= Transition(dim=profile_out_dim, hidden=profile_out_dim,out_dim=20)
        self.output_norm = nn.LayerNorm(20)
                
    def forward(self, graph, node_feature, batch, embed_tokens, mapping, alphabet):
        """Forward pass for light profile module."""
        profile_type = self.profile_type
        device = node_feature.device
        size = graph.num_residues
        if profile_type not in batch:
            raise ValueError(f"Profile type '{profile_type}' not found in batch.")
     
        profile = batch[profile_type]  # shape: (|V_{res}|, profile_dim)
        profile_feature = torch.einsum('na,ad->nd', profile, embed_tokens(mapping[:20]))  # shape(|V_{res}|, ESM embedding dimension)
        padded_feature, _ = functional.variadic_to_padded(profile_feature, size)
        context_feature = self.context_encoder(padded_feature)
        feature = functional.padded_to_variadic(context_feature, size)
        residue_feature = self.hidden(feature)
        logits = self.transition_logits(residue_feature)  
        logits = self.output_norm(logits)
        
        return {
            'logits':logits,
           
        }

@R.register("models.FusionNetwork")
class FusionNetwork(nn.Module, core.Configurable):
    """
    A fusion architecture integrating sequence, structure, and profile models.
    Combines representations to predict final node/graph features and logits.
    """

    def __init__(
        self, 
        sequence_model: nn.Module, 
        structure_model: nn.Module,
        profile_struc: Optional[nn.Module] = None,
        profile_if: Optional[nn.Module] = None,
        **kwargs
    ):
        super(FusionNetwork, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.output_dim = structure_model.output_dim
        self.profile_struc = profile_struc
        self.profile_if = profile_if
     
    def forward(self, graph, input, all_loss=None, metric=None, batch=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Fusion Network.
        """
        # Sequence model 
        seq_output = self.sequence_model(graph, input, all_loss, metric)
        node_output = seq_output["residue_feature"]
        sequence_logits = seq_output["logits"]
        # Profile modules
        if self.profile_struc is None or self.profile_if is None:
            raise ValueError("Both `profile_struc` and `profile_if` must be provided.")
            
        # structure profile module
        profile_struc = self.profile_struc(
            graph, node_output, batch, 
            embed_tokens=self.sequence_model.model.embed_tokens, 
            mapping=self.sequence_model.mapping, 
            alphabet=self.sequence_model.alphabet
        )
        
        # inverse folding profile module
        profile_if = self.profile_if(
            graph, node_output, batch, 
            embed_tokens=self.sequence_model.model.embed_tokens, 
            mapping=self.sequence_model.mapping, 
            alphabet=self.sequence_model.alphabet
        )
        
        # Structure model 
        struc_output = self.structure_model(graph, node_output, all_loss, metric)
        node_feature = struc_output["node_feature"]
        graph_feature = struc_output["graph_feature"] 
        
        # profile logit fusion
        profile_logits = profile_struc['logits'] + profile_if['logits']

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "profile_logits": profile_logits,
            "sequence_logits": sequence_logits   
        }
        
        
        
        
        
        
      