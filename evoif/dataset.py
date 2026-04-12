import os
import csv
import math
import pickle
import random

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import normalize
from Bio.PDB.PDBParser import PDBParser

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils, core
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.data.dataloader import graph_collate
from evoif import residue_constants

import re


def bio_load_pdb(pdb):
    """
    Load a raw PDB file using Biopython and convert it into a TorchDrug Protein object.
    
    Args:
        pdb (str): Path to the PDB file.
        
    Returns:
        tuple: (torchdrug.data.Protein, str) The constructed protein graph and its sequence string.
    """
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb)
    residues = [residue for residue in protein.get_residues()]
    residue_type = [data.Protein.residue2id.get(residue.get_resname(), 0) for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}
    residue_feature = functional.one_hot(torch.as_tensor(residue_type), len(data.Protein.residue2id)+1)

    atoms = [atom for atom in protein.get_atoms()]
    atoms = [atom for atom in atoms if atom.get_name() in data.Protein.atom_name2id]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name, 
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor,
                residue_number=residue_number, node_position=node_position, residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])


def seq_to_tensor(seq, format):
    """
    Convert a sequence string into a tensor of vocabulary indices.
    """
    if format == "a3m":
        seq = re.sub(r"[a-z]", "", seq)
    seq = seq.upper().replace('.', '-')
    symbol2id_pad = data.Protein.residue_symbol2id.copy()
    symbol2id_pad.update({'-': 20})
    return torch.tensor([symbol2id_pad.get(res, 21) for res in seq])


def read_multi_seqs(file_path, length, max_seqs=2048, format="fasta"):
    """
    Read Multiple Sequence Alignments (MSA) from a FASTA or A3M file.
    
    Args:
        file_path (str): Path to the alignment file.
        length (int): Expected sequence length.
        max_seqs (int): Maximum number of sequences to sample. Defaults to 2048.
        format (str): File format ("fasta" or "a3m").
        
    Returns:
        torch.Tensor: A tensor of aligned sequences of shape (length, num_seqs).
    """
    sequences = []
    current_sequence = ''
    if not os.path.exists(file_path):
        return torch.ones((length, 0), dtype=torch.long) * 21

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(seq_to_tensor(current_sequence, format))
                    current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(seq_to_tensor(current_sequence, format))
            
    if max_seqs is not None and len(sequences) > max_seqs:
        sequences = random.sample(sequences, max_seqs)
    if len(sequences) == 0:
        return torch.ones((length, 0), dtype=torch.long) * 21
    else:
        return torch.stack(sequences).transpose(0, 1)


@R.register("datasets.ProteinGym")
class ProteinGym(core.Configurable):
    """
    ProteinGym dataset wrapper for loading deep mutational scanning (DMS) assays.
    """

    def __init__(self, path, csv_file):
        path = os.path.expanduser(path)
        self.path = path
        csv_file = os.path.join(path, csv_file)

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            assay_list = [row for row in reader]
        self.ids = [assay["DMS_id"] for assay in assay_list]
        self.assay_dict = {assay["DMS_id"]: assay for assay in assay_list}


def custom_collate_fn(batch):
    """
    Custom collate function to process tensors of different sizes using padding and masking.
    Regular graph data is collated using TorchDrug's graph_collate, while auxiliary 
    profiles are padded dynamically to the batch maximums.
    """
    pad_keys = ['if_profile', 'struc_profile']
    stackable_items = [{k:v for k,v in item.items() if k not in pad_keys} for item in batch]
    pad_items = [{k:v for k,v in item.items() if k in pad_keys} for item in batch]
    result = graph_collate(stackable_items)
    
    for key in pad_keys:
        if key not in pad_items[0]:
            continue
        values = [item[key] for item in pad_items if key in item and item[key] is not None]
        if not values:
            continue 
        max_length = max(v.shape[0] for v in values)
        max_num_seqs = max(v.shape[1] for v in values)
        padded_tensors = []
        masks = []
        
        for v in values:
            seq_len, num_seqs = v.shape[0], v.shape[1]
            if num_seqs == 1 and (v == 21).all():
                num_seqs = 0
            padded = torch.ones(seq_len, max_num_seqs, dtype=v.dtype, device=v.device) * 21 # 21 is padding_index
            padded[:, :num_seqs] = v
            mask = torch.zeros(1, max_num_seqs, dtype=torch.bool, device=v.device)
            mask[:, :num_seqs] = True
            padded_tensors.append(padded)
            masks.append(mask)
        
        if padded_tensors and all(t is not None for t in padded_tensors):
            result[key] = torch.cat(padded_tensors, dim=0) 
            result[f"{key}_mask"] = torch.cat(masks, dim=0)
        else:
            result[key] = padded_tensors
    
    return result


class MutantDataset(torch_data.Dataset):
    """
    Dataset that pairs mutated protein sequences with their wild-type structures
    and auxiliary evolutionary/structural profiles.
    """

    def __init__(self, mutated_sequences, wild_type, if_profile=None, struc_profile=None, transform=None, embeddings=None):
        self.mutated_sequences = mutated_sequences
        self.wild_type = wild_type
        self.if_profile = if_profile       # Inverse folding profile
        self.struc_profile = struc_profile # Structure profile distributions
        self.transform = transform
        self.embeddings = embeddings

    def __len__(self):
        return len(self.mutated_sequences)
 
    def truncate(self, sequence_graph, structure_graph, if_profile=None, struc_profile=None):
        """Truncate multi-modal features to match a specific subsequence window."""
        num_residue = structure_graph.num_residue
        start = sequence_graph.start - structure_graph.start
        end = sequence_graph.end - structure_graph.start
        
        residue_mask = torch.zeros((num_residue, ), dtype=torch.bool)
        residue_mask[start:end] = 1
        structure_graph = structure_graph.subresidue(residue_mask)

        if if_profile is not None:
            if_profile = if_profile[start:end]
        if struc_profile is not None:
            struc_profile = struc_profile[start:end]

        return sequence_graph, structure_graph, if_profile, struc_profile

    def assign_structure(self, sequence_graph, structure_graph):
        """Map the mutated sequence onto the wild-type backbone structure."""
        graph = structure_graph.clone()
        assert graph.num_residue == sequence_graph.num_residue
        with graph.residue():
            graph.residue_type = sequence_graph.residue_type
        return graph

    def __getitem__(self, index):
        sequence_graph = self.mutated_sequences[index]
        structure_graph = self.wild_type
        if_profile = self.if_profile
        struc_profile = self.struc_profile

        # Truncate structure and profiles if the WT structure is longer than the mutated sequence snippet
        if structure_graph.start <= sequence_graph.start and structure_graph.end >= sequence_graph.end:
            sequence_graph, structure_graph, if_profile, struc_profile = self.truncate(
                sequence_graph, structure_graph, if_profile, struc_profile
            )
        elif structure_graph.start != sequence_graph.start or structure_graph.end != sequence_graph.end:
            raise ValueError("The structure range (%d, %d) doesn't match the sequence range (%d, %d)" % 
                             (structure_graph.start, structure_graph.end, sequence_graph.start, sequence_graph.end))
        
        graph = self.assign_structure(sequence_graph, structure_graph)
        
        item = {"graph": graph}
        if if_profile is not None:
            item["if_profile"] = if_profile
        if struc_profile is not None:
            item["struc_profile"] = struc_profile
            
        if self.transform:
            item = self.transform(item)
        return item
    
    
    
# Pre-training dataset utilities
atom_type_mapping = torch.tensor([data.feature.atom_vocab[n[0]] for n in residue_constants.atom_order])     # (37, )
atom_name_mapping = torch.tensor([data.Protein.atom_name2id[n] for n in residue_constants.atom_order])      # (37, )
inv_atom_name_mapping = torch.zeros((len(data.Protein.atom_name2id)), dtype=torch.long)
inv_atom_name_mapping[atom_name_mapping] = torch.arange(residue_constants.atom_type_num, dtype=torch.long)      # (37, )
residue_type_mapping = torch.tensor([data.Protein.residue_symbol2id.get(n, 0) for n in residue_constants.restypes_with_x])    # (21, )


def load_protein(data_dict):
    """
    Reconstruct a TorchDrug Protein object from a pickled dictionary 
    containing raw atom and residue features.
    """
    atom_mask = torch.tensor(data_dict['atom_mask']).bool()
    atom_type = atom_type_mapping[None, :]
    atom_type = atom_type.expand_as(atom_mask)[atom_mask]
    atom_name = atom_name_mapping[None, :]
    atom_name = atom_name.expand_as(atom_mask)[atom_mask]
    node_position = torch.tensor(data_dict['atom_positions'])[atom_mask]
    residue_type = torch.tensor(data_dict['aatype'])
    residue_type = residue_type_mapping[residue_type]
    residue_number = torch.tensor(data_dict['residue_index'])
    b_factor = torch.tensor(data_dict['b_factors'])[atom_mask]
    chain_id = torch.tensor(data_dict['chain_index'])
    num_residue = residue_type.shape[0]
    num_atom = atom_name.shape[0]

    atom2residue = torch.arange(num_residue)[:, None]
    atom2residue = atom2residue.expand_as(atom_mask)[atom_mask]
    edge_list = torch.zeros((1, 3), dtype=torch.long)
    bond_type = torch.zeros((1,), dtype=torch.long)

    residue_feature = F.one_hot(residue_type, len(residue_constants.restypes_with_x))
    atom_feature = torch.cat([
        F.one_hot(atom_name, residue_constants.atom_type_num),
        residue_feature[atom2residue]
    ], dim=-1)
    
    protein = data.Protein(edge_list=edge_list, atom_type=atom_type, bond_type=bond_type, 
                        residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue, 
                        residue_feature=residue_feature, atom_feature=atom_feature, bond_feature=None,
                        residue_number=residue_number, b_factor=b_factor, chain_id=chain_id,
                        node_position=node_position, num_node=num_atom, num_residue=num_residue,
    )
    # Reconstruct the sequence string (0-20 -> ACDEF...X-)
    seq_str = ''.join([residue_constants.restypes_with_x[i] for i in data_dict['aatype']])
    protein.seq_str = seq_str 
    return protein

@R.register("datasets.CATH")
class CATH(data.ProteinDataset):
    """
    CATH dataset for protein structure and sequence profiles.
    Supports on-the-fly loading from raw `.pkl` files or optimized preloaded directories.
    """

    def __init__(self, path, max_length=None, struc_align_path=None, if_profile_path=None, profile_types=['if_profile', 'struc_profile'], transform=None, preload=True, preloaded_path=None, load_graph=True):
        """
        Args:
            path (str): Path to CATH dataset.
            max_length (int, optional): Maximum sequence length for cropping.
            struc_align_path (str, optional): Path to structural alignments (Foldseek).
            if_profile_path (str, optional): Path to inverse folding logits/profiles.
            profile_types (list): Types of multi-modal features to load into the batch.
            transform (callable, optional): Data transformations to apply.
            preload (bool): Whether to read from preprocessed cache directories.
            preloaded_path (str, optional): Path to the preloaded cache.
            load_graph (bool): Whether to explicitly load the TorchDrug graph field.
        """
        self.preload = preload
        self.load_graph = load_graph
        path = os.path.expanduser(path)
        self.path = path
        self.max_length = max_length

        self.pkl_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pkl")])
        if struc_align_path:
            struc_align_path = os.path.expanduser(struc_align_path)
        self.struc_align_path = struc_align_path
        if if_profile_path: 
            if_profile_path = os.path.expanduser(if_profile_path)
        self.if_profile_path = if_profile_path
        self.profile_types = profile_types
        self.transform = transform
        
        if preload:
            if preloaded_path is None:
                raise ValueError("preloaded_path must be specified when preload=True")
            self.preloaded_path = os.path.expanduser(preloaded_path)
            # Retrieve all preloaded sample directories
            self.sample_dirs = []
            for item in os.listdir(self.preloaded_path):
                item_path = os.path.join(self.preloaded_path, item)
                if os.path.isdir(item_path) and item != "__pycache__":
                    self.sample_dirs.append(item)
            
            self.sample_dirs.sort()
            print(f"Found {len(self.sample_dirs)} preloaded samples")
            failed_file = os.path.join(self.preloaded_path, "failed_indices.pkl")
            if os.path.exists(failed_file):
                with open(failed_file, "rb") as f:
                    self.failed_indices = pickle.load(f)
                print(f"Found {len(self.failed_indices)} failed files during preloading")
            else:
                self.failed_indices = []

            self.preload_types = self.profile_types + ['graph']

    def truncate(self, length, data_dict, profile_dict=None):
        """Randomly crop sequences and structures if they exceed max_length."""
        if length <= self.max_length:
            return data_dict, profile_dict
            
        start = np.random.randint(length - self.max_length, size=(1,))[0]
        end = start + self.max_length
        for k in data_dict.keys():
            data_dict[k] = data_dict[k][start:end]
        if profile_dict is not None:
            for k in profile_dict.keys():
                profile_dict[k] = profile_dict[k][start:end]

        return data_dict, profile_dict

    def get_item(self, idx):
        if self.preload:
            return self._get_preloaded_item(idx)
        else:
            return self._get_original_item(idx)
    
    def _get_preloaded_item(self, idx):
        """Retrieve a specific sample item from the preprocessed cache."""
        try:
            sample_dir = self.sample_dirs[idx]
            sample_path = os.path.join(self.preloaded_path, sample_dir)
            item = {}
            for item_type in self.preload_types:
                profile_file = os.path.join(sample_path, f"{sample_dir}_{item_type}.pkl")
                if os.path.exists(profile_file):
                    with open(profile_file, "rb") as f:
                        item[item_type] = pickle.load(f)
                else:
                    print(f"Warning: field {item_type} not found in sample {profile_file}")
                    raise ValueError(f"field {item_type} not found in sample {profile_file}")

            if self.max_length:
                item, _ = self.truncate(item['graph'].num_residue, item)
            if self.transform:
                item = self.transform(item)
            
            return item
            
        except Exception as e:
            print(f"Error reading preloaded sample {self.sample_dirs[idx]}: {e}")
            # Fallback recursively to the next sample upon failure
            if idx + 1 < len(self):
                return self._get_preloaded_item(idx + 1)
            else:
                return self._get_preloaded_item(0)
    
    def _get_original_item(self, idx):
        """Retrieve, process, and align a sample dynamically from raw source files."""
        try:
            with open(self.pkl_files[idx], "rb") as fin:
                data_dict = pickle.load(fin)
                
            item = {}
                    
            # Process Structural Alignments (Foldseek)
            if self.struc_align_path and ('struc_align' in self.profile_types or 'struc_profile' in self.profile_types):
                struc_align_file = os.path.join(self.struc_align_path, os.path.basename(self.pkl_files[idx]).split('.')[0]+'.fasta' )
                struc_align = read_multi_seqs(struc_align_file, length=data_dict["aatype"].shape[0], format="fasta")
                
                # Generate structure token distribution profile (shape: [L, 20])
                L, num_seqs = struc_align.shape
                one_hot = F.one_hot(struc_align, num_classes=22)
                struc_align_count = one_hot.sum(dim=1)
                struc_profile = struc_align_count[:, :20] / (struc_align_count[:, :20].sum(dim=-1, keepdim=True) + 1e-8)
                
                if struc_align.shape[0] != data_dict["aatype"].shape[0]:
                    raise ValueError(f"struc_align length {struc_align.shape[0]} doesn't match data_dict length {data_dict['aatype'].shape[0]}")
                if 'struc_align' in self.profile_types:
                    item['struc_align'] = struc_align
                if 'struc_profile' in self.profile_types:
                    item['struc_profile'] = struc_profile
                    
            # Process Inverse Folding Profiles
            if self.if_profile_path and ('if_profile' in self.profile_types):
                if_profile_file = os.path.join(self.if_profile_path, os.path.basename(self.pkl_files[idx]).split('.')[0]+'.npz' )
                if_profile = np.load(if_profile_file)
                if_profile = torch.softmax(torch.tensor(if_profile["log_p"][0][...,:20]), dim=-1)

                residue_index = data_dict['residue_index']
                profile_indices = residue_index - residue_index.min()
                mask_value = 0.0  # Zero probabilities for missing data
                seq_len, num_features = if_profile.shape
                new_if_profile = torch.full((len(residue_index), num_features), mask_value, dtype=if_profile.dtype, device=if_profile.device)
                valid_mask = (profile_indices >= 0) & (profile_indices < seq_len)
                new_if_profile[valid_mask] = if_profile[profile_indices[valid_mask]]
                if_profile = new_if_profile 
                if 'if_profile' in self.profile_types:
                    item['if_profile'] = if_profile
                    
            # Apply truncations if sequence exceeds max_length
            if self.max_length:
                data_dict, item = self.truncate(data_dict["aatype"].shape[0], data_dict, profile_dict=item)
            protein = load_protein(data_dict)
            item["graph"] = protein

            if self.transform:
                item = self.transform(item)
            return item
            
        except Exception as e:
            print(f"Error in get_item: {e}, file: {self.pkl_files[idx]}")
            return None

    def __len__(self):
        if self.preload:
            return len(self.sample_dirs)
        else:
            return len(self.pkl_files)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
        ]
        if self.preload:
            lines.append(f"preloaded_path: {self.preloaded_path}")
            lines.append(f"load_graph: {self.load_graph}")
            if self.profile_types:
                lines.append(f"profile_types: {self.profile_types}")
        else:
            lines.append(f"path: {self.path}")
            if self.max_length:
                lines.append(f"max_length: {self.max_length}")
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))