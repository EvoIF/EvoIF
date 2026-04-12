import os
import argparse
import logging
import numpy as np
from Bio.PDB import PDBParser

STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

class AlignmentError(Exception):
    pass

def parse_pdb_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, resnum, _ = residue.get_id()
                if hetflag == " " and residue.get_resname() in STANDARD_AMINO_ACIDS:
                    residue_ids.append(resnum)
                    
    if not residue_ids:
        raise ValueError("No standard amino acids found.")
    return sorted(residue_ids)

def align_and_extract(npz_data, pdb_residues):
    min_id, max_id = min(pdb_residues), max(pdb_residues)
    total_expected = max_id - min_id + 1
    npz_length = npz_data['log_p'].shape[1]
    
    if npz_length != total_expected:
        raise AlignmentError(f"Length mismatch: NPZ={npz_length}, Expected={total_expected}")
        
    valid_indices = [res_id - min_id for res_id in pdb_residues]
    
    if min(valid_indices) < 0 or max(valid_indices) >= npz_length:
        raise AlignmentError("Index out of bounds.")
        
    new_data = {}
    for key in npz_data.files:
        arr = npz_data[key]
        if key == 'log_p':  
            new_arr = arr[:, valid_indices, :]
        elif arr.ndim == 1 and len(arr) == npz_length: 
            new_arr = arr[valid_indices]
        else:  
            new_arr = arr
        # The crucial bug fix: saving the sliced array, not the original one
        new_data[key] = new_arr
        
    if new_data['log_p'].shape[1] != len(pdb_residues):
        raise AlignmentError("Output dimension mismatch.")
        
    return new_data

def process_file(pdb_path, npz_path, out_path):
    try:
        pdb_residues = parse_pdb_residues(pdb_path)
        npz_data = np.load(npz_path)
        new_data = align_and_extract(npz_data, pdb_residues)
        
        np.savez(out_path, **new_data)
        npz_data.close()
        return True
    except Exception as e:
        logging.error(f"Failed {os.path.basename(pdb_path)}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Align ProteinMPNN NPZ outputs with PDB structures.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing input PDB files.")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing original NPZ files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save aligned NPZ files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdb_files = sorted([f for f in os.listdir(args.pdb_dir) if f.endswith(".pdb")])
    if not pdb_files:
        logging.warning("No PDB files found in the specified directory.")
        return

    success_count = 0
    for pdb_file in pdb_files:
        base_name = pdb_file[:-4]
        pdb_path = os.path.join(args.pdb_dir, pdb_file)
        npz_path = os.path.join(args.npz_dir, f"{base_name}.npz")
        out_path = os.path.join(args.out_dir, f"{base_name}.npz")

        if os.path.exists(out_path):
            logging.info(f"Skipping {base_name}: already exists.")
            continue

        if not os.path.exists(npz_path):
            logging.error(f"Missing NPZ for {base_name}.")
            continue

        if process_file(pdb_path, npz_path, out_path):
            success_count += 1
            logging.info(f"Processed {base_name} successfully.")

    logging.info(f"Done. Processed {success_count}/{len(pdb_files)} files.")

if __name__ == "__main__":
    main()