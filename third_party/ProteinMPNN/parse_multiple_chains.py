import argparse, os, json, glob, gzip
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
    x = np.array(x)
    if x.ndim == 0:
        x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

def parse_PDB_biounits(x, atoms=('N','CA','C'), chain=None):
    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    opener = gzip.open if x.endswith('.gz') else open
    with opener(x, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM","ATOM  ").replace("MSE","MET")
            if line[:4] != "ATOM":
                continue
            ch = line[21:22]
            if chain is not None and ch != chain:
                continue
            atom = line[12:16].strip()
            resi = line[17:20]
            resn = line[22:27].strip()
            if resn[-1].isalpha():
                resa, resn = resn[-1], int(resn[:-1])-1
            else:
                resa, resn = "", int(resn)-1
            x_c, y_c, z_c = (float(line[i:i+8]) for i in [30,38,46])

            min_resn = min(min_resn, resn)
            max_resn = max(max_resn, resn)
            xyz.setdefault(resn, {}).setdefault(resa, {})[atom] = np.array([x_c, y_c, z_c])
            seq.setdefault(resn, {}).setdefault(resa, resi)

    if min_resn == 1e6:
        return 'no_chain', 'no_chain'

    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn+1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        xyz_.append(xyz[resn][k].get(atom, np.full(3, np.nan)))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except Exception:
        return 'no_chain', 'no_chain'

def process_one_pdb(biounit: str, ca_only: bool):
    try:
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
        extra_alphabet = [str(i) for i in range(300)]
        chain_alphabet = init_alphabet + extra_alphabet

        my_dict = {}
        s = 0
        coords_dict = {}
        concat_seq = ''

        atoms = ['CA'] if ca_only else ['N','CA','C','O']  
        for letter in chain_alphabet:
            xyz, seq = parse_PDB_biounits(biounit, atoms=atoms, chain=letter)
            if isinstance(xyz, str):   
                continue
            concat_seq += seq[0]
            my_dict[f'seq_chain_{letter}'] = seq[0]
            cd = {}
            if ca_only:
                cd[f'CA_chain_{letter}'] = xyz[:, 0, :].tolist()
            else:
                cd[f'N_chain_{letter}'] = xyz[:, 0, :].tolist()
                cd[f'CA_chain_{letter}'] = xyz[:, 1, :].tolist()
                cd[f'C_chain_{letter}'] = xyz[:, 2, :].tolist()
                cd[f'O_chain_{letter}'] = xyz[:, 3, :].tolist()
            my_dict[f'coords_chain_{letter}'] = cd
            s += 1

        if s == 0:       
            return None
        if s >= len(chain_alphabet):
            return None
        
        name = os.path.basename(biounit).replace('.pdb', '').replace('.gz', '')
        my_dict['name'] = name
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        return my_dict
    except Exception as e:
        print(f"Skipping {biounit} due to error: {e}")
        return None
 

def main(args):
    input_path = args.input_path.rstrip('/') + '/'
    output_path = args.output_path
    ca_only = args.ca_only
    pdb_list = glob.glob(input_path + '*.pdb*')   
    if not pdb_list:
        print('No pdb files found!')
        return
    pdb_dict_list = []
    workers = min(args.workers, os.cpu_count())
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_one_pdb, pdb, ca_only): pdb for pdb in pdb_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Parsing PDB'):
            res = fut.result()
            if res is not None:
                pdb_dict_list.append(res)
    with open(output_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Done! Parsed {len(pdb_dict_list)} files -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', required=True, help='folder with .pdb(.gz) files')
    parser.add_argument('--output_path', required=True, help='output .jsonl path')
    parser.add_argument('--ca_only', action='store_true', help='backbone-only')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='number of parallel processes')
    args = parser.parse_args()
    main(args)