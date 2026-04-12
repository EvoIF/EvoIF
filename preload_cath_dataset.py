import os
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
from evoif.dataset import CATH
def parse_args():
    """
    Parse command line arguments for the CATH dataset preloading script.
    """
    parser = argparse.ArgumentParser(description="Preload CATH dataset and save preprocessed items for faster future loading.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the processed CATH dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the preloaded data.')
    parser.add_argument('--struc_align_path', type=str, default=None, help='Path to structural alignment data.')
    parser.add_argument('--if_profile_path', type=str, default=None, help='Path to inverse folding profiles.')
    
    # Dataset configurations
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length allowed.')
    parser.add_argument('--profile_types', nargs='*', default=None, help='List of profile types to include.')
    
    return parser.parse_args()

def init_dataset(args):
    """
    Initialize the CATH dataset object with provided arguments.
    """
    # Local import to avoid dependency issues if the script is run in different environments
    
    dataset = CATH(
        path=args.input_path,
        max_length=args.max_length,
        struc_align_path=args.struc_align_path,
        if_profile_path=args.if_profile_path,
        profile_types=args.profile_types
    )
    return dataset

def save_config(args, dataset, output_path):
    """
    Save the dataset configuration to ensure consistency during future loading.
    """
    config = {
        "input_path": args.input_path,
        "struc_align_path": args.struc_align_path,
        "if_profile_path": args.if_profile_path,
        "max_length": args.max_length,
        "profile_types": args.profile_types,
        "num_samples": len(dataset)
    }
    config_path = os.path.join(output_path, "dataset_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Dataset configuration saved to: {config_path}")

def save_item_fields(item, base_name, output_path):
    """
    Serialize each field of a data item into separate pickle files.
    """
    for field_name, field_data in item.items():
        if field_data is not None:
            field_file = os.path.join(output_path, f"{base_name}_{field_name}.pkl")
            with open(field_file, "wb") as f:
                pickle.dump(field_data, f)

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory initialized at: {args.output_path}")
    dataset = init_dataset(args)
    save_config(args, dataset, args.output_path)
    num_samples = len(dataset)
    print(f"Total samples to process: {num_samples}")
    failed_indices = []
    for i in tqdm(range(num_samples), desc="Preloading CATH Dataset"):
        try:
            item = dataset.get_item(i)
            if item is None:
                failed_indices.append(i)
                continue
            raw_filename = dataset.pkl_files[i]
            base_name = Path(raw_filename).stem
            sample_dir = os.path.join(args.output_path, base_name)
            os.makedirs(sample_dir, exist_ok=True)
            save_item_fields(item, base_name, sample_dir)
            
        except Exception as e:
            failed_indices.append(i)
            print(f"Error processing sample {i} ({dataset.pkl_files[i]}): {e}")

    # Final summary and error logging
    if failed_indices:
        fail_path = os.path.join(args.output_path, "failed_indices.pkl")
        with open(fail_path, "wb") as f:
            pickle.dump(failed_indices, f)
        print(f"Processing complete with {len(failed_indices)} failures. Details saved to: {fail_path}")
    else:
        print("Success! All samples have been preloaded and saved.")

if __name__ == "__main__":
    main()