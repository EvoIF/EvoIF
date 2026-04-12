import argparse
import logging
from pathlib import Path

def append_pdb_extension(target_dir: str) -> None:
    """
    Appends the '.pdb' extension to all files in the specified directory.
    Skips directories and files that already have the '.pdb' extension.
    """
    dir_path = Path(target_dir)

    if not dir_path.is_dir():
        logging.error(f"Directory not found: {target_dir}")
        return

    renamed_count = 0
    for file_path in dir_path.iterdir():
        # Ensure it's a file and doesn't already end with .pdb (idempotency)
        if file_path.is_file() and file_path.suffix.lower() != '.pdb':
            new_name = f"{file_path.name}.pdb"
            new_path = file_path.with_name(new_name)
            
            file_path.rename(new_path)
            renamed_count += 1

    logging.info(f"Successfully appended '.pdb' to {renamed_count} files in '{target_dir}'.")

if __name__ == "__main__":
    # Setup basic logging for clean console output
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Append .pdb extension to files in a directory.")
    parser.add_argument(
        "-d", "--dir", 
        type=str, 
        required=True, 
        help="Path to the target directory containing the files."
    )
    
    args = parser.parse_args()
    append_pdb_extension(args.dir)