#!/usr/bin/env python3
"""
Realign MSA files from Foldseek structure search to ensure all target sequences
have the same length as the query. Files are automatically renamed based on the
first header before realignment.

Usage:
    ./struc_align.py --input_dir ./msa_output --output_dir ./msa_aligned
"""

import os
import shutil
import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple


DEFAULT_EXTENSIONS = {".fasta", ".a3m"}
# Pattern to match Foldseek output files without extensions (e.g., 0a3m, 1a3m, 123a3m)
FOLDSEEK_PATTERN = re.compile(r'^\d+a3m$')

def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

def rename_files(input_dir: Path) -> None:
    """
    Rename each file under input_dir using the first token of its first header.
    New names are of the form "<first_word>.fasta". 
    Handles both standard extensions and extensionless Foldseek outputs (e.g., 0a3m).
    """
    files = []
    for p in input_dir.rglob("*"):
        if not p.is_file():
            continue
        # Check for standard extensions OR extensionless Foldseek format (e.g., 0a3m, 1a3m)
        if p.suffix.lower() in DEFAULT_EXTENSIONS or FOLDSEEK_PATTERN.match(p.name):
            files.append(p)
            
    if not files:
        logging.warning("No files with matching extensions found for renaming.")
        return

    for old_path in files:
        try:
            with old_path.open("r") as fh:
                first_line = fh.readline().strip()
        except Exception as e:
            logging.error(f"Cannot read {old_path}: {e}")
            continue
        if not first_line.startswith(">"):
            logging.warning(f"Skipping (no header): {old_path}")
            continue
        new_name = first_line[1:].split()[0] + ".fasta"
        new_path = old_path.with_name(new_name)
        if new_path.exists():
            logging.warning(f"Target exists, skipping: {old_path} -> {new_path}")
            continue

        shutil.move(str(old_path), str(new_path))
        logging.info(f"Renamed: {old_path} -> {new_path}")

def parse_fasta(path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Parse a FASTA-like file and return (query_seq, list of (header, target_seq)).
    All gap characters ('-') are removed from the sequences.
    """
    sequences = []
    with path.open() as fh:
        header = None
        seq_lines = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_lines).replace("-", "")))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequences.append((header, "".join(seq_lines).replace("-", "")))

    if not sequences:
        raise ValueError(f"{path}: no sequences found")
    return sequences[0][1], sequences[1:]

def realign_target(query_len: int, target_seq: str) -> str:
    """Truncate or pad target_seq to exactly match query_len."""
    if len(target_seq) >= query_len:
        return target_seq[:query_len]
    else:
        return target_seq + "-" * (query_len - len(target_seq))

def process_file(src: Path, dst: Path) -> None:
    """
    Read src MSA file, realign all targets to query length, write result to dst.
    """
    query_seq, targets = parse_fasta(src)
    query_len = len(query_seq)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as out:
        out.write(f">query\n{query_seq}\n")
        for header, target_seq in targets:
            aligned = realign_target(query_len, target_seq)
            if len(aligned) != query_len:
                raise RuntimeError(f"{src}: length mismatch after alignment")
            out.write(f"{header}\n{aligned}\n")
    logging.debug(f"Written: {dst}")

def process_all_files(input_dir: Path, output_dir: Path) -> None:
    """Recursively realign all MSA files under input_dir and save under output_dir."""
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in DEFAULT_EXTENSIONS]
    if not files:
        logging.warning("No files with matching extensions found for alignment.")
        return

    for src in files:
        if not src.is_file():
            continue
        rel_path = src.relative_to(input_dir)
        dst = output_dir / rel_path
        logging.info(f"Processing {src} -> {dst}")
        try:
            process_file(src, dst)
        except Exception as e:
            logging.error(f"Failed to process {src}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename and realign MSA files from Foldseek structure search."
    )
    parser.add_argument("--input_dir", "-i", type=Path, required=True,
                        help="Directory containing input MSA files (searched recursively).")
    parser.add_argument("--output_dir", "-o", type=Path, required=True,
                        help="Directory where realigned MSAs will be saved.")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()

    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("Starting renaming step...")
    rename_files(args.input_dir)
    logging.info("Renaming completed.")
    logging.info("Starting realignment step...")
    process_all_files(args.input_dir, args.output_dir)
    logging.info(f"All done. Results written to {args.output_dir}")

if __name__ == "__main__":
    main()