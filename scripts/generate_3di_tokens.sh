#!/bin/bash

# Generate 3Di tokens from TF PDB structures using Foldseek

PDB_DIR=$1
OUT_DIR=$2

mkdir -p ${OUT_DIR}

foldseek createdb -v 3 ${PDB_DIR} ${OUT_DIR}/pdb_3Di
foldseek lndb ${OUT_DIR}/pdb_3Di_h ${OUT_DIR}/pdb_3Di_ss_h
foldseek convert2fasta ${OUT_DIR}/pdb_3Di_ss ${OUT_DIR}/pdb_3Di_ss.fasta


"""
chmod +x scripts/generate_3di_tokens.sh 
./generate_3di_tokens.sh pdb_dir output_dir
"""