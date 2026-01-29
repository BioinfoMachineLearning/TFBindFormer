#!/bin/bash

# Generate 3Di tokens from TF PDB structures using Foldseek

set -e  # exit on error (recommended)

PDB_DIR=$1
OUT_DIR=$2

mkdir -p "${OUT_DIR}"

foldseek createdb -v 3 "${PDB_DIR}" "${OUT_DIR}/pdb_3Di"
foldseek lndb "${OUT_DIR}/pdb_3Di_h" "${OUT_DIR}/pdb_3Di_ss_h"
foldseek convert2fasta "${OUT_DIR}/pdb_3Di_ss" "${OUT_DIR}/pdb_3Di_ss.fasta"

echo "3Di FASTA written to ${OUT_DIR}/pdb_3Di_ss.fasta"
