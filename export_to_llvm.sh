#!/bin/bash
# Exports the op definitions that we hold in this repo to LLVM.
# This is a placeholder until ready to upstream more of this.

llvm_dir="$1"
if [ -z "$llvm_dir" ]; then
  echo "Usage: export_to_llvm.sh {llvm-project-dir}"
  exit 1
fi

linalg_dir="$llvm_dir/mlir/include/mlir/Dialect/Linalg/IR"
if ! [ -d "$linalg_dir" ]; then
  echo "Could not find directory: $linalg_dir"
  exit 1
fi

yaml_file="$linalg_dir/LinalgNamedStructuredOps.yaml"
if ! [ -f "$yaml_file" ]; then
  echo "Could not find file: $yaml_file"
  exit 1
fi

export PYTHONPATH="$(cd $(dirname $0) && pwd)"
python -m mlir_linalg.tools.dump_oplib mlir_linalg.oplib.core > $yaml_file

echo "Update file: $yaml_file"
