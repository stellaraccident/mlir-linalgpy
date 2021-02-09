# linalgpy - Experimental linalg python frontend

This project is just some prototyping of a python frontend to the MLIR
Linalg code generator. If it works out, we will upstream it to LLVM.

## Setting up

Requires pre-release MLIR python bindings:

```
# Optional but recommended - create a virtual environment.
python -m venv ~/.venv/linalg
source ~/.venv/linalg/bin/activate

# Install deps.
# Note that MLIR wheels are from the pre-release index.
pip install -r requirements.py -f https://github.com/stellaraccident/mlir-py-release/releases
```

