# linalgpy - Experimental linalg python frontend

This project is just some prototyping of a python frontend to the MLIR
Linalg code generator. If it works out, we will upstream it to LLVM.

## Setting up

Requires pre-release MLIR python bindings:

```shell
# Optional but recommended - create a virtual environment.
python -m venv ~/.venv/linalg
source ~/.venv/linalg/bin/activate

# Install deps.
# Note that MLIR wheels are from the pre-release index.
pip install -r requirements.txt -f https://github.com/stellaraccident/mlir-py-release/releases
```

Since this is aiming for upstream, it lays out tests for execution via `lit`
and `FileCheck`. `lit` comes in via the above `requirements.txt` but `FileCheck`
must be on your path. If you built LLVM, it is in the build tree. If you
installed a recent LLVM, you likely have it but with a version suffix like
`FileCheck-10`: In this case create a symlink to `FileCheck`.

## Testing

```shell
lit test -v
```

In order to run a test manually, you will need to set your PYTHONPATH as:

```shell
PYTHONPATH=. python test/tc_model.py
```

## Checking types

TODO: Automate this

```shell
mypy samples/kernels.py
```
