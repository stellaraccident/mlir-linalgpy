# RUN: %PYTHON -m mlir_linalg.tools.dump_oplib mlir_linalg.oplib.core --format=repr | FileCheck %s

# CHECK: LinalgOpConfig(
