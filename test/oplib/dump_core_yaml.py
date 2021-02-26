# RUN: %PYTHON -m mlir_linalg.tools.dump_oplib mlir_linalg.oplib.core --format=yaml | FileCheck %s

# CHECK: cpp_op_name: PolymorphicMatmulOp
