# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir_linalg.dsl.tc import *

@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  """This is a type polymorphic matmul with a configurable accumulator type.

  This is some more text to see how it prints.
  """
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

# CHECK: --- !LinalgOpConfig
# CHECK: metadata: !LinalgOpMetadata
# CHECK:   name: matmul_poly
# CHECK:   cpp_op_name: MatmulPolyOp
# CHECK:   doc: |-
# CHECK:     This is a type polymorphic matmul with a configurable accumulator type.
# CHECK:     This is some more text to see how it prints.
# CHECK: structured_op: !LinalgStructuredOpConfig
# CHECK:   args:
# CHECK:   - !<LinalgTensorDef>
# CHECK:     name: A
# CHECK:     usage: input
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s0, s2)>
# CHECK:   - !<LinalgTensorDef>
# CHECK:     name: B
# CHECK:     usage: input
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s2, s1)>
# CHECK:   - !<LinalgTensorDef>
# CHECK:     name: C
# CHECK:     usage: output
# CHECK:     shape: affine_map<()[s0, s1, s2] -> (s0, s1)>
# CHECK:   indexing_maps:
# CHECK:     static_indexing_maps:
# CHECK:     - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>
# CHECK:     - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>
# CHECK:     - affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>
# CHECK:   iterator_types:
# CHECK:   - parallel
# CHECK:   - parallel
# CHECK:   - reduction
print(matmul_poly.model.to_yaml())
