# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *
from mlir_linalg.dsl import yaml_helper

@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  """This is a type polymorphic matmul with a configurable accumulator type.

  This is some more text to see how it prints.
  """
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]

# CHECK-LABEL: !LinalgGenericNamedOpConfig
# CHECK:       args:
# CHECK:       - !<LinalgTensorDef>
# CHECK:         name: A
# CHECK:         shape: (d0, d1, d2)[s0, s1, s2] -> (s0, s2)
# CHECK:         usage: input
# CHECK:       - !<LinalgTensorDef>
# CHECK:         name: B
# CHECK:         shape: (d0, d1, d2)[s0, s1, s2] -> (s2, s1)
# CHECK:         usage: input
# CHECK:       - !<LinalgTensorDef>
# CHECK:         name: C
# CHECK:         shape: (d0, d1, d2)[s0, s1, s2] -> (s0, s1)
# CHECK:         usage: output
# CHECK:       cpp_op_name: MatmulPolyOp
# CHECK:       doc: "This is a type polymorphic matmul with a configurable accumulator type.\n\n\
# CHECK:         \  This is some more text to see how it prints.\n  "
# CHECK:       indexing_maps:
# CHECK:       - (d0, d1, d2)[s0, s1, s2] -> (d0, d2)
# CHECK:       - (d0, d1, d2)[s0, s1, s2] -> (d2, d1)
# CHECK:       - (d0, d1, d2)[s0, s1, s2] -> (d0, d1)
# CHECK:       iterator_types:
# CHECK:       - parallel
# CHECK:       - parallel
# CHECK:       - reduction
# CHECK:       name: matmul_poly
print(yaml_helper.dump(from_tc_op_def(matmul_poly.model)))
