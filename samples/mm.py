from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *
from mlir_linalg.dsl.yaml_helper import *


@tc_def_op
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(T, S.M, S.N, output=True)):
  """Non accumulator polymorphic version."""
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


# TODO: Make a neat little helper to interpret an op module as
# yaml linal ops.
configs = []
configs.extend(LinalgOpConfig.from_tc_op_def(matmul.model))
configs.extend(LinalgOpConfig.from_tc_op_def(matmul_poly.model))
print(yaml_dump_all(configs))
