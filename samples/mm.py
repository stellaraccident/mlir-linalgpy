from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *
from mlir_linalg.dsl.yaml_helper import *


@tc_def_op
def test_matmul(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  """Type polymorphic matrix multiplication.

  This op is presently here to test a new path for generation and will replace
  the existing 'matmul' op when ready. Do not use.
  """
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


# TODO: Make a neat little helper to interpret an op module as
# yaml linalg ops.
configs = []
configs.extend(LinalgOpConfig.from_tc_op_def(test_matmul.model))
print(yaml_dump_all(configs))
