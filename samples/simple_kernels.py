"""Some sample comprehensions."""

from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *
from mlir_linalg.dsl.yaml_helper import *

# TODO: Make a nice auto collector thingy.
all_ops = []


def collect_op(op):
  all_ops.extend(LinalgOpConfig.from_tc_op_def(op.model))
  return op


@collect_op
@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@collect_op
@tc_def_op
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@collect_op
@tc_def_op
def conv_1d(I=TensorDef(T, S.W),
            K=TensorDef(T, S.KW),
            O=TensorDef(T, S.W, output=True)):
  O[D.w] += I[D.w + D.kw] * K[D.kw]


@collect_op
@tc_def_op
def batch_matmul(A=TensorDef(T, S.Batch, S.M, S.K),
                 B=TensorDef(T, S.K, S.N),
                 C=TensorDef(T, S.Batch, S.M, S.N, output=True)):
  C[D.b, D.m, D.n] += A[D.b, D.m, D.k] * B[D.k, D.n]


print(yaml_dump_all(all_ops))
