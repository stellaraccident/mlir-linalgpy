"""Sample of a multi-contraction kernel."""

from mlir_linalg.dsl.tc import *


@tc_def_op
def log_matmul_exp(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True),
    Interim=TensorDef(T, S.K, S.M, S.N),  # TODO: Mark temp
    TmpShift=TensorDef(T, S.M, S.N),  # TODO: Mark temp
    Tmp=TensorDef(T, S.M, S.N)):  # TODO: Mark temp
  Interim[D.k, D.m, D.n] = A[D.m, D.k] + B[D.k, D.n]
  TmpShift[D.m, D.n] = ReduceFn.max(D.k)(Interim[D.k, D.m, D.n])
  Tmp[D.m, D.n] += PrimFn.exp(Interim[D.k, D.m, D.n] - TmpShift[D.m, D.n])
  C[D.m, D.n] = PrimFn.log(Tmp[D.m, D.n]) + TmpShift[D.m, D.n]


# NOTE: Does not currently map to a linalg op.
print(repr(log_matmul_exp.model))
