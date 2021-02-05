from mlir.ir import *
from tcmodel import *
from tcdsl import *


@tc_def_op
def matmul(A: TensorDef('f32', S.M, S.K), B: TensorDef('f32', S.M, S.N),
           C: TensorDef('f32', S.M, S.N, output=True)):
  C[D.n, D.m] = Prim.add[D.k](Prim.mul(A[D.m, D.k], B[D.k, D.n]))


@tc_def_op
def conv_1d(I: TensorDef('f32', S.W), K: TensorDef('f32', S.KW),
            O: TensorDef('f32', S.W)):
  O[D.w] = Prim.add[D.kw](Prim.mul(I[D.w + D.kw], K[D.kw]))


@tc_def_op
def batch_matmul(A: TensorDef('f32', S.Batch, S.M,
                              S.K), B: TensorDef('f32', S.K, S.N),
                 C: TensorDef('f32', S.Batch, S.M, S.N)):
  C[D.b, D.m, D.n] = Prim.add[D.k](Prim.mul(A[D.b, D.m, D.k], B[D.k, D.n]))


print(matmul.tc_model)
print(conv_1d.tc_model)
print(batch_matmul.tc_model)
