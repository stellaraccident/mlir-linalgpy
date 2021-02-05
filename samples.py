from mlir.ir import *
from tcmodel import *


def def_matmul():
  od = TcOpDef("matmul")
  A, B, C = od.add_tensor(A=TensorDef('f32', shape=(S.M, S.K)),
                          B=TensorDef('f32', shape=(S.M, S.N)),
                          C=TensorDef('f32', shape=(S.M, S.N), output=True))
  C[D.n, D.m] = Prim.add[D.k](Prim.mul(A[D.m, D.k], B[D.k, D.n]))
  print(repr(od))


def def_conv_1d():
  od = TcOpDef("conv_1d")
  I, K, O = od.add_tensor(I=TensorDef('f32', shape=(S.W,)),
                          K=TensorDef('f32', shape=(S.KW,)),
                          O=TensorDef('f32', shape=(S.W,)))
  O[D.w] = Prim.add[D.kw](Prim.mul(I[D.w + D.kw], K[D.kw]))
  print(od)


def def_batch_matmul():
  od = TcOpDef("batch_matmul")
  A, B, C = od.add_tensor(A=TensorDef('f32', shape=(S.Batch, S.M, S.K)),
                          B=TensorDef('f32', shape=(S.K, S.N)),
                          C=TensorDef('f32', shape=(S.Batch, S.M, S.N)))
  C[D.b, D.m, D.n] = Prim.add[D.k](Prim.mul(A[D.b, D.m, D.k], B[D.k, D.n]))
  print(od)


with Context():
  def_matmul()
  print()
  def_conv_1d()
  print()
  def_batch_matmul()
