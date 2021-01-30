from mlir.ir import *
from tcmodel import *


def def_matmul():
  od = TcOpDef("matmul")
  A, B, C = od.add_tensor(A=TensorDef('f32', shape=(S.M, S.K)),
                          B=TensorDef('f32', shape=(S.M, S.N)),
                          C=TensorDef('f32', shape=(S.M, S.N), output=True))
  c = od.add_comprehension(C[D.n, D.m])

  print(repr(od))


def def_conv_1d():
  od = TcOpDef("conv_1d")
  I, K, O = od.add_tensor(I=TensorDef('f32', shape=(S.W,)),
                          K=TensorDef('f32', shape=(S.KW,)),
                          O=TensorDef('f32', shape=(S.W,)))
  print(od)


def def_batch_matmul():
  od = TcOpDef("batch_matmul")
  A, B, C = od.add_tensor(A=TensorDef('f32', shape=(S.Batch, S.M, S.K)),
                          B=TensorDef('f32', shape=(S.K, S.N)),
                          C=TensorDef('f32', shape=(S.Batch, S.M, S.N)))
  c = od.add_comprehension(C[D.b, D.m, D.n])
  print(od)


with Context():
  def_matmul()
  print()
  def_conv_1d()
  print()
  def_batch_matmul()
