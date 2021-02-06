"""Some sample comprehensions.

Currently outputs:

TcOpDef(matmul_poly -> MatmulPolyOp,
  A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K)))
  B:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(N)))
  C:TensorDef(OUTPUT TypeVar(U), shape=(Symbol(M), Symbol(N))) {
    C[Dim(m), Dim(n)] = reduce_add(Dim(k))(mul(A[Dim(m), Dim(k)], B[Dim(k), Dim(n)]))
}
TcOpDef(matmul -> MatmulOp,
  A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K)))
  B:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(N)))
  C:TensorDef(OUTPUT TypeVar(T), shape=(Symbol(M), Symbol(N))) {
    C[Dim(m), Dim(n)] = reduce_add(Dim(k))(mul(A[Dim(m), Dim(k)], B[Dim(k), Dim(n)]))
}
TcOpDef(conv_1d -> Conv1DOp,
  I:TensorDef(TypeVar(T), shape=(Symbol(W),))
  K:TensorDef(TypeVar(T), shape=(Symbol(KW),))
  O:TensorDef(OUTPUT TypeVar(T), shape=(Symbol(W),)) {
    O[Dim(w)] = reduce_add(Dim(kw))(mul(I[AffineAddExpr(Dim(w), Dim(kw))], K[Dim(kw)]))
}
TcOpDef(batch_matmul -> BatchMatmulOp,
  A:TensorDef(TypeVar(T), shape=(Symbol(Batch), Symbol(M), Symbol(K)))
  B:TensorDef(TypeVar(T), shape=(Symbol(K), Symbol(N)))
  C:TensorDef(OUTPUT TypeVar(T), shape=(Symbol(Batch), Symbol(M), Symbol(N))) {
    C[Dim(b), Dim(m), Dim(n)] = reduce_add(Dim(k))(mul(A[Dim(b), Dim(m), Dim(k)], B[Dim(k), Dim(n)]))
}
TcOpDef(log_matmul_exp -> LogMatmulExpOp,
  A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K)))
  B:TensorDef(TypeVar(T), shape=(Symbol(K), Symbol(N)))
  C:TensorDef(OUTPUT TypeVar(T), shape=(Symbol(M), Symbol(N)))
  Interim:TensorDef(TypeVar(T), shape=(Symbol(K), Symbol(M), Symbol(N)))
  TmpShift:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(N)))
  Tmp:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(N))) {
    Interim[Dim(k), Dim(m), Dim(n)] = add(A[Dim(m), Dim(k)], B[Dim(k), Dim(n)])
    TmpShift[Dim(m), Dim(n)] = reduce_max(Dim(k))(Interim[Dim(k), Dim(m), Dim(n)])
    Tmp[Dim(m), Dim(n)] = reduce_add(Dim(k))(exp(sub(Interim[Dim(k), Dim(m), Dim(n)], TmpShift[Dim(m), Dim(n)])))
    C[Dim(m), Dim(n)] = add(log(Tmp[Dim(m), Dim(n)]), TmpShift[Dim(m), Dim(n)])
}

"""

from mlir.ir import *
from tcmodel import *
from tcdsl import *

@tc_def_op
def matmul_poly(A: TensorDef(T, S.M, S.K), B: TensorDef(T, S.M, S.N),
                C: TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


print(matmul_poly.tc_model)


@tc_def_op
def matmul(A: TensorDef(T, S.M, S.K), B: TensorDef(T, S.M, S.N),
           C: TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


print(matmul.tc_model)


@tc_def_op
def conv_1d(I: TensorDef(T, S.W), K: TensorDef(T, S.KW),
            O: TensorDef(T, S.W, output=True)):
  O[D.w] += I[D.w + D.kw] * K[D.kw]


print(conv_1d.tc_model)


@tc_def_op
def batch_matmul(A: TensorDef(T, S.Batch, S.M, S.K), B: TensorDef(T, S.K, S.N),
                 C: TensorDef(T, S.Batch, S.M, S.N, output=True)):
  C[D.b, D.m, D.n] += A[D.b, D.m, D.k] * B[D.k, D.n]


print(batch_matmul.tc_model)


@tc_def_op
def log_matmul_exp(
    A: TensorDef(T, S.M, S.K),
    B: TensorDef(T, S.K, S.N),
    C: TensorDef(T, S.M, S.N, output=True),
    Interim: TensorDef(T, S.K, S.M, S.N),  # TODO: Mark temp
    TmpShift: TensorDef(T, S.M, S.N),  # TODO: Mark temp
    Tmp: TensorDef(T, S.M, S.N)):  # TODO: Mark temp
  Interim[D.k, D.m, D.n] = A[D.m, D.k] + B[D.k, D.n]
  TmpShift[D.m, D.n] = Reduce.max(D.k)(Interim[D.k, D.m, D.n])
  Tmp[D.m, D.n] += Prim.exp(Interim[D.k, D.m, D.n] - TmpShift[D.m, D.n])
  C[D.m, D.n] = Prim.log(Tmp[D.m, D.n]) + TmpShift[D.m, D.n]


print(log_matmul_exp.tc_model)
