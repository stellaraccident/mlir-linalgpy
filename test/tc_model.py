# RUN: %PYTHON %s | FileCheck %s

from mlir_linalg.dsl.tc import *


# CHECK-LABEL: TcOpDef(matmul_poly -> MatmulPolyOp,
# CHECK:         A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K)))
# CHECK:         B:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(N)))
# CHECK:         C:TensorDef(OUTPUT TypeVar(U), shape=(Symbol(M), Symbol(N))) {
# CHECK:         C[Dim(m), Dim(n)] = reduce_add(Dim(k))(mul(A[Dim(m), Dim(k)], B[Dim(k), Dim(n)]))
# CHECK:       }

@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.M, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


print(matmul_poly.model)
