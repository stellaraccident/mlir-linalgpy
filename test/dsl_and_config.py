# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *

# CHECK-LABEL: TcOpDef(matmul_poly -> MatmulPolyOp,
# CHECK:         A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K)))
# CHECK:         B:TensorDef(TypeVar(T), shape=(Symbol(K), Symbol(N)))
# CHECK:         C:TensorDef(OUTPUT TypeVar(U), shape=(Symbol(M), Symbol(N))) {
# CHECK:         C[Dim(m), Dim(n)] = reduce_add(Dim(k))(mul(A[Dim(m), Dim(k)], B[Dim(k), Dim(n)]))
# CHECK:       }


@tc_def_op
def matmul_poly(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


print(matmul_poly.model)

# CHECK-LABEL: LinalgGenericOpConfig(reduction_dims=(Dim(k),),
# CHECK:       tensor_args=[
# CHECK:         Def(A:TensorDef(TypeVar(T), shape=(Symbol(M), Symbol(K))), shape_map=(d0, d1, d2)[s0, s1, s2] -> (s0, s2), indexing_map=(d0, d1, d2)[s0, s1, s2] -> (d0, d2))
# CHECK:         Def(B:TensorDef(TypeVar(T), shape=(Symbol(K), Symbol(N))), shape_map=(d0, d1, d2)[s0, s1, s2] -> (s2, s1), indexing_map=(d0, d1, d2)[s0, s1, s2] -> (d2, d1))
# CHECK:         Def(C:TensorDef(OUTPUT TypeVar(U), shape=(Symbol(M), Symbol(N))), shape_map=(d0, d1, d2)[s0, s1, s2] -> (s0, s1), indexing_map=(d0, d1, d2)[s0, s1, s2] -> (d0, d1))
# CHECK:       ], indexing_maps=[
# CHECK:         AffineMap((d0, d1, d2)[s0, s1, s2] -> (d0, d2))
# CHECK:         AffineMap((d0, d1, d2)[s0, s1, s2] -> (d2, d1))
# CHECK:         AffineMap((d0, d1, d2)[s0, s1, s2] -> (d0, d1))
# CHECK:       ], iterator_types=[
# CHECK:         parallel
# CHECK:         parallel
# CHECK:         reduction
# CHECK:       ])
config = LinalgGenericNamedOpConfig(matmul_poly.model.comprehensions[0])
print(config)
