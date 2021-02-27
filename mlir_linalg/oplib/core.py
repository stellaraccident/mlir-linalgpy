from mlir_linalg.dsl.tc import *

T1 = TV.T1
T2 = TV.T2

@tc_def_op
def polymorphic_matmul(A=TensorDef(T1, S.M, S.K),
                       B=TensorDef(T2, S.K, S.N),
                       C=TensorDef(U, S.M, S.N, output=True)):
  """Type polymorphic matrix multiplication.

  This op is presently here to test a new path for generation and will replace
  the existing 'matmul' op when ready. Do not use.
  """
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])
