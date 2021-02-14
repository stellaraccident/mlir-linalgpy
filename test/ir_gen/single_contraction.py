# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

from mlir_linalg.dsl.tc import *


@tc_def_op
def matmul(A=TensorDef(T, S.M, S.K),
           B=TensorDef(T, S.K, S.N),
           C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


with Context() as ctx, Location.unknown():
  module = Module.create()
  f32 = F32Type.get()
  lhs_type = RankedTensorType.get((4, 16), f32)
  rhs_type = RankedTensorType.get((16, 8), f32)
  result_type = NoneType.get()
  with InsertionPoint.at_block_terminator(module.body):
    func = builtin.FuncOp(name="matmul_test",
                          type=FunctionType.get(inputs=[lhs_type, rhs_type],
                                                results=[result_type]))
    with InsertionPoint(func.add_entry_block()):
      lhs, rhs = func.entry_block.arguments
      results = matmul(lhs, rhs)
      std.ReturnOp(results)
      # Rewrite the function return type now that we know.
      # TODO: Have an API or a setter for this.
      func.attributes["type"] = TypeAttr.get(
          FunctionType.get(func.type.inputs, [r.type for r in results]))

# TODO: This is not right yet.
# CHECK-LABEL: func @matmul_test
# CHECK:      %0 = linalg.generic {indexing_maps = [], iterator_types = []}
# CHECK-SAME:   ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>)
# CHECK-SAME:   outs(%arg0 : tensor<4x16xf32>) -> tensor<?x?xf32>
print(module)
