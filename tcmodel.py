"""Model classes representing a tensor-comprehension."""

from typing import Dict, Optional, Sequence, Tuple, Union

from mlir import ir as _ir

from affinedsl import *

# Type aliases.
AffineDimList = Dict[str, _ir.AffineExpr]
ShapeCoercable = Optional[Union[_ir.AffineMap, Sequence[AffineExprDef]]]
TypePredicate = Union[str, _ir.Type]


def _coerce_to_shape(shape_spec: ShapeCoercable) -> _ir.AffineMap:
  if isinstance(shape_spec, _ir.AffineMap):
    return shape_spec
  # Treat as a sequence of affine expressions.
  exprs = []
  # TODO: The symbol space must be preserved for the whole op.
  state = AffineBuildState()
  for expr_def in shape_spec:
    if not isinstance(expr_def, AffineExprDef):
      raise ValueError(
          f"Expected shape dim to be an AffineExprDef. Got {expr_def}")
    exprs.append(expr_def.build(state=state))
  if state.dim_pos_map:
    raise ValueError(
        f"Expected shape affine map to not reference dims. Got {exprs}.")
  return _ir.AffineMap.get(dim_count=0,
                           symbol_count=len(state.symbol_pos_map),
                           exprs=exprs)


class TensorDef:
  """Bookkeeping of a single registered tensor, held in dict by name.
    >>> with _ir.Context():
    ...   TensorDef('f32', shape=(S.M, S.K))
    TensorDef(type_pred=f32, shape=()[s0, s1] -> (s0, s1))

  """

  def __init__(self,
               type_pred: TypePredicate,
               shape: ShapeCoercable = None,
               indexing_map: Optional[_ir.AffineMap] = None,
               is_output: bool = False):
    self.type_pred = type_pred
    self.shape = _coerce_to_shape(shape) if shape is not None else None
    self.indexing_map = indexing_map
    self.is_output = is_output
    self.registered_index = None  # Optional[int]

  @property
  def is_registered(self):
    return self.registered_index is not None

  def __repr__(self):
    output = "OUTPUT " if self.is_output else ""
    return (f"TensorDef({output}type_pred={self.type_pred}, "
            f"shape={self.shape})")


class TcOpDef:
  """Definition of a named op.

    >>> with _ir.Context():
    ...   od = TcOpDef('matmul')
    ...   od.register_tensor(
    ...     A=TensorDef('f32', shape=(S.M, S.K)),
    ...     B=TensorDef('f32', shape=(S.M, S.N)))
    ...   od
    TcOpDef(matmul -> matmul
      A = TensorDef(type_pred=f32, shape=()[s0, s1] -> (s0, s1))
      B = TensorDef(type_pred=f32, shape=()[s0, s1] -> (s0, s1))
  """

  def __init__(self, name: str, cpp_op_name: str = None):
    self.name = name
    self.cpp_op_name = f"{cpp_op_name}Op" if cpp_op_name is not None else name
    self.registered_tensors = dict()  # type: Dict[str, TensorDef]

  def register_tensor(self, **regs: TensorDef):
    """Registers a tensor.
      >>> od = TcOpDef('foobar')
      >>> od.register_tensor(A=TensorDef('f32'))
      >>> od.tensor('A')
      TensorDef(type_pred=f32, shape=None)

    """
    for tensor_name, tensor in regs.items():
      if tensor_name in self.registered_tensors:
        raise ValueError(f"Tensor {tensor_name} is already registered "
                         f"to {self.register_tensors['tensor_name']}")
      if tensor.is_registered:
        raise ValueError(f"Tensor is already registered: {tensor}")
      tensor.registered_index = len(self.registered_tensors)
      self.registered_tensors[tensor_name] = tensor

  def tensor(self, name):
    """Gets a registered tensor by name."""
    try:
      return self.registered_tensors[name]
    except KeyError:
      raise KeyError(f"Tensor {name} is not registered")

  def __repr__(self):
    lines = [f"TcOpDef({self.name} -> {self.cpp_op_name}"]
    for name, tensor in self.registered_tensors.items():
      lines.append(f"  {name} = {tensor}")
    return "\n".join(lines)

if __name__ == "__main__":
  import doctest
  doctest.testmod()
