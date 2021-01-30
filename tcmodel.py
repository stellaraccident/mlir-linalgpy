"""Model classes representing a tensor-comprehension."""

from typing import Dict, Optional, Sequence, Tuple, Union

from mlir import ir as _ir

from affinedsl import *

# Type aliases.
AffineDimList = Dict[str, _ir.AffineExpr]
ShapeCoercable = Optional[Union[_ir.AffineMap, Sequence[AffineExprDef]]]
TypePredicate = Union[str, _ir.Type]


class TensorUse:
  """A used tensor represented by its (tensor_name, indexing_map)."""

  def __init__(self, tensor_name: str, indexing_map: _ir.AffineMap):
    self.tensor_name = tensor_name
    self.indexing_map = indexing_map

  def __repr__(self):
    return f"{self.tensor_name}[{self.indexing_map}]"


class TensorDef:
  """Bookkeeping of a single registered tensor, held in dict by name."""

  def __init__(self,
               type_pred: TypePredicate,
               shape: Optional[ShapeCoercable] = None,
               indexing_map: Optional[_ir.AffineMap] = None,
               output: bool = False):
    self._owner = None
    self.type_pred = type_pred
    self.shape = shape
    self.indexing_map = indexing_map
    self.output = output
    self.tensor_name = None
    self.registered_index = None  # Optional[int]

  def attach(self, index: int, tensor_name: str, owner: "TcOpDef"):
    if self._owner:
      raise ValueError(f"TensorDef already registered with op: {self}")
    self.registered_index = index
    self.tensor_name = tensor_name
    self._owner = owner

    # And do fixups that can only be done once attached.
    if self.shape:
      self.shape = self._owner._coerce_to_shape(self.shape)

  def __getitem__(self, dims) -> TensorUse:
    state = AffineBuildState(global_state=self._owner._affine_state,
                             allow_new_symbols=False)
    if not isinstance(dims, tuple):
      dims = (dims,)  # Handle single subscript case.
    exprs = []
    for expr_def in dims:
      if not isinstance(expr_def, AffineExprDef):
        raise KeyError(
            "A TensorDef can only be subscripted by a tuple of affine dims")
      exprs.append(expr_def.build(state=state))
    indexing_map = _ir.AffineMap.get(dim_count=state.dim_count,
                                     symbol_count=state.symbol_count,
                                     exprs=exprs)
    return TensorUse(self.tensor_name, indexing_map)

  def __repr__(self):
    output = "OUTPUT " if self.output else ""
    return (f"{self.tensor_name}:TensorDef({output}type_pred={self.type_pred}, "
            f"shape={self.shape})")


class Comprehension:
  """Represents a single comprehension."""

  def __init__(self, *definitions: TensorUse):
    self.definitions = definitions

  def __iadd__(self, rhs: TensorUse):
    pass

  def __repr__(self):
    if len(self.definitions) > 1:
      defs_repr = f"({', '.join(repr(d) for d in self.definitions)})"
    else:
      defs_repr = f"{repr(self.definitions[0])}"
    return defs_repr

class TcOpDef:
  """Definition of a named op.

    >>> with _ir.Context():
    ...   od = TcOpDef('matmul')
    ...   A, B, C = od.add_tensor(
    ...     A=TensorDef('f32', shape=(S.M, S.K)),
    ...     B=TensorDef('f32', shape=(S.M, S.N)),
    ...     C=TensorDef('f32', shape=(S.M, S.N), output=True))
    ...   _ = od.add_comprehension(A[D.n, D.m])
    ...   od
    TcOpDef(matmul -> matmul,
      A:TensorDef(type_pred=f32, shape=()[s0, s1] -> (s0, s1))
      B:TensorDef(type_pred=f32, shape=()[s0, s1] -> (s0, s2))
      C:TensorDef(OUTPUT type_pred=f32, shape=()[s0, s1] -> (s0, s2))
  """

  def __init__(self, name: str, cpp_op_name: str = None):
    self.name = name
    self.cpp_op_name = f"{cpp_op_name}Op" if cpp_op_name is not None else name
    self.registered_tensors = dict()  # type: Dict[str, TensorDef]
    self.comprehensions = list()  # type: List[Comprehension]
    self._affine_state = AffineBuildState()

  def add_tensor(self, **regs: TensorDef):
    """Registers a tensor.
      >>> od = TcOpDef('foobar')
      >>> A = od.add_tensor(A=TensorDef('f32'))
      >>> od.tensor('A')
      A:TensorDef(type_pred=f32, shape=None)

    """
    for tensor_name, tensor in regs.items():
      if tensor_name in self.registered_tensors:
        raise ValueError(f"Tensor {tensor_name} is already registered "
                         f"to {self.registered_tensors['tensor_name']}")
      tensor.attach(len(self.registered_tensors), tensor_name, self)
      self.registered_tensors[tensor_name] = tensor
    return list(regs.values())

  def tensor(self, name):
    """Gets a registered tensor by name."""
    try:
      return self.registered_tensors[name]
    except KeyError:
      raise KeyError(f"Tensor {name} is not registered")

  def add_comprehension(self, *definitions: TensorUse):
    c = Comprehension(*definitions)
    self.comprehensions.append(c)
    return c

  def _coerce_to_shape(self, shape_spec: ShapeCoercable) -> _ir.AffineMap:
    state = AffineBuildState(global_state=self._affine_state,
                             allow_new_dims=False)
    if isinstance(shape_spec, _ir.AffineMap):
      return shape_spec
    # Treat as a sequence of affine expressions.
    exprs = []
    for expr_def in shape_spec:
      if not isinstance(expr_def, AffineExprDef):
        raise ValueError(
            f"Expected shape dim to be an AffineExprDef. Got {expr_def}")
      exprs.append(expr_def.build(state=state))
    assert state.dim_count == 0
    return _ir.AffineMap.get(dim_count=0,
                             symbol_count=state.symbol_count,
                             exprs=exprs)

  def __repr__(self):
    lines = [f"TcOpDef({self.name} -> {self.cpp_op_name},"]
    for name, tensor in self.registered_tensors.items():
      lines.append(f"  {tensor}")
    if self.comprehensions:
      lines[-1] += " {"
      for comprehension in self.comprehensions:
        lines.append(f"    {comprehension}")
      lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
