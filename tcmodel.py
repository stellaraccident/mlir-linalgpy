"""Model classes representing a tensor-comprehension."""

from typing import Dict, Optional, Tuple, Union

from mlir import ir as _ir

from affinedsl import *

# Type aliases.
AffineDimList = Dict[str, _ir.AffineExpr]
TypePredicate = Union[str, _ir.Type]


class TensorDef:
  """Bookkeeping of a single registered tensor, held in dict by name."""

  def __init__(self,
               type_pred: TypePredicate,
               shape: Optional[_ir.AffineMap] = None,
               indexing_map: Optional[_ir.AffineMap] = None,
               is_output: bool = False):
    self.type_pred = type_pred
    self.shape = shape
    self.indexing_map = indexing_map
    self.is_output = is_output
    self.registered_index = None  # Optional[int]

  @property
  def is_registered(self):
    return self.registered_index is not None

  def __repr__(self):
    output = " OUTPUT" if self.is_output else ""
    return f"TensorDef(type_pred={self.type_pred}{output})"


class TcOpDef:
  """Definition of a named op."""

  def __init__(self, name: str, cpp_op_name: str = None):
    self.name = name
    self.cpp_op_name = f"{cpp_op_name}Op" if cpp_op_name is not None else name
    self.registered_tensors = dict()  # type: Dict[str, TensorDef]

  def register_tensor(self, **regs: TensorDef):
    """Registers a tensor.
      >>> od = TcOpDef('foobar')
      >>> od.register_tensor(A=TensorDef('f32'))
      >>> od.tensor('A')
      TensorDef(type_pred=f32)
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


if __name__ == "__main__":
  import doctest
  doctest.testmod()
