from typing import List, Optional, Sequence, Tuple

# TODO: Generate mypy stubs so we can do a * import.
from mlir.ir import ArrayAttr, BoolAttr, ShapedType, Type, RankedTensorType, UnrankedTensorType
from mlir.dialects import linalg

from ..dsl.tc_model import *

__all__ = [
    "TcEmitGenericCallable",
]

ShapeDim = Optional[int]
ShapeDims = Optional[List[ShapeDim]]


def _shaped_type_to_shape(shaped_type: Optional[ShapedType]) -> ShapeDims:
  """Extracts a shape tuple from a ShapedType.

  Returns:
    None for unranked or a Tuple of Union[None, int] for each dim.
  """
  if not shaped_type:
    return None
  if not shaped_type.has_rank:
    return None

  def get_dim(i):
    if shaped_type.is_dynamic_dim(i):
      return None
    else:
      return shaped_type.get_dim_size(i)

  return [get_dim(i) for i in range(shaped_type.rank)]


def _refine_shape(output_tensor: TensorDef, input_shapes: Sequence[ShapeDims],
                  output_shape: ShapeDims) -> ShapeDims:
  # TODO: Refine to a more specific shape.
  return [None] * output_tensor.rank


def _refine_element_type(output_tensor: TensorDef,
                         input_tensors: Sequence[TensorDef],
                         input_element_types: Sequence[Type],
                         output_element_type: Optional[Type]) -> Type:
  if output_element_type:
    return output_element_type
  # Populate type vars.
  type_var_bindings = dict()  # type: Dict[TypeVar, Type]
  for input_tensor, input_element_type in zip(input_tensors,
                                              input_element_types):
    tv = input_tensor.type_var
    existing_binding = type_var_bindings.get(tv)
    if not existing_binding:
      type_var_bindings[tv] = input_element_type
    elif existing_binding != input_element_type:
      raise ValueError(f"Mismatched input type (disagrees with previous input):"
                       f"{input_element_types} for {tv}")

  # Resolve the output.
  output_element_type = type_var_bindings.get(output_tensor.type_var)
  if not output_element_type:
    raise ValueError(
        f"Output tensor type {output_tensor.type_var} does not "
        f"relate to an input and the type must be explicitly specified.")
  return output_element_type


def _compose_shaped_type(input_types: Sequence[Type], shape: ShapeDims,
                         element_type: Type) -> Type:
  """Recompose a ShapedType from shape and element_type components.

  Takes the input_types to match specific ShapedType hierarchy (i.e. tensor vs
  memref).
  """
  # TODO: Check tensor vs memref.
  if shape is None:
    return UnrankedTensorType.get(element_type)
  else:
    return RankedTensorType.get([(-1 if d is None else d) for d in shape],
                                element_type)


class TcEmitGenericCallable:
  """Callable that emits a generic op sequence for a model."""

  def __init__(self, name: str, model: TcOpDef):
    self.__name__ = name
    self.model = model

  def __call__(self,
               *inputs: Type,
               output_types: Optional[Sequence[Type]] = None,
               sparse: Optional[bool] = None):
    output_types = self._infer_output_types(inputs, output_types)
    # Construct the linalg.generic op.
    # TODO: For multiple comprehensions, this expands into multiple generics.
    # TODO: Need to materialize init tensors.
    indexing_maps = ArrayAttr.get([])
    iterator_types = ArrayAttr.get([])
    generic_op = linalg.GenericOp(
        result_tensors=output_types,
        inputs=inputs,
        outputs=[inputs[0]],
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
        doc=None,  # TODO: Optional?
        library_call=None,  # TODO: Optional?
        sparse=BoolAttr.get(sparse)
        if sparse is not None else None)  # TODO: Optional?
    return generic_op.results

  def _infer_output_types(
      self,
      inputs: Sequence[Type],
      output_types: Optional[Sequence[Type]] = None) -> Sequence[Type]:
    op_inputs = self.model.inputs
    if len(op_inputs) != len(inputs):
      raise ValueError(
          f"Input arity mismatch: Got {len(inputs)} inputs but expected"
          f"{len(op_inputs)}")
    op_outputs = self.model.outputs

    # Expand input types to shapes and element types.
    input_types = [ShapedType(inp.type) for inp in inputs]
    input_shapes = [_shaped_type_to_shape(t) for t in input_types]
    input_element_types = [t.element_type for t in input_types]

    # Expand output types to shapes and element types.
    if not output_types:
      output_types = [None] * len(op_outputs)
    output_shapes = [_shaped_type_to_shape(st) for st in output_types]
    output_element_types = list([
        (t.element_type if t else None) for t in output_types
    ])

    # Refine output shapes and types.
    resolved_output_types = []
    for i, op_output in enumerate(op_outputs):
      output_shapes[i] = _refine_shape(op_output, input_shapes,
                                       output_shapes[i])
      output_element_types[i] = _refine_element_type(op_output, op_inputs,
                                                     input_element_types,
                                                     output_element_types[i])
      resolved_output_types.append(
          _compose_shaped_type(input_types, output_shapes[i],
                               output_element_types[i]))
    return resolved_output_types
