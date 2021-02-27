"""Models DAGs of scalar math expressions.

Used for generating region bodies at the "math" level where they are still type
polymorphic. This is modeled to be polymorphic by attribute name for interop
with serialization schemes that are just plain-old-dicts.
"""

from typing import Optional, Sequence

from .yaml_helper import *
from .types import *

__all__ = [
    "ScalarAssign",
    "ScalarApplyFn",
    "ScalarArg",
    "ScalarExpression",
    "ScalarSymbolicCast",
]


class ScalarApplyFn:
  """Application of a scalar function to operands."""

  def __init__(self, fn_name: str, *operands: "ScalarExpression"):
    self.fn_name = fn_name
    self.operands = operands

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_apply=self)

  def __repr__(self):
    return f"ScalarApplyFn<{self.fn_name}>({', '.join(self.operands)})"


class ScalarArg:
  """A reference to a named argument that is an input to the overall expr."""

  def __init__(self, arg: str):
    self.arg = arg

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(scalar_arg=self)

  def __repr__(self):
    return f"(ScalarArg({self.arg})"


class ScalarSymbolicCast:
  """Symbolically casts to a TypeVar."""

  def __init__(self, to_type: TypeVar, operand: "ScalarExpression"):
    self.to_type = to_type
    self.operand = operand

  def expr(self) -> "ScalarExpression":
    return ScalarExpression(symbolic_cast=self)

  def __repr__(self):
    return f"ScalarSymbolicCast({self.to_type}, {self.operand})"


class ScalarExpression(YAMLObject):
  """An expression on scalar values."""
  yaml_tag = "!ScalarExpression"

  def __init__(self,
               scalar_apply: Optional[ScalarApplyFn] = None,
               scalar_arg: Optional[ScalarArg] = None,
               symbolic_cast: Optional[ScalarSymbolicCast] = None):
    if (bool(scalar_apply) + bool(scalar_arg) + bool(symbolic_cast)) != 1:
      raise ValueError(
          "One of 'scalar_apply', 'scalar_block_arg', 'symbolic_cast' must be "
          "specified")
    self.scalar_apply = scalar_apply
    self.scalar_arg = scalar_arg
    self.symbolic_cast = symbolic_cast

  def to_yaml_custom_dict(self):
    if self.scalar_apply:
      return dict(scalar_apply=dict(
          fn_name=self.scalar_apply.fn_name,
          operands=list(self.scalar_apply.operands),
      ))
    elif self.scalar_arg:
      return dict(scalar_arg=self.scalar_arg.arg)
    elif self.symbolic_cast:
      # Note that even though operands must be arity 1, we write it the
      # same way as for apply because it allows handling code to be more
      # generic vs having a special form.
      return dict(symbolic_cast=dict(type_var=self.symbolic_cast.to_type.name,
                                     operands=[self.symbolic_cast.operand]))
    else:
      raise ValueError(f"Unexpected ScalarExpression type: {self}")


class ScalarAssign(YAMLObject):
  """An assignment to a named argument."""
  yaml_tag = "!ScalarAssign"

  def __init__(self, arg: str, value: ScalarExpression):
    self.arg = arg
    self.value = value

  def to_yaml_custom_dict(self):
    return dict(arg=self.arg, value=self.value)

  def __repr__(self):
    return f"ScalarAssign({self.arg}, {self.value})"
