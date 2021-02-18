"""Models DAGs of scalar math expressions.

Used for generating region bodies at the "math" level where they are still type
polymorphic. This is modeled to be polymorphic by attribute name for interop
with serialization schemes that are just plain-old-dicts.
"""

from typing import Optional, Sequence

from .yaml_helper import *

__all__ = [
    "ScalarAssign",
    "ScalarApplyFn",
    "ScalarArg",
    "ScalarExpression",
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


class ScalarExpression(YAMLObject):
  """An expression on scalar values."""
  yaml_tag = "!ScalarExpression"

  def __init__(self,
               scalar_apply: Optional[ScalarApplyFn] = None,
               scalar_arg: Optional[ScalarArg] = None):
    if (bool(scalar_apply) + bool(scalar_arg)) != 1:
      raise ValueError("One of 'scalar_apply' or 'scalar_block_arg' must be "
                       "specified")
    self.scalar_apply = scalar_apply
    self.scalar_arg = scalar_arg

  def to_yaml_custom_dict(self):
    if self.scalar_apply:
      return dict(scalar_apply=dict(
          fn_name=self.scalar_apply.fn_name,
          operands=list(self.scalar_apply.operands),
      ))
    elif self.scalar_arg:
      return dict(scalar_arg=self.scalar_arg.arg)
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
