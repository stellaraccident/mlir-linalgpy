from typing import Dict, List

from contextlib import contextmanager
import functools
import inspect
import threading

from mlir import ir
from .tc_model import *

_CONTEXT = threading.local()


def tc_def_op(dsl_func=None, *, op_name=None, op_class_name=None):
  if dsl_func is None:
    # Curry the keyword args in for delayed application.
    return functools.partial(tc_def_op,
                             op_name=op_name,
                             op_class_name=op_class_name)
  # Determine default names by introspecting the function.
  if op_name is None:
    op_name = dsl_func.__name__
  if op_class_name is None:
    # Camel case it.
    op_class_name = f"{''.join(x.title() for x in op_name.split('_'))}Op"

  tc_model = TcOpDef(name=op_name, cpp_op_name=op_class_name)

  # Extract arguments and TensorDefs from the signature.
  dsl_func_args = list()
  sig = inspect.signature(dsl_func)
  for param_name, param in sig.parameters.items():
    param_annot = param.annotation
    if not isinstance(param_annot, TensorDef):
      raise ValueError(f"@tc_def_op function parameters must be annotated as "
                       f"TensorDef(...): Found {param_name}: {param_annot}")
    dsl_func_args.append(param_annot)
    tc_model.add_tensor(param_name, param_annot)

  # Invoke the DSL func to finish populating the model.
  # TODO: Wrap in context manager.
  dsl_func(*dsl_func_args)

  return TcDefinedOpCallable(op_name, tc_model)


class TcDefinedOpCallable:
  """Callable that is generated via the @tc_def_op decorator."""

  def __init__(self, name: str, tc_model: TcOpDef):
    self.__name__ = name
    self.tc_model = tc_model

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("Not yet callable")


@contextmanager
def op_def(*args, **kwargs):
  """Define a new operation."""
  if hasattr(_CONTEXT, "op_def"):
    raise ValueError("Cannot nest multiple operations via 'OpDef'")
  inst = OpDef(*args, **kwargs)
  _CONTEXT.op_def = inst
  try:
    yield inst
  finally:
    del _CONTEXT.op_def


def type_param(*type_symbols):
  """Introduces one or more symbolic type parameters.

    >>> with op_def('matmul'):
    ...   type_param('I', 'J')
    ...   type_param('I')
    Traceback (most recent call last):
        ...
    ValueError: Duplicate type parameter: TypeParam('I')

  """
  for type_symbol in type_symbols:
    t = TypeParam(type_symbol)
    OpDef.current_op().add_type_param(t)


def specialize(**type_bindings):
  """Specializes the op def for a set of type bindings.

    >>> with ir.Context(), op_def('matmul') as op:
    ...   specialize(T=ir.F32Type.get(), TACCUM=ir.F32Type.get())
    ...   specialize(T=ir.IntegerType.get_signless(8), TACCUM=ir.IntegerType.get_signless(32))
    >>> op
    OpDef<matmul>:
      {'T': Type(f32), 'TACCUM': Type(f32)}
      {'T': Type(i8), 'TACCUM': Type(i32)}

    Looking to add something like:
      A = input_match('A', type='T')['M', 'K']
      B = input_match('B', type='T')['K', 'N']
      C = output_match('C', type='TACCUM')['M', 'N']
      C['m', 'n'] += reduce('k')(A['k', 'm'] * B['n', 'k'])

    Probably add a bunch of predefined symbol constants so that things like
    M, N, K, k, etc can just be used literal? Probably have all of these
    be aliases to uniqued 'sym("k")' instances and just sugar? Not thrilled
    with the stringiness here (but fine if it is sugar I guess).

  """
  op = OpDef.current_op()
  binding = {}
  for type_symbol, ir_type in type_bindings.items():
    op.add_type_param(TypeParam(type_symbol), exist_ok=True)
    if not isinstance(ir_type, ir.Type):
      raise TypeError(f"Expected mlir.ir.Type but got {ir_type}")
    binding[type_symbol] = ir_type
  op.specializations.append(binding)


class TypeParam:
  """Represents a named, generic type.

    >>> TypeParam('foo')
    TypeParam('foo')
    >>> TypeParam('foo') == TypeParam('bar')
    False
    >>> TypeParam('foo') == TypeParam('foo')
    True
    >>> TypeParam('foo') in [TypeParam('foo'), TypeParam('bar')]
    True
    >>> TypeParam('foo') in [TypeParam('bar')]
    False
  """

  def __init__(self, type_symbol: str):
    self.type_symbol = type_symbol

  def __repr__(self):
    return f"TypeParam('{self.type_symbol}')"

  def __eq__(self, other):
    try:
      return other.type_symbol == self.type_symbol
    except AttributeError:
      return False


class OpDef:
  """Context manager that defines a linalg operation."""

  def __init__(self, op_name: str):
    self.op_name = op_name
    self.type_params = {}  # type: Dict[str, TypeParam]
    self.specializations = []  # type: List[Dict[str, ir.Type]]

  @classmethod
  def current_op(cls):
    if not hasattr(_CONTEXT, "op_def"):
      raise AttributeError(
          "The expression requires an active OpDef(...) context")
    return _CONTEXT.op_def

  def add_type_param(self, type_param: TypeParam, exist_ok: bool = False):
    if type_param.type_symbol in self.type_params:
      if not exist_ok:
        raise ValueError(f"Duplicate type parameter: {type_param}")
    else:
      self.type_params[type_param.type_symbol] = type_param

  def __repr__(self):
    lines = [f"OpDef<{self.op_name}>:"]
    for s in self.specializations:
      lines.append(f"  {repr(s)}")
    return "\n".join(lines)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
