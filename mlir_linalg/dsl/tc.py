from typing import Dict, List

from contextlib import contextmanager
import functools
import inspect
import threading

from mlir import ir
from .tc_model import *
from ..ir_gen.emitter import *

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

  tc_model = TcOpDef(name=op_name,
                     cpp_op_name=op_class_name,
                     doc=inspect.getdoc(dsl_func))

  # Extract arguments and TensorDefs from the signature.
  dsl_func_args = list()
  sig = inspect.signature(dsl_func)
  for param_name, param in sig.parameters.items():
    param_default = param.default
    if not isinstance(param_default, TensorDef):
      raise ValueError(f"@tc_def_op function parameters must be defaulted as "
                       f"TensorDef(...): Found {param_name}: {param_default}")
    dsl_func_args.append(param_default)
    tc_model.add_tensor(param_name, param_default)

  # Invoke the DSL func to finish populating the model.
  # TODO: Wrap in context manager.
  dsl_func(*dsl_func_args)

  return TcEmitGenericCallable(op_name, tc_model)
