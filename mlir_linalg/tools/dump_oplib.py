#!/usr/bin/which python
# Command line tool to load an oplib module and dump all of the operations
# it contains in some format.

import argparse
import importlib

from mlir_linalg.dsl.tc import *
from mlir_linalg.dsl.linalg_op_config import *
from mlir_linalg.dsl.yaml_helper import *


def create_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Dump an oplib in various formats")
  p.add_argument("modules",
                 metavar="M",
                 type=str,
                 nargs="+",
                 help="Op module to dump")
  p.add_argument("--format",
                 type=str,
                 dest="format",
                 default="yaml",
                 choices=("yaml", "repr"),
                 help="Format in which to dump")
  return p


def main(args):
  # Load all configs.
  configs = []
  for module_name in args.modules:
    m = importlib.import_module(module_name)
    for attr_name, value in m.__dict__.items():
      # TODO: This class layering is awkward.
      if isinstance(value, TcEmitGenericCallable):
        try:
          linalg_config = LinalgOpConfig.from_tc_op_def(value.model)
        except Exception as e:
          raise ValueError(
              f"Could not create LinalgOpConfig from {value.model}") from e
        configs.extend(linalg_config)

  # Print.
  if args.format == "yaml":
    print(yaml_dump_all(configs))
  elif args.format == "repr":
    for config in configs:
      print(repr(config))


if __name__ == "__main__":
  main(create_arg_parser().parse_args())
