"""Defines types and type variables relevant to TC expressions."""

from enum import Enum
from typing import Dict

__all__ = [
    "TypeClass",
    "TypePredicate",
    "TypeVar",
    "TV",

    # TypeVar aliases.
    "T",
    "U",
    "V",

    # TypePredicate built-ins.
    "AnyFloat",
    "AnyInteger",
    "AnyReal",
]


class TypeVar:
  """A replaceable type variable.

  Type variables are uniqued by name.
  """
  ALL_TYPEVARS = dict()  # type: Dict[str, "TypeVar"]

  def __new__(cls, name: str):
    existing = cls.ALL_TYPEVARS.get(name)
    if existing is not None:
      return existing
    new = super().__new__(cls)
    new.name = name
    cls.ALL_TYPEVARS[name] = new
    return new

  def __repr__(self):
    return f"TypeVar({self.name})"

  @classmethod
  def create_expando(cls):
    """Create an expando class that creates unique type vars on attr access."""

    class ExpandoTypeVars:

      def __getattr__(self, n):
        return cls(n)

    return ExpandoTypeVars()


# Expando access via TV.foo
TV = TypeVar.create_expando()

# Some common type name aliases.
T = TV.T
U = TV.U
V = TV.V


class TypeClass(Enum):
  """Enumeration of numeric types.

  A different TypeClass is indicated if std dialect math operations defined
  for it are non-polymorphic.
  """
  # A floating point number of any type.
  FLOAT = 1

  # An integer that is treated as signed in contexts where the distinction
  # matters.
  INTEGER = 2

  # An integer that is treated as unsigned in contexts where the distinction
  # matters.
  # TODO: This is not fully thought through and may not compose properly.
  # We may not be able to support generalized type polymorphism in this way.
  UNSIGNED_INTEGER = 3

  # A complex number.
  COMPLEX = 3


class TypePredicate:
  """Tuple of a TypeClass and constraints that further narrow what is valid.

  TODO: Add general constraints.
  """

  def __init__(self, *type_classes: TypeClass):
    self.type_classes = type_classes

  def __repr__(self):
    return f"TypePredicate({', '.join(repr(c) for c in self.type_classes)})"


# Some common predicates.
AnyFloat = TypePredicate(TypeClass.FLOAT)
AnyInteger = TypePredicate(TypeClass.INTEGER)
AnyReal = TypePredicate(TypeClass.FLOAT, TypeClass.INTEGER)
