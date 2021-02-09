"""DSL for constructing affine expressions and maps.

Affine expression construction:
  >>> with _ir.Context():
  ...   s = AffineBuildState()
  ...   (S.K + S.M).build(s)
  ...   (S.K * S.M).build(s)
  ...   (S.K // S.M).build(s)
  ...   (S.K / S.M).build(s)
  ...   (S.K % 4).build(s)
  ...   (D.i + D.j * 4).build(s)
  ...   s
  AffineExpr(s0 + s1)
  AffineExpr(s0 * s1)
  AffineExpr(s0 floordiv s1)
  AffineExpr(s0 ceildiv s1)
  AffineExpr(s0 mod 4)
  AffineExpr(d0 + d1 * 4)
  AffineBuildState<
    symbols={'K': 0, 'M': 1}
    dims={'i': 0, 'j': 1}>
"""

from typing import Callable, Dict, Optional, Tuple, Union

from mlir import ir as _ir

__all__ = [
    "AffineBuildState",
    "AffineExprDef",
    "D",
    "DimDef",
    "S",
    "SymbolDef",
]

# Type aliases.
SymbolPosMap = Dict[str, int]


class AffineBuildState:
  """Internal state for the AffineExprDef._create impls."""

  def __init__(self,
               *,
               global_state: "AffineBuildState" = None,
               allow_new_symbols: bool = True,
               allow_new_dims: bool = True):
    if not global_state:
      self.all_symbols = dict()  # type: Dict[str, int]
      self.all_dims = dict()  # type: Dict[str, int]
    else:
      # Alias the global dict.
      self.all_symbols = global_state.all_symbols
      self.all_dims = global_state.all_dims

    # Map of symbols and dims in the current build.
    self.symbol_pos_map = dict()  # type: Dict[str, int]
    self.dim_pos_map = dict()  # type: Dict[str, int]
    self.allow_new_symbols = allow_new_symbols
    self.allow_new_dims = allow_new_dims

  def get_dim(self, dimname: str) -> int:
    """Gets the dim position given a name."""
    pos = self.all_dims.get(dimname)
    if pos is None:
      if not self.allow_new_dims:
        raise ValueError(
            f"New dimensions not allowed in the current affine expression: "
            f"Requested '{dimname}', Availble: {self.all_dims}")
      pos = len(self.all_dims)
      self.all_dims[dimname] = pos
    self.dim_pos_map[dimname] = pos
    return pos

  def get_symbol(self, symname: str) -> int:
    """Geta a symbol position given a name."""
    pos = self.all_symbols.get(symname)
    if pos is None:
      if not self.allow_new_symbols:
        raise ValueError(
            f"New symbols not allowed in the current affine expression: "
            f"Requested '{symname}', Availble: {self.all_symbols}")
      pos = len(self.all_symbols)
      self.all_symbols[symname] = pos
    self.symbol_pos_map[symname] = pos
    return pos

  @property
  def dim_count(self):
    return len(self.dim_pos_map)

  @property
  def symbol_count(self):
    return len(self.symbol_pos_map)

  def __repr__(self):
    lines = [f"AffineBuildState<"]
    lines.append(f"  symbols={self.symbol_pos_map}")
    lines.append(f"  dims={self.dim_pos_map}>")
    return "\n".join(lines)


class AffineExprDef:
  """Base class for an affine expression being defined."""

  def build(self, state: Optional[AffineBuildState] = None) -> _ir.AffineExpr:
    """Builds the corresponding _ir.AffineExpr from the definitions.
    """
    state = AffineBuildState() if state is None else state
    expr = self._create(state)
    return expr

  def _create(self, state: AffineBuildState) -> _ir.AffineExpr:
    raise NotImplementedError()

  @staticmethod
  def coerce_from(py_value):
    if isinstance(py_value, int):
      return AffineConstantExpr(py_value)
    assert isinstance(py_value, AffineExprDef)
    return py_value

  def visit_affine_exprs(self, callback):
    """Visits all AffineExprDefs including self."""
    callback(self)

  def __add__(lhs, rhs):
    rhs = AffineExprDef.coerce_from(rhs)
    return AffineBinaryExprDef(_ir.AffineAddExpr, lhs, rhs)

  def __mul__(lhs, rhs):
    rhs = AffineExprDef.coerce_from(rhs)
    return AffineBinaryExprDef(_ir.AffineMulExpr, lhs, rhs)

  def __mod__(lhs, rhs):
    rhs = AffineExprDef.coerce_from(rhs)
    return AffineBinaryExprDef(_ir.AffineModExpr, lhs, rhs)

  def __floordiv__(lhs, rhs):
    rhs = AffineExprDef.coerce_from(rhs)
    return AffineBinaryExprDef(_ir.AffineFloorDivExpr, lhs, rhs)

  def __truediv__(lhs, rhs):
    # TODO: Not really a ceil div - taking liberties for the DSL.
    rhs = AffineExprDef.coerce_from(rhs)
    return AffineBinaryExprDef(_ir.AffineCeilDivExpr, lhs, rhs)


class AffineConstantExpr(AffineExprDef):
  """An affine constant being defined."""

  def __init__(self, value: int):
    assert isinstance(value, int)
    self.value = value

  def _create(self, state: AffineBuildState) -> _ir.AffineExpr:
    return _ir.AffineConstantExpr.get(self.value)


class AffineBinaryExprDef(AffineExprDef):
  """An affine binary expression being defined."""

  def __init__(self, ir_ctor, lhs: AffineExprDef, rhs: AffineExprDef):
    self.ir_ctor = ir_ctor
    self.lhs = lhs
    self.rhs = rhs

  def _create(self, state: AffineBuildState) -> _ir.AffineExpr:
    return self.ir_ctor.get(self.lhs._create(state), self.rhs._create(state))

  def visit_affine_exprs(self, callback):
    """Visits all AffineExprDefs including self."""
    super().visit_affine_exprs(callback)
    self.lhs.visit_affine_exprs(callback)
    self.rhs.visit_affine_exprs(callback)

  def __repr__(self):
    return f"{self.ir_ctor.__name__}({repr(self.lhs)}, {repr(self.rhs)})"


class DimDef(AffineExprDef):
  """Represents a named dimension.

  """
  ALL_DIMS = dict()  # type: Dict[str, "DimDef"]

  def __new__(cls, dimname: str):
    existing = cls.ALL_DIMS.get(dimname)
    if existing is not None:
      return existing
    new = super().__new__(cls)
    new.dimname = dimname
    cls.ALL_DIMS[dimname] = new
    return new

  def __repr__(self):
    return f"Dim({self.dimname})"

  def _create(self, state: AffineBuildState) -> _ir.AffineExpr:
    pos = state.get_dim(self.dimname)
    return _ir.AffineDimExpr.get(position=pos)

  @classmethod
  def create_expando(cls):
    """Create an expando class that creates unique symbols based on attr access.
      >>> D.i
      Dim(i)
      >>> D.a is D.b
      False
      >>> D.a is D.a
      True
    """

    class ExpandoDims:

      def __getattr__(self, n):
        return cls(n)

    return ExpandoDims()


D = DimDef.create_expando()


class SymbolDef(AffineExprDef):
  """Represents a named symbol.

    >>> s1 = SymbolDef("s1")
    >>> s1
    Symbol(s1)
    >>> s2 = SymbolDef("s2")
    >>> s1 is s2
    False
    >>> s1 is SymbolDef("s1")
    True
  """
  ALL_SYMBOLS = dict()  # type: Dict[str, "SymbolDef"]

  def __new__(cls, symname: str):
    existing = cls.ALL_SYMBOLS.get(symname)
    if existing is not None:
      return existing
    new = super().__new__(cls)
    new.symname = symname
    cls.ALL_SYMBOLS[symname] = new
    return new

  def __repr__(self):
    return f"Symbol({self.symname})"

  def _create(self, state: AffineBuildState) -> _ir.AffineExpr:
    pos = state.get_symbol(self.symname)
    return _ir.AffineSymbolExpr.get(position=pos)

  @classmethod
  def create_expando(cls):
    """Create an expando class that creates unique symbols based on attr access.
      >>> S.foobar
      Symbol(foobar)
      >>> S.a is S.b
      False
      >>> S.a is S.a
      True
    """

    class ExpandoSymbols:

      def __getattr__(self, n):
        return cls(n)

    return ExpandoSymbols()


# Global accessor for on-demand symbols.
S = SymbolDef.create_expando()

if __name__ == "__main__":
  import doctest
  doctest.testmod()
