"""Safe arithmetic tool."""
from __future__ import annotations
import ast, operator as op
from .registry import tool
_BIN={ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.FloorDiv:op.floordiv,ast.Mod:op.mod,ast.Pow:op.pow}
_UNARY={ast.UAdd:op.pos,ast.USub:op.neg}
def _eval(n):
    if isinstance(n,ast.Expression): return _eval(n.body)
    if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
    if isinstance(n,ast.BinOp) and type(n.op) in _BIN:
        r=_eval(n.right)
        if isinstance(n.op,ast.Pow) and abs(r)>100: raise ValueError("Exponent too large")
        return _BIN[type(n.op)](_eval(n.left),r)
    if isinstance(n,ast.UnaryOp) and type(n.op) in _UNARY: return _UNARY[type(n.op)](_eval(n.operand))
    raise ValueError("Only basic arithmetic is allowed")
@tool(description="Perform basic arithmetic safely.")
def calculate(expression: str) -> str:
    """Calculate an arithmetic expression.

    Args:
        expression: Expression such as '(25 + 37) * 2'.
    """
    try: return str(_eval(ast.parse(expression,mode="eval")))
    except Exception as exc: return f"Calculation failed: {exc}"
