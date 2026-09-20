"""Dynamic tool registry for Ultron."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., Any]
    description: str


TOOLS: dict[str, ToolSpec] = {}


def tool(*, name: str | None = None, description: str = ""):
    """Register a Python function as an Ultron tool."""

    def decorator(func: Callable[..., Any]):
        tool_name = name or func.__name__

        if tool_name in TOOLS:
            raise RuntimeError(f"Duplicate tool name: {tool_name}")

        TOOLS[tool_name] = ToolSpec(
            name=tool_name,
            function=func,
            description=(
                description.strip()
                or inspect.getdoc(func)
                or tool_name
            ).strip(),
        )

        return func

    return decorator


def _json_type(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation into a basic JSON schema."""

    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        return _json_type(non_none[0]) if non_none else {"type": "string"}

    if annotation is str:
        return {"type": "string"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is bool:
        return {"type": "boolean"}

    if origin is list:
        return {
            "type": "array",
            "items": _json_type(args[0]) if args else {"type": "string"},
        }

    if origin is dict:
        return {"type": "object"}

    return {"type": "string"}


def schema_for(spec: ToolSpec) -> dict[str, Any]:
    """Build a JSON schema for callers that need manual schemas."""

    sig = inspect.signature(spec.function)

    try:
        hints = get_type_hints(spec.function)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(param_name, param.annotation)
        schema = _json_type(annotation)

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        elif param.default is not None:
            schema["default"] = param.default

        properties[param_name] = schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": parameters,
        },
    }


def all_schemas() -> list[dict[str, Any]]:
    """Return manually generated schemas."""

    return [schema_for(spec) for spec in TOOLS.values()]


def all_functions() -> list[Callable[..., Any]]:
    """
    Return the actual registered Python functions.

    Ollama's Python SDK can convert these callables into tool schemas
    automatically and lets Qwen3 return native tool calls.
    """
    return [spec.function for spec in TOOLS.values()]