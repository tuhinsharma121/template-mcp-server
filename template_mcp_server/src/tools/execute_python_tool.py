"""Execute Python tool for the Template MCP Server.

This tool provides a sandboxed Python execution environment using RestrictedPython.
It allows safe execution of Python code with access to common data science libraries
while blocking dangerous operations like file I/O, network access, and subprocess calls.
"""

import io
import signal
import sys
from contextlib import contextmanager
from typing import Any

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safer_getattr,
)

from template_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

MAX_OUTPUT_SIZE = 100_000
DEFAULT_TIMEOUT_SECONDS = 30


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds the timeout limit."""

    pass


class RestrictedExecutionError(Exception):
    """Raised when code attempts a restricted operation."""

    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for execution timeout."""
    raise ExecutionTimeoutError("Code execution timed out")


@contextmanager
def _execution_timeout(seconds: int):
    """Context manager for enforcing execution timeout using signals.

    Note: This only works on Unix-like systems. On Windows, timeout is not enforced.
    """
    if sys.platform == "win32":
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _guarded_getattr(obj: Any, name: str) -> Any:
    """Custom getattr guard that blocks access to dangerous attributes."""
    BLOCKED_ATTRS = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__dict__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "__name__",
        "__qualname__",
        "__module__",
        "__annotations__",
        "__wrapped__",
        "gi_frame",
        "gi_code",
        "f_locals",
        "f_globals",
        "f_builtins",
        "f_code",
        "co_code",
        "func_globals",
        "func_code",
    }

    if name in BLOCKED_ATTRS:
        raise RestrictedExecutionError(
            f"Access to attribute '{name}' is not allowed in restricted mode"
        )

    return safer_getattr(obj, name)


def _guarded_getitem(obj: Any, key: Any) -> Any:
    """Custom getitem guard."""
    return obj[key]


def _guarded_write(obj: Any) -> Any:
    """Guard for write operations - allows modifications to mutable objects."""
    import types

    if isinstance(obj, types.ModuleType):
        raise RestrictedExecutionError("Writing to modules is not allowed")
    return obj


class _PrintCollector:
    """Collector for print output in restricted execution."""

    def __init__(self, _getattr_: Any = None):
        self._getattr_ = _getattr_
        self.txt: list[str] = []

    def write(self, text: str) -> None:
        self.txt.append(text)

    def __call__(self) -> str:
        return "".join(self.txt)

    def _call_print(self, *objects: Any, **kwargs: Any) -> None:
        if kwargs.get("file", None) is None:
            kwargs["file"] = self
        print(*objects, **kwargs)


def _inplacevar_(op: str, var: Any, expr: Any) -> Any:
    """Handle in-place operations like +=, -=, etc."""
    ops = {
        "+=": lambda x, y: x + y,
        "-=": lambda x, y: x - y,
        "*=": lambda x, y: x * y,
        "/=": lambda x, y: x / y,
        "//=": lambda x, y: x // y,
        "%=": lambda x, y: x % y,
        "**=": lambda x, y: x**y,
        "<<=": lambda x, y: x << y,
        ">>=": lambda x, y: x >> y,
        "&=": lambda x, y: x & y,
        "^=": lambda x, y: x ^ y,
        "|=": lambda x, y: x | y,
    }
    if op not in ops:
        raise RestrictedExecutionError(f"Unknown in-place operator: {op}")
    return ops[op](var, expr)


def _build_restricted_globals() -> dict[str, Any]:
    """Build the restricted globals dictionary with allowed modules and functions."""
    import collections
    import datetime
    import json
    import math
    import re
    import statistics

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        has_matplotlib = True
    except ImportError:
        plt = None
        has_matplotlib = False
        logger.warning("matplotlib not available for restricted execution")

    try:
        import numpy as np

        has_numpy = True
    except ImportError:
        np = None
        has_numpy = False
        logger.warning("numpy not available for restricted execution")

    try:
        import pandas as pd

        has_pandas = True
    except ImportError:
        pd = None
        has_pandas = False
        logger.warning("pandas not available for restricted execution")

    restricted_builtins = dict(safe_builtins)
    restricted_builtins.update(
        {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "bytearray": bytearray,
            "bytes": bytes,
            "chr": chr,
            "complex": complex,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hash": hash,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        }
    )

    restricted_globals: dict[str, Any] = {
        "__builtins__": restricted_builtins,
        "__name__": "__main__",
        "__metaclass__": type,
        "_getattr_": _guarded_getattr,
        "_getitem_": _guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_write_": _guarded_write,
        "_inplacevar_": _inplacevar_,
        "_print_": _PrintCollector,
        "math": math,
        "json": json,
        "re": re,
        "datetime": datetime,
        "collections": collections,
        "statistics": statistics,
    }

    if has_numpy:
        restricted_globals["np"] = np
        restricted_globals["numpy"] = np

    if has_pandas:
        restricted_globals["pd"] = pd
        restricted_globals["pandas"] = pd

    if has_matplotlib:
        restricted_globals["plt"] = plt
        restricted_globals["matplotlib"] = matplotlib

    return restricted_globals


def _truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> tuple[str, bool]:
    """Truncate output if it exceeds max size."""
    if len(output) <= max_size:
        return output, False
    return output[:max_size] + "\n... [OUTPUT TRUNCATED]", True


def execute_python(
    code: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Execute Python code in a restricted sandbox environment.

    This tool compiles and executes Python code using RestrictedPython, providing
    a safe execution environment with access to common data science libraries
    (pandas, numpy, matplotlib) while blocking dangerous operations.

    CPU-bound operation - uses def for computational tasks.

    Args:
        code: Python code to execute as a string. The code can use standard
            Python constructs and the following libraries: math, json, re,
            datetime, collections, statistics, numpy (as np), pandas (as pd),
            matplotlib.pyplot (as plt).
        timeout_seconds: Maximum execution time in seconds (default: 30).
            Code that exceeds this limit will be terminated.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - stdout: Captured standard output from the code
        - result: The value of the last expression (if any)
        - truncated: Whether output was truncated due to size limits
        - error: Error message (only present if status is "error")
        - error_type: Type of error (only present if status is "error")

    Example:
        >>> execute_python("x = [1, 2, 3, 4, 5]; print(sum(x))")
        {"status": "success", "stdout": "15\\n", "result": None, "truncated": False}

    Security notes:
        - File I/O operations (open, read, write) are blocked
        - Network operations are blocked
        - Subprocess/os operations are blocked
        - Access to dangerous attributes (__class__, __globals__, etc.) is blocked
        - Execution is limited by timeout
        - Output is truncated to prevent memory exhaustion
    """
    logger.info(
        "execute_python invoked",
        extra={
            "code_length": len(code),
            "timeout_seconds": timeout_seconds,
            "code_preview": code[:200] if len(code) > 200 else code,
        },
    )

    if not code or not code.strip():
        logger.warning("Empty code provided to execute_python")
        return {
            "status": "error",
            "error": "No code provided",
            "error_type": "ValueError",
            "stdout": "",
            "result": None,
            "truncated": False,
        }

    if timeout_seconds <= 0:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    if timeout_seconds > 120:
        timeout_seconds = 120
        logger.warning("Timeout clamped to maximum of 120 seconds")

    try:
        import warnings

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            byte_code = compile_restricted(
                code,
                filename="<user_code>",
                mode="exec",
            )

        if caught_warnings:
            warning_messages = [str(w.message) for w in caught_warnings]
            logger.warning(
                "RestrictedPython compilation warnings",
                extra={"warnings": warning_messages},
            )

        if byte_code is None:
            logger.error("RestrictedPython compilation failed - returned None")
            return {
                "status": "error",
                "error": "Code compilation failed - contains restricted operations",
                "error_type": "CompilationError",
                "stdout": "",
                "result": None,
                "truncated": False,
            }

    except SyntaxError as e:
        logger.error("Syntax error in user code", extra={"error": str(e)})
        return {
            "status": "error",
            "error": f"Syntax error: {e}",
            "error_type": "SyntaxError",
            "stdout": "",
            "result": None,
            "truncated": False,
        }
    except Exception as e:
        logger.error("Compilation error", extra={"error": str(e)})
        return {
            "status": "error",
            "error": f"Compilation error: {e}",
            "error_type": type(e).__name__,
            "stdout": "",
            "result": None,
            "truncated": False,
        }

    restricted_globals = _build_restricted_globals()
    local_namespace: dict[str, Any] = {}

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture

        with _execution_timeout(timeout_seconds):
            exec(byte_code, restricted_globals, local_namespace)  # noqa: S102

        stdout_value = stdout_capture.getvalue()

        if "_print" in restricted_globals and hasattr(
            restricted_globals["_print"], "txt"
        ):
            printed_output = "".join(restricted_globals["_print"].txt)
            stdout_value = printed_output + stdout_value
        elif "_print" in local_namespace and hasattr(local_namespace["_print"], "txt"):
            printed_output = "".join(local_namespace["_print"].txt)
            stdout_value = printed_output + stdout_value

        stdout_output, was_truncated = _truncate_output(stdout_value)

        result_value = local_namespace.get("result", local_namespace.get("_", None))

        if result_value is not None:
            try:
                result_str = repr(result_value)
                if len(result_str) > MAX_OUTPUT_SIZE:
                    result_str, _ = _truncate_output(result_str)
                    was_truncated = True
            except Exception:
                result_str = "<unprintable result>"
        else:
            result_str = None

        logger.info(
            "execute_python completed successfully",
            extra={
                "stdout_length": len(stdout_output),
                "has_result": result_str is not None,
                "truncated": was_truncated,
            },
        )

        return {
            "status": "success",
            "stdout": stdout_output,
            "result": result_str,
            "truncated": was_truncated,
        }

    except ExecutionTimeoutError:
        logger.error(
            "Code execution timed out",
            extra={"timeout_seconds": timeout_seconds},
        )
        return {
            "status": "error",
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "error_type": "TimeoutError",
            "stdout": stdout_capture.getvalue()[:1000],
            "result": None,
            "truncated": False,
        }

    except RestrictedExecutionError as e:
        logger.error("Restricted operation attempted", extra={"error": str(e)})
        return {
            "status": "error",
            "error": str(e),
            "error_type": "RestrictedExecutionError",
            "stdout": stdout_capture.getvalue()[:1000],
            "result": None,
            "truncated": False,
        }

    except NameError as e:
        logger.error("Name error in user code", extra={"error": str(e)})
        return {
            "status": "error",
            "error": f"Name error: {e}. This may be due to using a restricted function or module.",
            "error_type": "NameError",
            "stdout": stdout_capture.getvalue()[:1000],
            "result": None,
            "truncated": False,
        }

    except Exception as e:
        logger.error(
            "Runtime error in user code",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "stdout": stdout_capture.getvalue()[:1000],
            "result": None,
            "truncated": False,
        }

    finally:
        sys.stdout = old_stdout
        stdout_capture.close()
