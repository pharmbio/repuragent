#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import asyncio
import builtins
import difflib
import inspect
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from functools import wraps
from importlib import import_module
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]

def custom_print(*args):
    return None

BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

MAX_LENGTH_TRUNCATE_CONTENT = 20000

def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


class InterpreterError(ValueError):
    '''An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    '''

    pass


ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}

# Print output is transcript content: it is replayed to the model on every
# later call in the run, so the old 5,000,000-char ceiling was no ceiling at
# all. Callers may still override it per executor.
DEFAULT_MAX_LEN_OUTPUT = 20000
MAX_OPERATIONS = 1000000000
MAX_WHILE_ITERATIONS = 100000000



def custom_print(*args):
    return None


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

DANGEROUS_FUNCTIONS = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]

# Builtins we never resolve by bare name (dangerous, or intentionally replaced by
# an injected/scoped equivalent such as `open`). Everything else in `builtins`
# is fair game — see `resolve_builtin`.
_BUILTIN_DENYLIST = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "input",
    "help",
    "exit",
    "quit",
    "breakpoint",
    "memoryview",
}


def resolve_builtin(name: str):
    '''Return the Python builtin for `name`, or None if it's absent or denylisted.

    This is the flexibility escape hatch: any safe builtin (``repr``, ``bytes``,
    ``hex``, ``frozenset``, ``format``, ``vars``, …) resolves instead of raising
    "it is not permitted to evaluate other functions". Dangerous builtins stay
    unreachable because they are denylisted here (and blocked again by
    ``safer_eval`` / ``is_dangerous_callable``).

    Parameters:
    ---------
    name (str): the builtin the interpreted code referred to.

    Returns:
    ----------
    builtin (callable): the builtin, or None when it is absent or on the denylist.
    '''

    if name in _BUILTIN_DENYLIST:
        return None
    return getattr(builtins, name, None)


def is_dangerous_callable(func: Any) -> bool:
    '''True when `func` is one of the explicitly forbidden functions.

    Parameters:
    ---------
    func (Any): the callable the interpreted code is about to invoke.

    Returns:
    ----------
    dangerous (boolean): True when it is one of the explicitly forbidden functions.
    '''

    for qualified in DANGEROUS_FUNCTIONS:
        module_name, function_name = qualified.rsplit(".", 1)
        if getattr(func, "__name__", None) == function_name and getattr(func, "__module__", None) == module_name:
            return True
    return False


# ---------------------------------------------------------------------------
# Async support
# ---------------------------------------------------------------------------
# The interpreter is a synchronous tree-walker, so it cannot itself suspend on
# `await`. Instead we keep ONE event loop running on a background thread for the
# whole process and drive every awaitable to completion on it. Two payoffs:
#   1. `async def` / `await` / `asyncio.run(...)` written by the model just work.
#   2. Every coroutine runs on the *same* loop across separate executor cells,
#      so connection pools bound to that loop (e.g. the ECHA httpx client) stay
#      valid — no "Event loop is closed" errors between calls.

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ASYNC_LOOP_LOCK = threading.Lock()


def _get_async_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP
    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
            _ASYNC_LOOP = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_ASYNC_LOOP.run_forever,
                name="local-executor-async-loop",
                daemon=True,
            )
            thread.start()
        return _ASYNC_LOOP


async def _await_value(awaitable: Any) -> Any:
    return await awaitable


def drive_awaitable(value: Any) -> Any:
    '''Run an awaitable to completion on the shared background loop and return its
    result. Non-awaitables are returned unchanged, which makes ``asyncio.run``
    tolerant of code that passes an already-resolved value.

    Parameters:
    ---------
    value (Any): the awaitable produced inside the interpreter.

    Returns:
    ----------
    result (any): what it resolved to on the shared background loop.
    '''

    if not inspect.isawaitable(value):
        return value
    loop = _get_async_loop()
    future = asyncio.run_coroutine_threadsafe(_await_value(value), loop)
    return future.result()


def drive_gather(*awaitables: Any, return_exceptions: bool = False) -> list:
    '''`asyncio.gather` replacement. Interpreter-defined ``async def``s resolve
    eagerly to plain values, so `gather` may receive a mix of values and real
    coroutines — drive each to a result. Runs sequentially (no concurrency),
    which is acceptable and matches how these tools are meant to be called.

    Parameters:
    ---------
    *awaitables (awaitable): the awaitables to run concurrently.
    return_exceptions (boolean): return a raised exception in place of a result rather than propagating it.

    Returns:
    ----------
    results (list): one entry per awaitable, in the order given.
    '''

    results = []
    for awaitable in awaitables:
        if inspect.isawaitable(awaitable):
            if return_exceptions:
                try:
                    results.append(drive_awaitable(awaitable))
                except Exception as exc:  # noqa: BLE001 - mirror gather semantics
                    results.append(exc)
            else:
                results.append(drive_awaitable(awaitable))
        else:
            results.append(awaitable)
    return results


class PrintContainer:
    def __init__(self):
        self.value = ""

    def append(self, text):
        self.value += text
        return self

    def __iadd__(self, other):
        '''Implements the += operator

        Parameters:
        ---------
        other (str): text to append to the captured output.

        Returns:
        ----------
        self (PrintContainer): the same container, as `+=` requires.
        '''

        self.value += str(other)
        return self

    def __str__(self):
        '''String representation

        Returns:
        ----------
        text (str): everything printed so far.
        '''

        return self.value

    def __repr__(self):
        '''Representation for debugging

        Returns:
        ----------
        text (str): everything printed so far, for debugging.
        '''

        return f"PrintContainer({self.value})"

    def __len__(self):
        '''Implements len() function support

        Returns:
        ----------
        length (int): the number of characters captured.
        '''

        return len(self.value)


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


def safer_eval(func: Callable):
    '''Decorator to make the evaluation of a function safer by checking its return value.

    Parameters:
    ---------
    func (Callable): Function to make safer.

    Returns:
    ----------
    wrapped (callable): Safer function with return value check.
    '''

    @wraps(func)
    def _check_return(expression, state, static_tools, custom_tools,
                      authorized_imports=BASE_BUILTIN_MODULES):

        def is_allowed_module(name: str) -> bool:
            # allow exact matches or any submodule of an allowed package
            return any(a == "*" or name == a or name.startswith(a + ".")
                       for a in authorized_imports)

        result = func(expression, state, static_tools, custom_tools,
                      authorized_imports=authorized_imports)

        if "*" not in authorized_imports:
            if isinstance(result, ModuleType):
                if not is_allowed_module(result.__name__):
                    raise InterpreterError(f"Forbidden access to module: {result.__name__}")

            elif isinstance(result, dict) and result.get("__spec__"):
                name = result.get("__name__", "")
                if not is_allowed_module(name):
                    raise InterpreterError(f"Forbidden access to module: {name}")

            elif isinstance(result, (FunctionType, BuiltinFunctionType)):
                # Still block explicitly dangerous functions by qualified name
                for qualified in DANGEROUS_FUNCTIONS:
                    module_name, function_name = qualified.rsplit(".", 1)
                    if (
                        function_name not in static_tools
                        and result.__name__ == function_name
                        and result.__module__ == module_name
                    ):
                        raise InterpreterError(f"Forbidden access to function: {function_name}")

        return result
    return _check_return


# Only these dunder attributes are blocked — they are the introspection vectors
# used to break out of the sandbox (e.g. ``().__class__.__bases__[0].__subclasses__()``).
# Benign dunders like ``__name__``, ``__class__``, ``__doc__``, ``__dict__`` are
# allowed so ordinary generated code (``type(e).__name__``) is not rejected.
_FORBIDDEN_DUNDERS = {
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__base__",
    "__code__",
    "__closure__",
    "__getattribute__",
    "__reduce__",
    "__reduce_ex__",
    "__import__",
}


def evaluate_attribute(
    expression: ast.Attribute,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if expression.attr in _FORBIDDEN_DUNDERS:
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)


def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > MAX_WHILE_ITERATIONS:
            raise InterpreterError(f"Maximum number of {MAX_WHILE_ITERATIONS} iterations in While loop exceeded")
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    source_code = ast.unparse(func_def)

    def new_func(*args: Any, **kwargs: Any) -> Any:
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    # Store original AST, source code, and name
    new_func.__ast__ = func_def
    new_func.__source__ = source_code
    new_func.__name__ = func_def.name

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> type:
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]
    class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            class_dict[stmt.name] = evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
                elif isinstance(target, ast.Attribute):
                    class_dict[target.attr] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
        elif isinstance(stmt, ast.AnnAssign):
            # `x: int = 1` in a class body — bind only when a value is present.
            if isinstance(stmt.target, ast.Name) and stmt.value is not None:
                class_dict[stmt.target.id] = evaluate_ast(
                    stmt.value, state, static_tools, custom_tools, authorized_imports
                )
        elif isinstance(stmt, (ast.Pass, ast.Expr)):
            # Bare `pass` or a docstring / expression statement — nothing to bind.
            continue
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = type(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_augassign(
    expression: ast.AugAssign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    def get_current_value(target: ast.AST) -> Any:
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state: current_value has been updated in-place
    set_value(
        expression.target,
        current_value,
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )

    return current_value


def evaluate_boolop(
    node: ast.BoolOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool:
    if isinstance(node.op, ast.And):
        for value in node.values:
            if not evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return False
        return True
    elif isinstance(node.op, ast.Or):
        for value in node.values:
            if evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return True
        return False


def evaluate_binop(
    binop: ast.BinOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    # Recursively evaluate the left and right operands
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # Determine the operation based on the type of the operator in the BinOp
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)

        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if not isinstance(call.func, (ast.Call, ast.Lambda, ast.Attribute, ast.Name, ast.Subscript)):
        raise InterpreterError(f"This is not a correct function: {call.func}).")

    func, func_name = None, None

    if isinstance(call.func, ast.Call):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            builtin = resolve_builtin(func_name)
            if builtin is not None:
                func = builtin
            else:
                raise InterpreterError(
                    f"It is not permitted to evaluate other functions than the provided tools or functions defined/imported in previous code (tried to execute {call.func.id})."
                )
    elif isinstance(call.func, ast.Subscript):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None

    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {
        keyword.arg: evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
        for keyword in call.keywords
    }

    if func_name == "super":
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            else:
                raise InterpreterError("super() needs at least one argument")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        elif len(args) == 2:
            instance = args[1]
            return super(cls, instance)
        else:
            raise InterpreterError("super() takes at most 2 arguments")
    elif func_name == "print":
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None
    else:  # Assume it's a callable object
        if is_dangerous_callable(func):
            raise InterpreterError(f"Forbidden access to function: {func_name}")
        return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Could not index {value} with '{index}': {type(e).__name__}: {e}"
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f". Maybe you meant one of these indexes instead: {str(close_matches)}"
        raise InterpreterError(error_message) from e


def evaluate_name(
    name: ast.Name,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if name.id in state:
        return state[name.id]
    elif name.id in static_tools:
        return static_tools[name.id]
    elif name.id in custom_tools:
        return custom_tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    builtin = resolve_builtin(name.id)
    if builtin is not None:
        return builtin
    # Suggest, never substitute. Silently returning the nearest-named variable
    # turns a typo into a wrong number with no error: `herg_scores` would quietly
    # evaluate to `hergs_scores`. The caller must name the variable it means.
    close_matches = difflib.get_close_matches(
        name.id, [key for key in state if not key.startswith("_")]
    )
    if close_matches:
        raise InterpreterError(
            f"The variable `{name.id}` is not defined. Did you mean one of these? "
            f"{close_matches}"
        )
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool | object:
    result = True
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    for i, (op, comparator) in enumerate(zip(condition.ops, condition.comparators)):
        op = type(op)
        right = evaluate_ast(comparator, state, static_tools, custom_tools, authorized_imports)
        if op == ast.Eq:
            current_result = left == right
        elif op == ast.NotEq:
            current_result = left != right
        elif op == ast.Lt:
            current_result = left < right
        elif op == ast.LtE:
            current_result = left <= right
        elif op == ast.Gt:
            current_result = left > right
        elif op == ast.GtE:
            current_result = left >= right
        elif op == ast.Is:
            current_result = left is right
        elif op == ast.IsNot:
            current_result = left is not right
        elif op == ast.In:
            current_result = left in right
        elif op == ast.NotIn:
            current_result = left not in right
        else:
            raise InterpreterError(f"Unsupported comparison operator: {op}")

        if current_result is False:
            return False
        result = current_result if i == 0 else (result and current_result)
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                break
            except ContinueException:
                continue
        else:
            continue
        break
    return result


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> List[Any]:
    def inner_evaluate(generators: List[ast.comprehension], index: int, current_state: Dict[str, Any]) -> List[Any]:
        if index >= len(generators):
            return [
                evaluate_ast(
                    listcomp.elt,
                    current_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
            ]
        generator = generators[index]
        iter_value = evaluate_ast(
            generator.iter,
            current_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        result = []
        for value in iter_value:
            new_state = current_state.copy()
            if isinstance(generator.target, ast.Tuple):
                for idx, elem in enumerate(generator.target.elts):
                    new_state[elem.id] = value[idx]
            else:
                new_state[generator.target.id] = value
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in generator.ifs
            ):
                result.extend(inner_evaluate(generators, index + 1, new_state))
        return result

    return inner_evaluate(listcomp.generators, 0, state)


def evaluate_setcomp(
    setcomp: ast.SetComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Set[Any]:
    result = set()
    for gen in setcomp.generators:
        iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
        for value in iter_value:
            new_state = state.copy()
            set_value(
                gen.target,
                value,
                new_state,
                static_tools,
                custom_tools,
                authorized_imports,
            )
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in gen.ifs
            ):
                element = evaluate_ast(
                    setcomp.elt,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                result.add(element)
    return result


def evaluate_try(
    try_node: ast.Try,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(
                e,
                evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports),
            ):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(
    with_node: ast.With,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        if item.optional_vars:
            state[item.optional_vars.id] = context_expr.__enter__()
            contexts.append(state[item.optional_vars.id])
        else:
            context_var = context_expr.__enter__()
            contexts.append(context_var)

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        for context in reversed(contexts):
            context.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def get_safe_module(raw_module, authorized_imports, visited=None):
    '''Creates a safe copy of a module or returns the original if it's a function

    Parameters:
    ---------
    raw_module (module): the module the interpreted code imported.
    authorized_imports (list): the import allowlist this session runs under.
    visited (set): modules already wrapped, to terminate on circular references.

    Returns:
    ----------
    module (module): a safe copy, or the original when it is a function or class rather than a module.
    '''

    # If it's a function or non-module object, return it directly
    if not isinstance(raw_module, ModuleType):
        return raw_module

    # Handle circular references: Initialize visited set for the first call
    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module  # Return original for circular refs

    visited.add(module_id)

    # Create new module for actual modules. Some lazily-loaded modules (e.g.
    # TensorFlow's LazyLoader when Keras 3 is present) raise on ``.__name__``,
    # so fall back to a placeholder rather than letting that abort the import.
    module_name = getattr(raw_module, "__name__", "module")
    safe_module = ModuleType(module_name)

    # Copy all attributes by reference, recursively checking modules
    for attr_name in dir(raw_module):
        try:
            attr_value = getattr(raw_module, attr_name)
            # Recursively process nested modules, passing visited set. Kept
            # inside the try so a submodule that errors while being copied
            # (lazy loaders, optional deps) is skipped, not fatal to the whole
            # import.
            if isinstance(attr_value, ModuleType):
                attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)
        except Exception as e:
            # lazy / dynamic loading module -> INFO log and skip
            logger.info(
                f"Skipping error while copying {module_name}.{attr_name}: {type(e).__name__} - {e}"
            )
            continue

        setattr(safe_module, attr_name, attr_value)

    if module_name == "asyncio":
        # Route asyncio.run/gather through the shared background loop so they work
        # from the interpreter thread and reuse one loop across executor cells.
        safe_module.run = drive_awaitable
        safe_module.gather = drive_gather

    return safe_module


def check_module_authorized(module_name, authorized_imports):
    if "*" in authorized_imports:
        return True
    else:
        module_path = module_name.split(".")
        # ["A", "B", "C"] -> ["A", "A.B", "A.B.C"]
        module_subpaths = [".".join(module_path[:i]) for i in range(1, len(module_path) + 1)]
        return any(subpath in authorized_imports for subpath in module_subpaths)


def evaluate_import(expression, state, authorized_imports):
    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_module_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        if check_module_authorized(expression.module, authorized_imports):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":  # Handle "from module import *"
                if hasattr(module, "__all__"):  # If module has __all__, import only those names
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:  # If no __all__, import all public names (those not starting with '_')
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:  # regular from imports
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module {expression.module} has no attribute {alias.name}")
        else:
            raise InterpreterError(
                f"Import from {expression.module} is not allowed. Authorized imports are: {str(authorized_imports)}"
            )
        return None


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Dict[Any, Any]:
    result = {}
    for gen in dictcomp.generators:
        iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
        for value in iter_value:
            new_state = state.copy()
            set_value(
                gen.target,
                value,
                new_state,
                static_tools,
                custom_tools,
                authorized_imports,
            )
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in gen.ifs
            ):
                key = evaluate_ast(
                    dictcomp.key,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                val = evaluate_ast(
                    dictcomp.value,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                result[key] = val
    return result


def evaluate_delete(
    delete_node: ast.Delete,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    '''Evaluate a delete statement (del x, del x[y]).

    Parameters:
    ---------
    delete_node (ast.Delete): The AST Delete node to evaluate
    state (dict): The current state dictionary
    static_tools (dict): Dictionary of static tools
    custom_tools (dict): Dictionary of custom tools
    authorized_imports (list): List of authorized imports
    '''

    for target in delete_node.targets:
        if isinstance(target, ast.Name):
            # Handle simple variable deletion (del x)
            if target.id in state:
                del state[target.id]
            else:
                raise InterpreterError(f"Cannot delete name '{target.id}': name is not defined")
        elif isinstance(target, ast.Subscript):
            # Handle index/key deletion (del x[y])
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            try:
                del obj[index]
            except (TypeError, KeyError, IndexError) as e:
                raise InterpreterError(f"Cannot delete index/key: {str(e)}")
        else:
            raise InterpreterError(f"Deletion of {type(target).__name__} targets is not supported")


def evaluate_annassign(
    node: ast.AnnAssign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    '''Annotated assignment: `x: int = 5` (or a bare annotation `x: int`).

    Parameters:
    ---------
    node (ast.AnnAssign): the `x: int = 5` node, or a bare `x: int` annotation.
    state (dict): the interpreter namespace.
    static_tools (dict): tools the interpreted code may call but not rebind.
    custom_tools (dict): tools defined for this session.
    authorized_imports (list): the import allowlist this session runs under.

    Returns:
    ----------
    value (any): the assigned value, or None for a bare annotation.
    '''

    if node.value is None:
        # Pure annotation with no value — nothing to bind.
        return None
    result = evaluate_ast(node.value, state, static_tools, custom_tools, authorized_imports)
    set_value(node.target, result, state, static_tools, custom_tools, authorized_imports)
    return result


def evaluate_namedexpr(
    node: ast.NamedExpr,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    '''Walrus operator: `(y := f(x))`.

    Parameters:
    ---------
    node (ast.NamedExpr): the `(y := f(x))` node.
    state (dict): the interpreter namespace.
    static_tools (dict): tools the interpreted code may call but not rebind.
    custom_tools (dict): tools defined for this session.
    authorized_imports (list): the import allowlist this session runs under.

    Returns:
    ----------
    value (any): the value bound by the walrus, which is also the expression's value.
    '''

    value = evaluate_ast(node.value, state, static_tools, custom_tools, authorized_imports)
    set_value(node.target, value, state, static_tools, custom_tools, authorized_imports)
    return value


def evaluate_await(
    node: ast.Await,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    '''`await expr` — resolve the awaitable on the shared background loop.

    Parameters:
    ---------
    node (ast.Await): the `await expr` node.
    state (dict): the interpreter namespace.
    static_tools (dict): tools the interpreted code may call but not rebind.
    custom_tools (dict): tools defined for this session.
    authorized_imports (list): the import allowlist this session runs under.

    Returns:
    ----------
    value (any): what the awaitable resolved to on the shared background loop.
    '''

    value = evaluate_ast(node.value, state, static_tools, custom_tools, authorized_imports)
    return drive_awaitable(value)


def evaluate_async_for(
    for_loop: ast.AsyncFor,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    '''`async for x in aiter: ...` driven synchronously via the background loop.

    Parameters:
    ---------
    for_loop (ast.AsyncFor): the `async for x in aiter: ...` node.
    state (dict): the interpreter namespace.
    static_tools (dict): tools the interpreted code may call but not rebind.
    custom_tools (dict): tools defined for this session.
    authorized_imports (list): the import allowlist this session runs under.

    Returns:
    ----------
    value (any): whatever a `break` or `return` inside the loop produced, otherwise None.
    '''

    result = None
    iterable = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    async_iter = iterable.__aiter__()
    while True:
        try:
            item = drive_awaitable(async_iter.__anext__())
        except StopAsyncIteration:
            break
        set_value(for_loop.target, item, state, static_tools, custom_tools, authorized_imports)
        broke = False
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                broke = True
                break
            except ContinueException:
                break
        if broke:
            break
    return result


def evaluate_async_with(
    with_node: ast.AsyncWith,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    '''`async with ctx as x: ...` driven synchronously via the background loop.

    Parameters:
    ---------
    with_node (ast.AsyncWith): the `async with ctx as x: ...` node.
    state (dict): the interpreter namespace.
    static_tools (dict): tools the interpreted code may call but not rebind.
    custom_tools (dict): tools defined for this session.
    authorized_imports (list): the import allowlist this session runs under.
    '''

    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        entered = drive_awaitable(context_expr.__aenter__())
        if item.optional_vars:
            set_value(item.optional_vars, entered, state, static_tools, custom_tools, authorized_imports)
        contexts.append(context_expr)

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        for context in reversed(contexts):
            drive_awaitable(context.__aexit__(type(e), e, e.__traceback__))
        raise
    else:
        for context in reversed(contexts):
            drive_awaitable(context.__aexit__(None, None, None))


@safer_eval
def evaluate_ast(
    expression: ast.AST,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
):
    '''Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree provided.

    Parameters:
    ---------
    expression (`ast.AST`): The code to evaluate, as an abstract syntax tree.
    state (`Dict[str, Any]`): A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation encounters assignments.
    static_tools (`Dict[str, Callable]`): Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
    custom_tools (`Dict[str, Callable]`): Functions that may be called during the evaluation. These static_tools can be overwritten.
    authorized_imports (`List[str]`): The list of modules that can be imported by the code. By default, only a few safe modules are allowed. If it contains "*", it will authorize any import. Use this at your own risk!

    Returns:
    ----------
    value (any): the value of the evaluated node.
    '''

    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    state["_operations_count"]["counter"] += 1
    common_params = (state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression, ast.Assign):
        # Assignment -> we evaluate the assignment which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, *common_params)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, *common_params)
    elif isinstance(expression, ast.AnnAssign):
        # Annotated assignment: `x: int = 5`
        return evaluate_annassign(expression, *common_params)
    elif isinstance(expression, ast.NamedExpr):
        # Walrus: `(y := f(x))`
        return evaluate_namedexpr(expression, *common_params)
    elif isinstance(expression, ast.Await):
        return evaluate_await(expression, *common_params)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, *common_params)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, (ast.ListComp, ast.GeneratorExp)):
        return evaluate_listcomp(expression, *common_params)
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, *common_params)
    elif isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, *common_params)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, *common_params)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, *common_params)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, *common_params)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, *common_params)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, *common_params)
    elif isinstance(expression, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Async functions are defined exactly like sync ones; their `await`
        # nodes resolve on the shared background loop when the body runs.
        return evaluate_function_def(expression, *common_params)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = (evaluate_ast(k, *common_params) for k in expression.keys)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, *common_params)
    elif isinstance(expression, ast.AsyncFor):
        return evaluate_async_for(expression, *common_params)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and format it
        value = evaluate_ast(expression.value, *common_params)
        # Early return if no format spec
        if not expression.format_spec:
            return value
        # Apply format specification
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, *common_params)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, *common_params)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, *common_params)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, *common_params)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, *common_params)
        if test_val:
            return evaluate_ast(expression.body, *common_params)
        else:
            return evaluate_ast(expression.orelse, *common_params)
    elif isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, *common_params)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, *common_params)
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        return evaluate_import(expression, state, authorized_imports)
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, *common_params)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, *common_params)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, *common_params)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, *common_params)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, *common_params)
    elif isinstance(expression, ast.AsyncWith):
        return evaluate_async_with(expression, *common_params)
    elif isinstance(expression, (ast.Global, ast.Nonlocal)):
        # The interpreter uses a shared state dict, so scope declarations are a
        # no-op rather than an error — this keeps generated code running.
        return None
    elif isinstance(expression, ast.Set):
        return set((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    elif isinstance(expression, ast.Pass):
        return None
    elif isinstance(expression, ast.Delete):
        return evaluate_delete(expression, *common_params)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


class FinalAnswerException(Exception):
    def __init__(self, value):
        self.value = value


def evaluate_python_code(
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
):
    '''Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Parameters:
    ---------
    code (`str`): The code to evaluate.
    static_tools (`Dict[str, Callable]`): The functions that may be called during the evaluation. These can also be agents in a multiagent setting. These tools cannot be overwritten in the code: any assignment to their name will raise an error.
    custom_tools (`Dict[str, Callable]`): The functions that may be called during the evaluation. These tools can be overwritten in the code: any assignment to their name will overwrite them.
    state (`Dict[str, Any]`): A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be updated by this function to contain all variables as they are evaluated. The print outputs will be stored in the state under the key "_print_outputs".
    authorized_imports (`List[str]`): The list of modules that can be imported by the code. By default, only a few safe modules are allowed. If it contains "*", it will authorize any import. Use this at your own risk!

    Returns:
    ----------
    result (tuple): the value of the final expression together with whatever the code printed.
    '''

    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools if custom_tools is not None else {}
    result = None
    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter": 0}

    if "final_answer" in static_tools:
        previous_final_answer = static_tools["final_answer"]

        def final_answer(answer):  # Using 'answer' as the argument like in the original function
            raise FinalAnswerException(previous_final_answer(answer))

        static_tools["final_answer"] = final_answer

    try:
        for node in expression.body:
            result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        is_final_answer = False
        return result, is_final_answer
    except FinalAnswerException as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        is_final_answer = True
        return e.value, is_final_answer
    except Exception as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        raise InterpreterError(
            f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to: {type(e).__name__}: {e}"
        )


class PythonExecutor:
    pass


class LocalPythonExecutor(PythonExecutor):
    def __init__(
        self,
        additional_authorized_imports: List[str],
        max_print_outputs_length: Optional[int] = None,
    ):
        self.custom_tools = {}
        self.state = {}
        self.max_print_outputs_length = max_print_outputs_length
        if max_print_outputs_length is None:
            self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        # TODO: assert self.authorized imports are all installed locally
        self.static_tools = BASE_PYTHON_TOOLS.copy()

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        output, is_final_answer = evaluate_python_code(
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_print_outputs_length=self.max_print_outputs_length,
        )
        logs = str(self.state["_print_outputs"])
        return output, logs, is_final_answer

    def send_variables(self, variables: dict):
        self.state.update(variables)

    # PISEK COMMENTED IT OUT
    # def send_tools(self, tools: Dict[str, Tool]):
    #     self.static_tools = {**tools, **BASE_PYTHON_TOOLS.copy()}


class ExecutionTimeout(InterpreterError):
    '''Raised when a single execution exceeds its wall-clock budget.'''

class ExecutionCancelled(InterpreterError):
    '''Raised when the user stopped the run while this execution was in flight.'''

DEFAULT_SESSION_KEY = "default"
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 900
# How long a call waits for a sibling execution in the same conversation before
# giving up. Only reachable when a previous call is still running.
SESSION_ACQUIRE_TIMEOUT_SECONDS = 60
_MAX_SESSIONS = 32
# One interpreter per (user, conversation) rather than one per process. The
# previous single global was shared by every user and every conversation:
# concurrent runs interleaved in one namespace, and one conversation's
# `reset_python_state` wiped everybody else's variables.
_executors: "OrderedDict[str, LocalPythonExecutor]" = OrderedDict()
_registry_lock = threading.RLock()
_session_locks: Dict[str, threading.RLock] = {}


def set_max_executor_sessions(size: int) -> None:
    '''How many interpreter sessions stay resident before LRU eviction.

    Parameters:
    ---------
    size (int): how many interpreter sessions stay resident before LRU eviction.
    '''

    global _MAX_SESSIONS
    _MAX_SESSIONS = max(1, int(size))


def _session_lock(session_key: str) -> threading.RLock:
    with _registry_lock:
        lock = _session_locks.get(session_key)
        if lock is None:
            lock = threading.RLock()
            _session_locks[session_key] = lock
        return lock


def _get_executor(session_key: str, authorized_imports: List[str]) -> "LocalPythonExecutor":
    with _registry_lock:
        executor = _executors.get(session_key)
        if executor is None:
            executor = LocalPythonExecutor(additional_authorized_imports=authorized_imports)
            _executors[session_key] = executor
            while len(_executors) > _MAX_SESSIONS:
                evicted, _ = _executors.popitem(last=False)
                _session_locks.pop(evicted, None)
        else:
            _executors.move_to_end(session_key)
            merged = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
            if set(executor.authorized_imports) != set(merged):
                executor.authorized_imports = merged
        return executor


def _drop_session(session_key: str) -> None:
    '''Forget one session's interpreter and lock, leaving other sessions alone.

    Parameters:
    ---------
    session_key (str): the `<user>::<conversation>` session to forget, leaving every other session alone.
    '''

    with _registry_lock:
        _executors.pop(session_key, None)
        _session_locks.pop(session_key, None)


def active_session_keys() -> List[str]:
    with _registry_lock:
        return list(_executors)


def _interrupt_thread(thread: threading.Thread) -> None:
    '''Best-effort attempt to unwind a worker that overran its budget.

    Only fires between bytecodes, so it recovers a runaway pure-Python loop but
    cannot interrupt a blocking socket read. Nothing may depend on it — the
    session is dropped either way so the conversation stays usable.

    Parameters:
    ---------
    thread (threading.Thread): the worker that overran its wall-clock budget.
    '''

    thread_id = getattr(thread, "ident", None)
    if thread_id is None:
        return
    try:
        import ctypes

        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit)
        )
    except Exception:  # pragma: no cover - platform dependent
        logger.debug("Could not signal runaway execution thread %s", thread_id)


def _call_with_timeout(func, timeout: Optional[float], cancel_event=None):
    '''Run `func` in a worker thread, giving up after `timeout` seconds.

    `cancel_event` lets the user's Stop abandon the wait promptly instead of
    sitting out the full budget; the worker itself cannot be killed, so the
    session is dropped and the thread is left to finish into nothing.

    Parameters:
    ---------
    func (callable): the work to run in a worker thread.
    timeout (float): seconds to allow before giving up.
    cancel_event (threading.Event): set when the user stops the run, so a long call ends early.

    Returns:
    ----------
    result (any): what `func` returned, or a raised timeout — LangChain runs sync tools in a thread pool, so cancelling a run does not kill the thread and the session is dropped instead.
    '''

    if (not timeout or timeout <= 0) and cancel_event is None:
        return func()

    box: Dict[str, Any] = {}

    def target():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name="repuragent-python-exec")
    worker.start()

    deadline = None if not timeout or timeout <= 0 else time.monotonic() + timeout
    poll = 0.1
    while worker.is_alive():
        if cancel_event is not None and cancel_event.is_set():
            _interrupt_thread(worker)
            raise ExecutionCancelled("Execution abandoned because the run was stopped.")
        if deadline is not None and time.monotonic() >= deadline:
            _interrupt_thread(worker)
            raise ExecutionTimeout(
                f"Execution exceeded the {timeout:.0f}s limit and was abandoned."
            )
        worker.join(poll)

    if "error" in box:
        raise box["error"]
    return box.get("value")


def reset_executor_state(session_key: Optional[str] = None):
    '''Reset persistent executor state.

    Clears variables and functions defined in previous executions. With a session
    key only that conversation's interpreter is cleared; without one, every
    session is cleared.

    Example:
    ----------
    >>> local_python_executor("x = 42", [], session_key="s1")
    42
    >>> reset_executor_state("s1")
    >>> local_python_executor("print(x)", [], session_key="s1")  # NameError

    Parameters:
    ---------
    session_key (str): the session to clear, or None for every session.
    '''

    with _registry_lock:
        if session_key is None:
            _executors.clear()
            _session_locks.clear()
            return
        _executors.pop(session_key, None)
        _session_locks.pop(session_key, None)


def local_python_executor(
    code: str,
    authorized_imports: List[str],
    variables: Optional[Dict[str, Any]] = None,
    session_key: str = DEFAULT_SESSION_KEY,
    timeout: Optional[float] = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    cancel_event=None,
):
    '''Execute Python in a sandboxed interpreter with restricted imports.

    Variables defined in previous executions of the same session are preserved,
    giving REPL-like behaviour within one conversation.

    Parameters:
    ---------
    code (str): The Python code to execute.
    authorized_imports (list): Modules importable in addition to `BASE_BUILTIN_MODULES`. Include `"*"` for unrestricted imports.
    variables (dict): Values injected into the session before the code runs.
    session_key (str): Identifies the interpreter session. Callers must scope this to a (user, conversation) pair so state never leaks between them.
    timeout (float): Wall-clock budget in seconds. On expiry the session is dropped and `ExecutionTimeout` is raised. Pass None to disable.
    cancel_event: Optional `threading.Event`; when set, the call raises `ExecutionCancelled` instead of waiting out the budget.

    Returns:
    ----------
    result (any): The value of the last statement, or the captured print output when the
    last statement produced nothing.

    Raises:
    ----------
    InterpreterError: syntax errors, unauthorized imports, runtime errors. ExecutionTimeout / ExecutionCancelled: as described above.
    '''

    executor = _get_executor(session_key, authorized_imports)

    # Serialize calls within a session: LangChain runs sync tools in a thread
    # pool, so two calls of the same conversation could otherwise mutate one
    # interpreter namespace concurrently.
    lock = _session_lock(session_key)
    if not lock.acquire(timeout=SESSION_ACQUIRE_TIMEOUT_SECONDS):
        raise ExecutionTimeout(
            "A previous execution in this conversation is still running. Wait for "
            "it to finish, or call reset_python_state to start a clean session."
        )
    try:
        if variables:
            executor.send_variables(variables)
        output, logs, is_final_answer = _call_with_timeout(
            lambda: executor(code_action=code), timeout, cancel_event
        )
    except (ExecutionTimeout, ExecutionCancelled):
        # The worker may still be stuck in a blocking call holding this
        # interpreter. Drop the session so the next call gets a clean one instead
        # of queueing behind a thread that may never return.
        _drop_session(session_key)
        raise
    finally:
        lock.release()

    # A statement that only printed still has something to report.
    if output is None and logs.strip():
        return logs.strip()
    return output


__all__ = [
    "DEFAULT_EXECUTION_TIMEOUT_SECONDS",
    "DEFAULT_SESSION_KEY",
    "ExecutionCancelled",
    "ExecutionTimeout",
    "LocalPythonExecutor",
    "active_session_keys",
    "evaluate_python_code",
    "local_python_executor",
    "reset_executor_state",
    "set_max_executor_sessions",
    "truncate_content",
]
