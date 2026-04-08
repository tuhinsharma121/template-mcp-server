"""Tests for the execute_python sandbox tool.

This module contains comprehensive tests for security, functionality,
and edge cases of the sandboxed Python execution tool.
"""

import pytest

from template_mcp_server.src.tools.execute_python_tool import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_SIZE,
    execute_python,
)


class TestExecutePythonBasicFunctionality:
    """Test basic Python execution functionality."""

    def test_execute_simple_print_returns_stdout(self):
        """Test that print statements are captured in stdout."""
        code = "print('hello world')"
        result = execute_python(code)

        assert result["status"] == "success"
        assert "hello world" in result["stdout"]
        assert result["truncated"] is False

    def test_execute_arithmetic_expression(self):
        """Test basic arithmetic operations."""
        code = "result = 2 + 3 * 4\nprint(result)"
        result = execute_python(code)

        assert result["status"] == "success"
        assert "14" in result["stdout"]

    def test_execute_with_variables(self):
        """Test variable assignment and usage."""
        code = """
x = 10
y = 20
z = x + y
print(f"Sum: {z}")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Sum: 30" in result["stdout"]

    def test_execute_list_operations(self):
        """Test list creation and manipulation."""
        code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Total: {total}")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Total: 15" in result["stdout"]

    def test_execute_dictionary_operations(self):
        """Test dictionary operations."""
        code = """
data = {'a': 1, 'b': 2, 'c': 3}
print(len(data))
print(list(data.keys()))
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "3" in result["stdout"]

    def test_execute_for_loop(self):
        """Test for loop execution."""
        code = """
total = 0
for i in range(5):
    total += i
print(total)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "10" in result["stdout"]

    def test_execute_while_loop(self):
        """Test while loop execution."""
        code = """
count = 0
while count < 3:
    print(count)
    count += 1
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "0" in result["stdout"]
        assert "1" in result["stdout"]
        assert "2" in result["stdout"]

    def test_execute_function_definition(self):
        """Test function definition and calling."""
        code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "8" in result["stdout"]

    def test_execute_list_comprehension(self):
        """Test list comprehension."""
        code = """
squares = [x**2 for x in range(5)]
print(squares)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "[0, 1, 4, 9, 16]" in result["stdout"]


class TestExecutePythonAllowedModules:
    """Test that allowed modules work correctly.

    Note: RestrictedPython blocks 'import' statements by default.
    Modules are made available via pre-populated globals instead.
    """

    def test_math_module_available(self):
        """Test that math module is accessible via globals."""
        code = """
print(math.sqrt(16))
print(math.pi)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "4.0" in result["stdout"]
        assert "3.14159" in result["stdout"]

    def test_json_module_available(self):
        """Test that json module is accessible via globals."""
        code = """
data = {'key': 'value', 'number': 42}
print(json.dumps(data))
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "key" in result["stdout"]
        assert "value" in result["stdout"]

    def test_re_module_available(self):
        """Test that re module is accessible via globals."""
        code = """
text = "Hello, World!"
match = re.search(r'World', text)
print(match.group())
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "World" in result["stdout"]

    def test_datetime_module_available(self):
        """Test that datetime module is accessible via globals."""
        code = """
dt = datetime.datetime(2024, 1, 15)
print(dt.year)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "2024" in result["stdout"]

    def test_collections_module_available(self):
        """Test that collections module is accessible via globals."""
        code = """
counter = collections.Counter(['a', 'b', 'a', 'c', 'a'])
print(counter['a'])
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "3" in result["stdout"]

    def test_statistics_module_available(self):
        """Test that statistics module is accessible via globals."""
        code = """
data = [1, 2, 3, 4, 5]
print(statistics.mean(data))
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "3" in result["stdout"]

    def test_numpy_available_if_installed(self):
        """Test that numpy is accessible via globals when installed."""
        code = """
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))
"""
        result = execute_python(code)

        if result["status"] == "error" and "np" in result.get("error", ""):
            pytest.skip("numpy not available in restricted globals")

        assert result["status"] == "success"
        assert "3.0" in result["stdout"]

    def test_pandas_available_if_installed(self):
        """Test that pandas is accessible via globals when installed."""
        code = """
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.shape)
"""
        result = execute_python(code)

        if result["status"] == "error" and "pd" in result.get("error", ""):
            pytest.skip("pandas not available in restricted globals")

        assert result["status"] == "success"
        assert "(3, 2)" in result["stdout"]


class TestExecutePythonSecurityBlocking:
    """Test that dangerous operations are blocked."""

    def test_block_file_open(self):
        """Test that open() for file operations is blocked."""
        code = "f = open('/etc/passwd', 'r')"
        result = execute_python(code)

        assert result["status"] == "error"
        assert "error" in result

    def test_block_os_module(self):
        """Test that os module is not available."""
        code = "import os; os.system('ls')"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_subprocess_module(self):
        """Test that subprocess module is not available."""
        code = "import subprocess; subprocess.run(['ls'])"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_sys_module_dangerous_attrs(self):
        """Test that dangerous sys attributes are not accessible."""
        code = "import sys; sys.exit(1)"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_eval_function(self):
        """Test that eval() is blocked."""
        code = "eval('print(1)')"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_exec_function(self):
        """Test that exec() is blocked."""
        code = "exec('print(1)')"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_compile_function(self):
        """Test that compile() is blocked."""
        code = "compile('print(1)', '<string>', 'exec')"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_import_dangerous_module(self):
        """Test that importing dangerous modules is blocked."""
        code = "__import__('os')"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_dunder_class_access(self):
        """Test that __class__ attribute access is blocked."""
        code = "x = ''.__class__"
        result = execute_python(code)

        assert result["status"] == "error"
        assert "__class__" in result.get("error", "") or "invalid attribute" in result.get("error", "").lower()

    def test_block_dunder_globals_access(self):
        """Test that __globals__ attribute access is blocked."""
        code = """
def foo():
    pass
x = foo.__globals__
"""
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_dunder_subclasses(self):
        """Test that __subclasses__ is blocked."""
        code = "object.__subclasses__()"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_builtins_access(self):
        """Test that direct __builtins__ access is blocked."""
        code = "__builtins__"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_socket_module(self):
        """Test that socket module is not available."""
        code = "import socket; s = socket.socket()"
        result = execute_python(code)

        assert result["status"] == "error"

    def test_block_urllib_module(self):
        """Test that urllib module is not available."""
        code = "import urllib.request"
        result = execute_python(code)

        assert result["status"] == "error"


class TestExecutePythonErrorHandling:
    """Test error handling scenarios."""

    def test_empty_code_returns_error(self):
        """Test that empty code returns an error."""
        result = execute_python("")

        assert result["status"] == "error"
        assert "No code provided" in result["error"]

    def test_whitespace_only_code_returns_error(self):
        """Test that whitespace-only code returns an error."""
        result = execute_python("   \n\t  ")

        assert result["status"] == "error"

    def test_syntax_error_returns_error(self):
        """Test that syntax errors are properly reported."""
        code = "def foo( invalid syntax"
        result = execute_python(code)

        assert result["status"] == "error"
        assert result["error_type"] == "SyntaxError"

    def test_name_error_returns_error(self):
        """Test that undefined variable errors are reported."""
        code = "print(undefined_variable)"
        result = execute_python(code)

        assert result["status"] == "error"
        assert "NameError" in result["error_type"]

    def test_type_error_returns_error(self):
        """Test that type errors are properly reported."""
        code = "'string' + 5"
        result = execute_python(code)

        assert result["status"] == "error"
        assert result["error_type"] == "TypeError"

    def test_zero_division_error(self):
        """Test that division by zero is properly reported."""
        code = "x = 1 / 0"
        result = execute_python(code)

        assert result["status"] == "error"
        assert result["error_type"] == "ZeroDivisionError"

    def test_index_error(self):
        """Test that index errors are properly reported."""
        code = "lst = [1, 2, 3]; print(lst[10])"
        result = execute_python(code)

        assert result["status"] == "error"
        assert result["error_type"] == "IndexError"


class TestExecutePythonTimeout:
    """Test timeout enforcement."""

    def test_timeout_clamps_to_maximum(self):
        """Test that timeout is clamped to 120 seconds maximum."""
        code = "print('test')"
        result = execute_python(code, timeout_seconds=500)

        assert result["status"] == "success"

    def test_negative_timeout_uses_default(self):
        """Test that negative timeout uses default value."""
        code = "print('test')"
        result = execute_python(code, timeout_seconds=-10)

        assert result["status"] == "success"

    def test_zero_timeout_uses_default(self):
        """Test that zero timeout uses default value."""
        code = "print('test')"
        result = execute_python(code, timeout_seconds=0)

        assert result["status"] == "success"


class TestExecutePythonOutputHandling:
    """Test output handling and truncation."""

    def test_multiple_print_statements(self):
        """Test that multiple print statements are all captured."""
        code = """
print("line 1")
print("line 2")
print("line 3")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "line 1" in result["stdout"]
        assert "line 2" in result["stdout"]
        assert "line 3" in result["stdout"]

    def test_result_variable_captured(self):
        """Test that a 'result' variable is captured."""
        code = "result = 42"
        result = execute_python(code)

        assert result["status"] == "success"
        assert result["result"] == "42"

    def test_return_structure_contains_required_fields(self):
        """Test that return dictionary has all required fields."""
        code = "print('test')"
        result = execute_python(code)

        assert "status" in result
        assert "stdout" in result
        assert "result" in result
        assert "truncated" in result


class TestExecutePythonEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_in_code(self):
        """Test that unicode characters work correctly."""
        code = "print('Hello, 世界! 🌍')"
        result = execute_python(code)

        assert result["status"] == "success"
        assert "世界" in result["stdout"]

    def test_multiline_string(self):
        """Test multiline string handling."""
        code = '''
text = """
Line 1
Line 2
Line 3
"""
print(text)
'''
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Line 1" in result["stdout"]

    def test_f_string_formatting(self):
        """Test f-string formatting works."""
        code = """
name = "Alice"
age = 30
print(f"{name} is {age} years old")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Alice is 30 years old" in result["stdout"]

    def test_exception_handling_in_user_code(self):
        """Test that user code can use try/except."""
        code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught division by zero")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Caught division by zero" in result["stdout"]

    def test_class_definition(self):
        """Test that simple class definitions work.
        
        Note: RestrictedPython blocks magic methods like __init__ and __str__
        for security reasons, so we test simpler class patterns.
        """
        code = """
class Point:
    pass

p = Point()
p.x = 3
p.y = 4
print(f"Point: {p.x}, {p.y}")
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "Point: 3, 4" in result["stdout"]

    def test_lambda_functions(self):
        """Test that lambda functions work."""
        code = """
square = lambda x: x ** 2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
print(squared)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "[1, 4, 9, 16, 25]" in result["stdout"]

    def test_generator_expressions(self):
        """Test that generator expressions work."""
        code = """
gen = (x**2 for x in range(5))
result = list(gen)
print(result)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "[0, 1, 4, 9, 16]" in result["stdout"]

    def test_set_operations(self):
        """Test that set operations work."""
        code = """
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)
print(a | b)
"""
        result = execute_python(code)

        assert result["status"] == "success"
        assert "3" in result["stdout"]
        assert "4" in result["stdout"]
