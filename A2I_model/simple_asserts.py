"""
Lightweight input-validation helpers. Useful for catching errors early.
"""

import os
from typing import Union, List

def assert_type(to_check, expected_type, variable_name:str, allow_none:bool=False) -> None:
    """
    Check variable against the expected type

    # EXAMPLES
    # (PASS)
    >> assert_type("abc", str, "some_var")
    >> assert_type(None, str, "some_var", allow_none=True)

    # (FAIL)
    >> assert_type(123, str, "some_var")                     # raises TypeError
    >> assert_type(None, int, "some_var", allow_none=False)  # raises TypeError

    :param to_check: Object for type check
    :param expected_type: Expected type of `to_check`
    :param variable_name: The name of `to_check` that will be displayed in error messages
    :param allow_none: Weather or not None is acceptable
    """

    if not isinstance(allow_none, bool):
        error_message = f"Expected `allow_None` to by of type bool, but received `{type(allow_none)=}`"
        raise TypeError(error_message)
    if not isinstance(variable_name, str):
        error_message = f"Expected `variable_name` to by of type str, but received `{type(variable_name)=}`"
        raise TypeError(error_message)
    if (to_check is None) and (expected_type is None): # TODO: Test this
        error_message = f"`None` is not a valid type. If you're trying to check if `type(to_check) == None` try setting `expected_type=type(None)` instead."
        raise TypeError(error_message)

    try:
        is_ok = isinstance(to_check, expected_type)
    except:
        error_message = f"Failed to execute `isinstance({variable_name}, {expected_type})`, " \
                        f"most likely because `{expected_type}` is not valid for type checking"
        raise TypeError(error_message)
    if allow_none:
        is_ok = (to_check is None) or is_ok

    if not is_ok:
        error_message = f"Expected type `type({variable_name})={type(to_check)}` to be of type `{str(expected_type)}`"
        raise TypeError(error_message)

def assert_types(to_check: list, expected_types: list, variable_names: List[str], allow_nones: Union[List[int], bool] = None) -> None:
    """
    Check list of values against expected types


    # EXAMPLES
    # (PASS)
    >> assert_types([1, "x"], [int, str], ["a", "b"])
    >> assert_types([None, 3], [str, int], ["x", "y"], [1, 0])

    # (FAIL)
    >> assert_types([1, "x"], [int, int], ["a", "b"])              # raises TypeError (wrong type)
    >> assert_types([None, 3], [str, int], ["x", "y"], [0, 0])     # raises TypeError (None not allowed)
    >> assert_types([1, 2], [int], ["a"])                          # raises ValueError (length mismatch)


    :param to_check: List of values for type check
    :param expected_types: Expected types of `to_check`
    :param variable_names: The name of `to_check` that will be displayed in error messages
    :param allow_nones: list of booleans or 0/1
    """

    # Checks
    assert_type(to_check, list, "to_check")
    assert_type(expected_types, list, "expected_types")
    assert_type(variable_names, list, "variable_names")
    assert_type(allow_nones, list, "allow_nones", allow_none=True)
    if len(to_check) != len(expected_types):
        error_message = "length mismatch between `{to_check_values}` and `{expected_types}`"
        raise ValueError(error_message)

    # If `allow_nones` is None all values are set to False.
    if allow_nones is None:
        allow_nones = [False]*len(to_check)
    else:
        if not (len(variable_names) == len(allow_nones) == len(to_check)):
            raise ValueError(f"Expected equal lengths, but found: `{len(to_check)=}`, `{len(allow_nones)=}` and `{len(variable_names)=}`")
        for i, element in enumerate(allow_nones):
            if element in [0, 1]:
                allow_nones[i] = bool(element == 1) # the `== 1` is just to allow for zeros as False and ones as True
            else:
                raise ValueError(f"`{allow_nones=}` may only contain [False, True, 0, 1]")

    # check if all elements are of the correct type
    for (v, t, n, a) in zip(to_check, expected_types, variable_names, allow_nones):
        assert_type(to_check=v, expected_type=t, variable_name=n, allow_none=a)

def assert_folder_exists(path: str, variable_name: str) -> None:
    """
    Assert that the given path exists and is a folder

    # EXAMPLES
    >> assert_folder_exists("some/existing/folder", "folder_path") # Pass
    >> assert_folder_exists("non_existing/folder", "input_folder") # raises ValueError
    >> assert_folder_exists("some/existing/file.txt", "file_path") # raises ValueError (not a folder)

    :param path: Path to check
    :param variable_name: Name used in error message
    """
    assert_types([path, variable_name], [str, str], ["path", "variable_name"])
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: `{variable_name}`=`{path}`")
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a folder: `{variable_name}`=`{path}`")


def assert_path_exists(path:str, variable_name:str) -> None:
    """
    Assert that the given path exists

    # EXAMPLES
    >> assert_path_exists("some/existing/file.txt", "file_path") # Pass
    >> assert_path_exists("non_existing/path", "input_path")  # raises ValueError

    :param path: Path to check
    :param variable_name: Name used in error message
    """
    assert_types([path, variable_name], [str, str], ["variable_name", "path"])
    if not os.path.exists(path):
        raise ValueError(f"Received bad path: `{variable_name}`=`{path}`")

def assert_path_dont_exists(path:str, variable_name:str) -> None:
    """
    Assert that the given path does NOT exist

    # EXAMPLES
    >> assert_path_dont_exists("non_existing/output/file.txt", "output_path") # Pass
    >> assert_path_dont_exists("some/existing/file.txt", "file_path")  # raises ValueError

    :param path: Path to check
    :param variable_name: Name used in error message
    """
    assert_types([path, variable_name], [str, str], ["variable_name", "path"])
    if os.path.exists(path):
        raise ValueError(f"Path `{variable_name}`=`{path}` already exists")

def assert_positive_int(to_check:int, zero_allowed:bool=False, max_value_allowed:int=None, variable_name:str=None) -> None:
    """
    Check that an integer is positive (or non-negative) and optionally below a max value

    # EXAMPLES
    # (PASS)
    >> assert_positive_int(5)
    >> assert_positive_int(0, zero_allowed=True)
    >> assert_positive_int(10, max_value_allowed=10)

    # (FAIL)
    >> assert_positive_int(0)                         # raises ValueError (zero not allowed)
    >> assert_positive_int(-3)                        # raises ValueError (negative)
    >> assert_positive_int(15, max_value_allowed=10)  # raises ValueError (exceeds max)

    :param to_check: Integer to check
    :param zero_allowed: If True, zero is accepted as valid input
    :param max_value_allowed: Optional upper bound; if set, `to_check` must be <= this value
    :param variable_name: Optional name used in the error message
    """

    assert_types([to_check, zero_allowed, max_value_allowed, variable_name],
                 [int, bool, int, str],
                 ["to_check", "zero_allowed", "max_value_allowed", "variable_name"],
                 [0, 0, 1, 1])

    assert (max_value_allowed is None) or (max_value_allowed >= 0), f"Expected max_value >= 0, but found `{max_value_allowed=}`"
    if max_value_allowed is None:
        max_value_allowed = float("inf")

    if (not zero_allowed) and ( (to_check <= 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 < {to_check} <= {max_value_allowed}`")
    elif zero_allowed and ( (to_check < 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 <= {to_check} <= {max_value_allowed}`")

def assert_positive_float(to_check: float, zero_allowed: bool = False, max_value_allowed: float = None, variable_name: str = None) -> None:
    """
    Check that a float is positive (or non-negative) and optionally below a max value

    # EXAMPLES
    # (PASS)
    >> assert_positive_float(5.0)
    >> assert_positive_float(0.0, zero_allowed=True)
    >> assert_positive_float(10.0, max_value_allowed=10.0)

    # (FAIL)
    >> assert_positive_float(0.0)                         # raises ValueError (zero not allowed)
    >> assert_positive_float(-3.2)                        # raises ValueError (negative)
    >> assert_positive_float(15.1, max_value_allowed=10)  # raises ValueError (exceeds max)

    :param to_check: Float to check
    :param zero_allowed: If True, zero is accepted as valid input
    :param max_value_allowed: Optional upper bound; if set, `to_check` must be <= this value
    :param variable_name: Optional name used in the error message
    """

    assert_types([to_check, zero_allowed, max_value_allowed, variable_name],
                 [float, bool, float, str],
                 ["to_check", "zero_allowed", "max_value_allowed", "variable_name"],
                 [0, 0, 1, 1])

    assert (max_value_allowed is None) or (max_value_allowed >= 0.0), f"Expected max_value >= 0, but found `{max_value_allowed=}`"
    if max_value_allowed is None:
        max_value_allowed = float("inf")

    if (not zero_allowed) and ((to_check <= 0.0) or (to_check > max_value_allowed)):
        variable_name_string = f"{variable_name}=" if variable_name is not None else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 < {to_check} <= {max_value_allowed}`")
    elif zero_allowed and ((to_check < 0.0) or (to_check > max_value_allowed)):
        variable_name_string = f"{variable_name}=" if variable_name is not None else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 <= {to_check} <= {max_value_allowed}`")