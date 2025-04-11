import os
from typing import Union, List

def assert_positive_int(to_check:int, zero_allowed:bool=False, max_value_allowed:int=None, variable_name:str=None) -> None:
    assert_types([to_check, zero_allowed, max_value_allowed, variable_name],
                 [int, bool, int, str],
                 ["to_check", "zero_allowed", "max_value_allowed", "variable_name"],
                 [0, 0, 1, 1])

    assert (max_value_allowed is None) or (max_value_allowed >= 0), f"Expected max_value >= 0, but found `{max_value_allowed=}`"
    if max_value_allowed is None:
        max_value_allowed = float('inf')

    if (not zero_allowed) and ( (to_check <= 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 < {to_check} <= {max_value_allowed}`")
    elif zero_allowed and ( (to_check < 0) or (to_check > max_value_allowed) ):
        variable_name_string = f"{variable_name}=" if (variable_name is not None) else ""
        raise ValueError(f"`{variable_name_string}{to_check}` does not fulfill `0 <= {to_check} <= {max_value_allowed}`")

def assert_type(to_check, expected_type, variable_name:str, allow_none:bool=False) -> None:
    """
    Check variable against the expected type

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

def assert_path(path:str, variable_name:str) -> None:
    assert_types([path, variable_name], [str, str], ["variable_name", "path"])
    if not os.path.exists(path):
        raise ValueError(f"Received bad path: `{variable_name}`=`{path}`")

def assert_path_dont_exists(path:str, variable_name:str) -> None:
    assert_types([path, variable_name], [str, str], ["variable_name", "path"])
    if os.path.exists(path):
        raise ValueError(f"Path `{variable_name}`=`{path}` already exists")