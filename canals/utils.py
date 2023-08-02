from typing import get_args


def _type_name(type_):
    """
    Util methods to get a nice readable representation of a type.
    Handles Optional and Literal in a special way to make it more readable.
    """
    # Literal args are strings, so we wrap them in quotes to make it clear
    if isinstance(type_, str):
        return f"'{type_}'"

    name = getattr(type_, "__name__", str(type_))

    if name.startswith("typing."):
        name = name[7:]
    if "[" in name:
        name = name.split("[")[0]
    args = get_args(type_)
    if name == "Union" and type(None) in args and len(args) == 2:
        # Optional is technically a Union of type and None
        # but we want to display it as Optional
        name = "Optional"

    if args:
        args = ", ".join([_type_name(a) for a in args if a is not type(None)])
        return f"{name}[{args}]"

    return f"{name}"
