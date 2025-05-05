from dataclasses import fields, is_dataclass
from typing import Any, Dict, Union, get_args, get_origin

from haystack.tools.errors import SchemaGenerationError


def _create_list_schema(item_type: Any, description: str) -> Dict[str, Any]:
    """
    Creates a schema for a list type.

    :param item_type: The type of items in the list.
    :param description: The description of the list.
    :returns: A dictionary representing the list schema.
    """
    items_schema = _create_property_schema(item_type, "")
    items_schema.pop("description", None)
    return {"type": "array", "description": description, "items": items_schema}


def _create_dataclass_schema(python_type: Any, description: str) -> Dict[str, Any]:
    """
    Creates a schema for a dataclass.

    :param python_type: The dataclass type.
    :param description: The description of the dataclass.
    :returns: A dictionary representing the dataclass schema.
    """
    schema: Dict[str, Any] = {"type": "object", "description": description, "properties": {}}
    cls = python_type if isinstance(python_type, type) else python_type.__class__
    for field in fields(cls):
        field_description = f"Field '{field.name}' of '{cls.__name__}'."
        field_schema = _create_property_schema(field.type, field_description)
        schema["properties"][field.name] = field_schema
    return schema


def _create_basic_type_schema(python_type: Any, description: str) -> Dict[str, Union[str, bool]]:
    """
    Creates a schema for a basic Python type.

    :param python_type: The Python type.
    :param description: The description of the type.
    :returns: A dictionary representing the basic type schema.
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    resolved_type_mapping = type_mapping.get(python_type)
    if resolved_type_mapping is None:
        # If the type is not found in the mapping, default to "string"
        resolved_type_mapping = "string"

    schema: Dict[str, Union[str, bool]] = {"type": resolved_type_mapping, "description": description}
    if schema["type"] == "object":
        # If the type is an object, set additionalProperties to True
        schema["additionalProperties"] = True
    return schema


def _create_union_schema(types: tuple, description: str) -> Dict[str, Any]:
    """
    Creates a schema for a Union type.

    :param types: The types in the Union.
    :param description: The description of the Union.
    :returns: A dictionary representing the Union schema.
    """
    schemas = []
    for arg_type in types:
        arg_schema = _create_property_schema(arg_type, "")
        arg_schema.pop("description", None)
        schemas.append(arg_schema)

    if len(schemas) == 1:
        schema = schemas[0]
        schema["description"] = description
    else:
        schema = {"oneOf": schemas, "description": description}
    return schema


def _create_property_schema(python_type: Any, description: str, default: Any = None) -> Dict[str, Any]:
    """
    Creates a property schema for a given Python type, recursively if necessary.

    :param python_type: The Python type to create a property schema for.
    :param description: The description of the property.
    :param default: The default value of the property.
    :returns: A dictionary representing the property schema.
    :raises SchemaGenerationError: If schema generation fails, e.g., for unsupported types like Pydantic v2 models
    """
    origin = get_origin(python_type)
    args = get_args(python_type)
    schema: Dict[str, Any]

    # Handle dict type. Not possible to use args info here because we don't know the name of the key.
    if origin is dict:
        schema = {"type": "object", "description": description, "additionalProperties": True}
    # Handle list
    elif origin is list:
        schema = _create_list_schema(get_args(python_type)[0] if get_args(python_type) else Any, description)
    # Handle Union (including Optional)
    elif origin is Union:
        schema = _create_union_schema(args, description)
    # Handle dataclass
    elif is_dataclass(python_type):
        schema = _create_dataclass_schema(python_type, description)
    # Handle Pydantic v2 models (unsupported)
    elif hasattr(python_type, "model_validate"):
        raise SchemaGenerationError(
            f"Pydantic models (e.g. {python_type.__name__}) are not supported as input types for "
            f"component's run method."
        )
    else:
        schema = _create_basic_type_schema(python_type, description)

    if default is not None:
        schema["default"] = default

    return schema
