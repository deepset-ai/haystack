# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


def import_class_by_name(fully_qualified_name: str) -> type:
    """
    Utility function to import (load) a class object based on its fully qualified class name.

    This function dynamically imports a class based on its string name.
    It splits the name into module path and class name, imports the module,
    and returns the class object.

    :param fully_qualified_name: the fully qualified class name as a string
    :returns: the class object.
    :raises ImportError: If the class cannot be imported or found.
    """
    import logging

    logger = logging.getLogger(__name__)
    from haystack.utils.type_serialization import thread_safe_import

    try:
        module_path, class_name = fully_qualified_name.rsplit(".", 1)
        logger.debug(
            "Attempting to import class '{cls_name}' from module '{md_path}'", cls_name=class_name, md_path=module_path
        )
        module = thread_safe_import(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        logger.error("Failed to import class '{full_name}'", full_name=fully_qualified_name)
        raise ImportError(f"Could not import class '{fully_qualified_name}'") from error
