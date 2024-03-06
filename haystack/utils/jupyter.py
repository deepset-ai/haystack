# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


def is_in_jupyter() -> bool:
    """
    Returns `True` if in Jupyter or Google Colab, `False` otherwise.
    """
    # Inspired by:
    # https://github.com/explosion/spaCy/blob/e1249d3722765aaca56f538e830add7014d20e2a/spacy/util.py#L1079
    try:
        # We don't need to import `get_ipython` as it's always present in Jupyter notebooks
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":  # type: ignore[name-defined]
            return True  # Jupyter notebook or qtconsole
        if get_ipython().__class__.__module__ == "google.colab._shell":  # type: ignore[name-defined]
            return True  # Colab notebook
    except NameError:
        pass  # Probably standard Python interpreter
    return False
