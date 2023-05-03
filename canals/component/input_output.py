from dataclasses import dataclass


def component_input(cls, *args, **kwargs):
    """component input dataclass"""
    return dataclass(cls, *args, **kwargs)


def component_output(cls, *args, **kwargs):
    """component output dataclass"""
    return dataclass(cls, *args, **kwargs)
