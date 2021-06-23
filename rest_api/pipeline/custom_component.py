"""
Pipelines allow putting together Components to build a graph.

In addition to the standard Haystack Components, custom user-defined Components
can be used in a Pipeline YAML configuration.

The classes for the Custom Components must be defined in this file.
"""


from haystack import BaseComponent


class SampleComponent(BaseComponent):

    def run(self, **kwargs):
        raise NotImplementedError
