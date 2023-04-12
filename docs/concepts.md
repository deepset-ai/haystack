# Basic Concepts

Canals is a **component orchestration engine**. It can be used to connect a group of smaller objects, called Components,
that  perform well-defined tasks into a network, called Pipeline, that achieves a larger goal.

Components are Python objects that can execute a task, like reading a file, performing calculations, or making API calls.
Canals connects these objects together: it builds a graph of components and takes care of managing their execution order,
making sure that each object receives the input it expects from the other components of the pipeline.

# Components
