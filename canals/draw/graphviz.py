# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import networkx
from networkx.drawing.nx_agraph import to_agraph as nx_to_agraph
from pygraphviz import AGraph


logger = logging.getLogger(__name__)


def to_agraph(graph: networkx.MultiDiGraph) -> AGraph:
    """
    Renders a pipeline graph using PyGraphViz. You need to install it and all its system dependencies for it to work.
    """
    for inp, outp, key, data in graph.out_edges("input", keys=True, data=True):
        data["style"] = "dashed"
        graph.add_edge(inp, outp, key=key, **data)

    for inp, outp, key, data in graph.in_edges("output", keys=True, data=True):
        data["style"] = "dashed"
        graph.add_edge(inp, outp, key=key, **data)

    graph.nodes["input"]["shape"] = "plain"
    graph.nodes["output"]["shape"] = "plain"
    agraph = nx_to_agraph(graph)
    agraph.layout("dot")
    return agraph
