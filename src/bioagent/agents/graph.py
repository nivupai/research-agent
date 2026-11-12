"""Assemble the LangGraph pipeline.

START -> planner -> dispatch --(Send per ready task)--> literature | chemistry | analysis
                      ^                                        |
                      +----------------------------------------+
         dispatch --(no ready tasks)--> synthesizer -> verifier -> END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from bioagent.agents.nodes import (
    AgentDeps,
    dispatch,
    make_planner,
    make_specialist,
    make_synthesizer,
    route_ready,
    verifier,
)
from bioagent.agents.state import TASK_KINDS, AgentState


def build_graph(deps: AgentDeps) -> Any:
    """Return an uncompiled ``StateGraph`` (typed as Any: LangGraph's node overloads do not
    model closures over different input schemas such as the Send payload)."""
    graph: Any = StateGraph(AgentState)
    graph.add_node("planner", make_planner(deps))
    graph.add_node("dispatch", dispatch)
    for kind in TASK_KINDS:
        graph.add_node(kind, make_specialist(kind, deps))
        graph.add_edge(kind, "dispatch")
    graph.add_node("synthesizer", make_synthesizer(deps))
    graph.add_node("verifier", verifier)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "dispatch")
    graph.add_conditional_edges("dispatch", route_ready, [*TASK_KINDS, "synthesizer"])
    graph.add_edge("synthesizer", "verifier")
    graph.add_edge("verifier", END)
    return graph
