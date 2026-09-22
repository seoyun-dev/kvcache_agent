"""
3-4절 메인 에이전트 그래프를 코드로 그대로 옮긴 것.

    START -> A -> B -> {C, D, E} -> F -> G <-> H -> END

노드 함수 대부분은 notebooks/01~04에서 담당자가 각자 만들어 저장한
src/nodes_b.py, nodes_cd.py, nodes_e.py, nodes_fgh.py에서 가져온다.
A는 LLM 호출이 없는 정적 노드라 별도 파일 없이 여기 바로 둔다.

**이 파일을 실행하기 전에 notebooks/01~04번을 한 번씩 끝까지 돌려서
nodes_b.py 등 4개 파일을 먼저 만들어둬야 한다.** (이 리포에는 이미 한 번
생성해서 넣어뒀지만, 노트북에서 프롬프트나 로직을 고쳤으면 해당 노트북의
"파일로 저장" 셀을 다시 실행해야 여기 반영된다.)

MVP라 B'/E'(LLM-as-Judge 근거검증)는 없다. 시간 남으면 이 파일에
add_node("B_verify", ...) 식으로 끼워넣으면 된다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import config
from .state import GraphState

try:
    from .nodes_b import make_node_b
    from .nodes_cd import make_node_c, make_node_d
    from .nodes_e import make_node_e
    from .nodes_fgh import make_node_f, make_node_g, node_h_validate, route_after_h
except ImportError as e:
    raise ImportError(
        "notebooks/01~04번을 먼저 한 번씩 끝까지 실행해야 한다 - 각 노트북의 "
        "마지막 '파일로 저장' 셀이 src/nodes_b.py 등을 만든다. "
        f"(원래 에러: {e})"
    ) from e


def node_a_select_technologies(state: GraphState) -> dict:
    """A. 기술 선정 (Human 기반, 정적 노드 - LLM 호출 없음)."""
    return {
        "selected_technologies": config.SELECTED_TECHNOLOGIES,
        "target_domain": config.TARGET_DOMAIN,
    }


def build_graph(llm, tech_retriever, domain_retriever, web_search_tool):
    """실제 의존성(llm, retriever, 웹서치 도구)을 주입해 컴파일된 그래프를 반환.

    테스트할 때는 fake llm/retriever/tool을 넣어서 그래프 배선(엣지, 조건부
    분기, state 병합)만 검증할 수 있다 - tests/test_graph_wiring.py 참고.
    """
    g = StateGraph(GraphState)

    g.add_node("A", node_a_select_technologies)
    g.add_node("B", make_node_b(llm, tech_retriever, web_search_tool))
    g.add_node("C", make_node_c(llm, web_search_tool))
    g.add_node("D", make_node_d(llm, web_search_tool))
    g.add_node("E", make_node_e(llm, domain_retriever))
    g.add_node("F", make_node_f(llm))
    g.add_node("G", make_node_g(llm))
    g.add_node("H", node_h_validate)

    g.add_edge(START, "A")
    g.add_edge("A", "B")

    # Fan-out: B 완료 후 C, D, E 병렬 시작
    g.add_edge("B", "C")
    g.add_edge("B", "D")
    g.add_edge("B", "E")

    # Fan-in: C, D, E 모두 끝나야 F 실행 (LangGraph 자동 동기화)
    g.add_edge("C", "F")
    g.add_edge("D", "F")
    g.add_edge("E", "F")

    g.add_edge("F", "G")
    g.add_edge("G", "H")

    g.add_conditional_edges(
        "H",
        route_after_h,
        {"END": END, "G": "G"},
    )

    return g.compile()
