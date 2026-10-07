"""
메인 에이전트 그래프 - Orchestrator-Workers 패턴.

    START -> A -> B -> Orchestrator =(Send, 동적 fan-out)=> {C, D, E}
          -> Synthesizer(F) -(실패 Worker 있음 & retry<MAX_RETRY_ORCH)-> Orchestrator
                             -(전원 완료 또는 재시도 상한)-> G
          -> G <-> Evaluator -(통과 또는 재시도 상한)-> END

기존 고정 DAG(A->B->{C,D,E}->F->G<->H) 대비 바뀐 지점 두 곳:
  1) B 다음이 바로 C/D/E 고정 fan-out이 아니라 Orchestrator가 plan을 세운 뒤
     Send로 내보낸다 (orchestrator.py).
  2) H(규칙 기반 형식 검사만 하던 노드) 자리가 Evaluator로 바뀌어 Groundedness/
     중립성/편향통제/관점커버리지까지 본다 (가이드 D절).

노드 함수 대부분은 notebooks에서 담당자가 만들어 저장한 src/nodes_*.py에서
가져온다. A(이 파일 포함 state.py/orchestrator.py/config.py)만 직접 고치고,
나머지는 아래 except 블록에 적힌 계약대로 각자 노트북에서 채워 넣는다 -
그래야 노트북 저장 셀을 다시 돌려도 이 배선과 어긋나지 않는다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import config
from .orchestrator import node_orchestrator, route_after_synthesis, route_dynamic_fanout
from .state import OrchestratorState

try:
    from .nodes_b import make_node_b
    from .nodes_cd import make_node_c, make_node_d
    from .nodes_e import make_node_e
    from .nodes_fg import make_node_f, make_node_g
    from .nodes_eval import make_node_evaluator, route_after_eval
except ImportError as e:
    raise ImportError(
        "Orchestrator-Workers 구조에 맞춰 아직 준비 안 된 모듈이 있다:\n"
        "\n"
        "  nodes_cd.py / nodes_e.py (워커 담당)\n"
        "    make_node_c(llm, web_search_tool), make_node_d(llm, web_search_tool),\n"
        "    make_node_e(llm_full, domain_retriever, web_search_tool) 이 리턴하는\n"
        "    노드 함수는 {'market_eval':...} 식이 아니라 node_utils.wrap_worker(agent)\n"
        "    데코레이터로 감싸 (output: dict, references: list) 튜플을 리턴해야 한다.\n"
        "    (E는 기존 RAG 전용에서 RAG+Web Search 병행으로 바뀌는 중이라 인자가\n"
        "    하나 늘었다 - web_search_tool 추가.)\n"
        "\n"
        "  nodes_fg.py (신규 - 기존 nodes_fgh.py에서 F/G만 분리)\n"
        "    make_node_f(llm): state['worker_results']를 node_utils.get_worker_output/\n"
        "    latest_worker_result로 읽어 종합하고, 끝에 plan의 각 SubTask.status를\n"
        "    done/failed로 갱신해 같이 리턴해야 한다(route_after_synthesis가 그걸 봄).\n"
        "    make_node_g(llm_full): 기존 로직 그대로, 원자료 접근만\n"
        "    node_utils.get_worker_output/get_worker_references 경유로 바꾸면 된다.\n"
        "\n"
        "  nodes_eval.py (신규 - 기존 nodes_fgh.py의 H를 확장 이전)\n"
        "    make_node_evaluator(llm_full), route_after_eval(state).\n"
        "    state['final_report']를 평가해 state['eval_result']에\n"
        "    {'passed': bool, 'missing_items': [...], 'forced_pass': bool, ...} 리턴.\n"
        "    기존 nodes_fgh.py의 _check_format/FORBIDDEN_WORDS 규칙(중립성·편향통제)은\n"
        "    그대로 재사용하고, Groundedness·관점커버리지는 LLM-judge로 추가한다.\n"
        "    config.MAX_RETRY_EVAL이 재시도 상한.\n"
        f"\n(원래 에러: {e})"
    ) from e


def node_a_select_technologies(state: OrchestratorState) -> dict:
    """A. 기술 선정 (Human 기반, 정적 노드 - LLM 호출 없음)."""
    return {
        "selected_technologies": config.SELECTED_TECHNOLOGIES,
        "target_domain": config.TARGET_DOMAIN,
    }


def build_graph(llm, llm_full, tech_retriever, domain_retriever, web_search_tool):
    """실제 의존성을 주입해 컴파일된 그래프를 반환.

    llm      : gpt-4.1-mini (C, D, F 등 단순 구조화 출력용)
    llm_full : gpt-4.1      (B, E, G, Evaluator 등 품질이 중요한 노드용)
    """
    g = StateGraph(OrchestratorState)

    g.add_node("A", node_a_select_technologies)
    g.add_node("B", make_node_b(llm_full, tech_retriever, web_search_tool))
    g.add_node("Orchestrator", node_orchestrator)
    g.add_node("C", make_node_c(llm, web_search_tool))
    g.add_node("D", make_node_d(llm, web_search_tool))
    g.add_node("E", make_node_e(llm_full, domain_retriever, web_search_tool))
    g.add_node("Synthesizer", make_node_f(llm))
    g.add_node("G", make_node_g(llm_full))
    g.add_node("Evaluator", make_node_evaluator(llm_full))

    g.add_edge(START, "A")
    g.add_edge("A", "B")
    g.add_edge("B", "Orchestrator")

    # 동적 fan-out: 고정 g.add_edge(..., "C") 세 줄이 아니라 plan 기반 Send.
    g.add_conditional_edges("Orchestrator", route_dynamic_fanout, ["C", "D", "E"])

    # fan-in: Send로 실제 호출된 워커만 여기로 모인다 (LangGraph가 자동 동기화).
    g.add_edge("C", "Synthesizer")
    g.add_edge("D", "Synthesizer")
    g.add_edge("E", "Synthesizer")

    # fallback: 실패 워커 있으면 Orchestrator 재진입(해당 워커만 재시도), 아니면 G.
    g.add_conditional_edges(
        "Synthesizer", route_after_synthesis, {"Orchestrator": "Orchestrator", "G": "G"}
    )

    g.add_edge("G", "Evaluator")
    g.add_conditional_edges("Evaluator", route_after_eval, {"END": END, "G": "G"})

    return g.compile()
