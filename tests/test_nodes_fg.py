"""nodes_fg.py(Synthesizer·G) 배선 테스트 — 실제 LLM·검색 없이 돈다.

백본(orchestrator.py)의 라우팅 함수에 가짜 워커 C/D/E 를 붙여
  1) 워커 1회 실패 -> 재디스패치 -> 성공 : plan 전부 done, 종합 LLM 1회
  2) 워커 계속 실패 -> 상한 도달          : plan failed, 라벨 판단보류, 보고서 지시에 사유
  3) 보고서가 10장 제한을 넘으면 한 번만 줄여 쓴다
를 확인한다.

    .venv/bin/python -m pytest tests/test_nodes_fg.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from src import nodes_fg as fg  # noqa: E402
from src.node_utils import wrap_worker  # noqa: E402
from src.orchestrator import node_orchestrator, route_after_synthesis, route_dynamic_fanout  # noqa: E402
from src.state import OrchestratorState  # noqa: E402


class FakeStructured:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        labels = self.model.model_fields["labels"].annotation
        return self.model(
            labels=labels(market="압축 유리 조건", stakeholder="조건 의존",
                          domain="확장 유리 조건", trl="차이 없음"),
            conflicts=[], reasoning="가짜",
        )


class FakeLLM:
    def __init__(self, report_lengths):
        self.report_lengths = list(report_lengths)
        self.prompts = []

    def with_structured_output(self, model):
        self.structured = FakeStructured(model)
        return self.structured

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="가" * self.report_lengths.pop(0))


def run_graph(d_failures, report_lengths=(9000,)):
    calls = {"D": 0}

    @wrap_worker("C")
    def c(state):
        return {"label": "압축 유리 조건"}, [{"source": "s1", "detail": "x"}]

    @wrap_worker("D")
    def d(state):
        calls["D"] += 1
        if calls["D"] <= d_failures:
            raise RuntimeError("가짜 장애")
        return {"label": "조건 의존"}, [{"source": "s1", "detail": "x"}, {"source": "s2", "detail": "y"}]

    @wrap_worker("E")
    def e(state):
        return {"label": "확장 유리 조건"}, []

    llm = FakeLLM(report_lengths)
    g = StateGraph(OrchestratorState)
    g.add_node("Orchestrator", node_orchestrator)
    for name, fn in (("C", c), ("D", d), ("E", e)):
        g.add_node(name, fn)
        g.add_edge(name, "Synthesizer")
    g.add_node("Synthesizer", fg.make_node_f(llm))
    g.add_node("G", fg.make_node_g(llm))
    g.add_edge(START, "Orchestrator")
    g.add_conditional_edges("Orchestrator", route_dynamic_fanout, ["C", "D", "E"])
    g.add_conditional_edges("Synthesizer", route_after_synthesis, {"Orchestrator": "Orchestrator", "G": "G"})
    g.add_edge("G", END)
    init = {"tech_research": {}, "tech_references": [{"source": "p", "detail": "q"}]}
    out = g.compile().invoke(init, {"recursion_limit": 30})
    return out, calls, llm


def test_retry_then_success():
    out, calls, llm = run_graph(d_failures=1)
    assert [t["status"] for t in out["plan"]] == ["done", "done", "done"]
    assert calls["D"] == 2                      # 실패한 D 만 재디스패치
    assert llm.structured.calls == 1            # 재시도 전에는 종합하지 않는다
    assert out["synthesis"]["held_perspectives"] == []
    assert out["synthesis"]["labels"]["stakeholder"] == "조건 의존"
    assert "s2" in llm.prompts[0]               # 재시도로 얻은 D 근거가 보고서까지 간다


def test_retry_exhausted_marks_hold():
    out, calls, llm = run_graph(d_failures=99)
    statuses = {t["agent"]: t["status"] for t in out["plan"]}
    assert statuses == {"C": "done", "D": "failed", "E": "done"}
    assert out["synthesis"]["labels"]["stakeholder"] == fg.HOLD   # LLM 답을 코드가 덮는다
    assert out["synthesis"]["held_perspectives"] == ["stakeholder"]
    assert "이해관계자(D): 실패 사유: 가짜 장애 (시도 2회 실패)" in llm.prompts[0]


def test_report_shrinks_once_when_over_limit():
    over = fg.MAX_REPORT_CHARS + 5000
    out, _, llm = run_graph(d_failures=0, report_lengths=(over, over))
    assert len(llm.prompts) == 2                # 재생성은 한 번뿐 (종료 보장)
    assert f"{over:,}자로 제한을 넘었다" in llm.prompts[1]
    assert len(out["final_report"]) == over     # 그래도 넘으면 그대로 넘긴다 — Evaluator 몫


def test_references_deduplicated():
    state = {
        "tech_references": [{"source": "s1", "detail": "x"}],
        "worker_results": [
            {"agent": "C", "output": {}, "references": [{"source": "s1", "detail": "x"}], "status": "ok", "ts": "t"},
            {"agent": "D", "output": {}, "references": [{"source": "s2", "detail": "y"}], "status": "ok", "ts": "t"},
        ],
    }
    assert len(fg.collect_references(state)) == 2


def test_eval_feedback_reaches_prompt():
    out, _, llm = run_graph(d_failures=0, report_lengths=(9000, 9000))
    state = dict(out, eval_result={"passed": False, "missing_items": ["Groundedness: 3장 주장 2개 출처 없음"]})
    fg.make_node_g(llm)(state)
    assert "Groundedness: 3장 주장 2개 출처 없음" in llm.prompts[-1]


# ---------- 동적 재디스패치: 근거 공백 재조사 ----------

def run_gap_graph(gap_agents):
    """gap_agents 의 워커는 1차에 빈 칸을 남기고, 재조사(task.gap_fields 있음)면 채운다."""
    seen = []

    def make(agent):
        def node(state):
            task = state.get("task") or {}
            seen.append((agent, tuple(task.get("gap_fields") or ())))
            gaps = [] if task.get("gap_fields") or agent not in gap_agents else [
                {"field": "market_size_growth", "tech": "공통", "partial": False},
                {"field": "adoption_status", "tech": "InfiniGen", "partial": False},
                {"field": "memory_budget", "tech": "TurboQuant", "partial": True},  # 일부만 빈 칸은 표적 아님
            ]
            return {"worker_results": [{"agent": agent, "output": {"label": "조건 의존"}, "references": [],
                                        "status": "ok", "ts": "t", "evidence_gaps": gaps}]}
        return node

    llm = FakeLLM([9000])
    g = StateGraph(OrchestratorState)
    g.add_node("Orchestrator", node_orchestrator)
    for name in ("C", "D", "E"):
        g.add_node(name, make(name))
        g.add_edge(name, "Synthesizer")
    g.add_node("Synthesizer", fg.make_node_f(llm))
    g.add_node("G", fg.make_node_g(llm))
    g.add_edge(START, "Orchestrator")
    g.add_conditional_edges("Orchestrator", route_dynamic_fanout, ["C", "D", "E"])
    g.add_conditional_edges("Synthesizer", route_after_synthesis, {"Orchestrator": "Orchestrator", "G": "G"})
    g.add_edge("G", END)
    out = g.compile().invoke({"tech_research": {}}, {"recursion_limit": 30})
    return out, seen, llm


def test_gap_rework_redispatches_only_gapped_worker():
    out, seen, llm = run_gap_graph({"C"})
    assert sorted(a for a, f in seen if not f) == ["C", "D", "E"]          # 1라운드 3건
    assert [(a, f) for a, f in seen if f] == [("C", ("market_size_growth", "adoption_status"))]  # 2라운드 1건, partial 제외
    task_c = next(t for t in out["plan"] if t["agent"] == "C")
    assert task_c["reworked"] and task_c["status"] == "done" and "재조사" in task_c["reason"]
    assert llm.structured.calls == 1                                      # 종합은 재조사 뒤 한 번


def test_no_gaps_means_no_second_round():
    out, seen, _ = run_gap_graph(set())
    assert len(seen) == 3 and out.get("orch_retry_count", 0) == 0


def test_gap_queries_are_symmetric():
    from src.worker_utils import gap_queries
    qs = gap_queries({"task": {"gap_fields": ["adoption_status"]}})
    assert qs == ["TurboQuant KV cache adoption status", "InfiniGen KV cache adoption status"]
    assert gap_queries({}) == []


# ---------- Evaluator 장 분할 ----------

def test_eval_split_includes_subsections():
    from src import nodes_eval as ev
    report = "# 3. 기술 개요\n## TurboQuant\n- TRL 6\n## InfiniGen\n- TRL 3\n# 4. 관점별 평가\n## 4.2 이해관계자 평가\n### 4.2.1 a\n- x [6]\n## 4.3 도메인 평가\n- y\n"
    ch = ev._split_chapters(report)
    assert "TRL" in ch["3. 기술 개요"]
    assert ev._citations(ch["4.2 이해관계자 평가"]) == [6]
    assert "[6]" not in ch["4.3 도메인 평가"]


def test_access_date_in_prompt():
    import datetime
    _, _, llm = run_graph(d_failures=0)
    assert datetime.date.today().isoformat() in llm.prompts[0]
