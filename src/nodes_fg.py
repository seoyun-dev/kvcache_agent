"""F, G. 평가 종합(Synthesizer) / 보고서 생성 — Orchestrator-Workers 패턴의 수렴부.

nodes_fgh.py 에서 F·G 만 떼어 왔다. H(품질 평가)는 nodes_eval.py 로 갔다.
노트북에서 생성하지 않고 이 파일을 직접 고친다. 배선 테스트는 tests/test_nodes_fg.py.

바뀐 점
- 원자료를 관점별 전용 키(market_eval 등)가 아니라 worker_results 에서 읽는다.
  읽는 길은 node_utils.get_worker_output / get_worker_references 하나로 통일한다
  (재시도로 같은 agent 가 두 번 들어와도 최신 시도만 본다).
- Synthesizer 는 plan 의 pending 태스크를 done/failed 로 닫아 같이 리턴한다.
  route_after_synthesis 가 이 값을 보고 재디스패치 여부를 정한다.
- 실패한 관점의 라벨은 LLM 에 맡기지 않고 코드로 '판단보류' 를 박는다.
  (근거 없으면 보류 — 확률적 판정에 맡기지 않는다)
- G 는 보고서가 최대 10장을 넘는지 보고, 넘으면 한 번만 줄여 쓰게 한다.
"""
import datetime as _dt

from pydantic import create_model

from src import config, prompts
from src.node_utils import get_worker_output, get_worker_references, latest_worker_result
from src.orchestrator import route_after_synthesis
from src.schemas import Label

HOLD: str = "판단보류"

# 워커 agent → (synthesis 라벨 키, 기존 프롬프트 슬롯, 한국어 이름)
WORKERS = {
    "C": ("market", "market_eval", "시장성"),
    "D": ("stakeholder", "stakeholder_eval", "이해관계자"),
    "E": ("domain", "domain_eval", "도메인"),
}

# 10장 제한. 페이지는 PDF 로 바꿔야 정확히 알 수 있어서 글자 수(파이썬 len)로 근사한다.
# 실측 (2026-10-07, A4 · 본문 10.5pt · 여백 20mm 로 Chrome PDF 변환):
#   RAG 과제 제출본 11,720자 -> 6쪽 (실제 제출에서도 10장 안이었다)
#   이 그래프 첫 통과본 14,336자 -> 7쪽  => 약 2,000자/쪽
# 조가 쓰는 서식이 더 클 수 있어 한 장을 1,600자로 보수적으로 잡는다 -> 상한 16,000자(위 서식 약 8쪽).
# ⚠ wc -m 으로 세지 말 것 - 로케일이 비어 있으면 한글을 바이트 단위로 세서 1.6배로 나온다.
MAX_PAGES = 10
CHARS_PER_PAGE = 1600
MAX_REPORT_CHARS = MAX_PAGES * CHARS_PER_PAGE
MAX_SHRINK_RETRY = 1  # 분량 초과 시 재생성 상한 (G 내부 루프 종료 보장)


def _ok(state, agent):
    r = latest_worker_result(state, agent)
    return bool(r) and r["status"] == "ok"


def held_agents(state):
    """성공한 최신 결과가 없는 워커. 계획에 없던 워커도 여기 들어간다."""
    return [a for a in WORKERS if not _ok(state, a)]


def close_plan(state):
    """pending 태스크를 이번 시도 결과로 닫는다. 이미 done 인 태스크는 그대로 둔다."""
    return [
        {**t, "status": "done" if _ok(state, t["agent"]) else "failed"}
        if t["status"] == "pending" else t
        for t in state.get("plan", [])
    ]


def collect_references(state):
    """B 의 기술 근거 + 성공한 워커 근거. 같은 출처는 한 번만."""
    refs, seen = [], set()
    pool = list(state.get("tech_references", []))
    for a in WORKERS:
        pool += get_worker_references(state, a)
    for ref in pool:
        key = (ref.get("source"), ref.get("detail"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


# ---------- F. Synthesizer ----------

def make_node_f(llm):
    """F. 평가 종합. 워커 결과를 모아 4관점(시장/이해관계자/도메인/TRL) 라벨을 낸다.

    labels 를 create_model 로 고정하는 이유는 nodes_fgh.py 때와 같다 —
    schemas.Synthesis.labels 가 자유 dict 라 strict 구조화 출력이 거부된다.
    """
    labels_model = create_model(
        "SynthesisLabels",
        market=(Label, ...),
        stakeholder=(Label, ...),
        domain=(Label, ...),
        trl=(Label, ...),
    )
    strict_model = create_model(
        "SynthesisStrict",
        labels=(labels_model, ...),
        conflicts=(list[str], ...),
        reasoning=(str, ...),
    )
    structured_llm = llm.with_structured_output(strict_model)

    def node_f_synthesis(state):
        plan = close_plan(state)

        # 곧 Orchestrator 로 되돌아갈 판이면(실패 재시도·근거 공백 재조사) 종합은 미룬다 —
        # 재디스패치 결과까지 모은 뒤 한 번만 한다. 판단은 라우터와 같은 함수로 한다.
        if route_after_synthesis({**state, "plan": plan}) == "Orchestrator":
            return {"plan": plan}

        held = held_agents(state)
        slots = {slot: get_worker_output(state, a) for a, (_, slot, _) in WORKERS.items()}

        tech_research = state.get("tech_research", {})
        trl_eval = {name: r.get("trl_assessment", {}) for name, r in tech_research.items()}

        prompt = prompts.SYNTHESIS_PROMPT.format(**slots, trl_eval=trl_eval)
        result = structured_llm.invoke(prompt)

        labels = result.labels.model_dump()
        for a in held:  # LLM 이 뭐라 했든 근거 없는 관점은 보류
            labels[WORKERS[a][0]] = HOLD

        return {
            "plan": plan,
            "synthesis": {
                "labels": labels,
                "conflicts": result.conflicts,
                "reasoning": result.reasoning,
                "held_perspectives": [WORKERS[a][0] for a in held],
            },
        }

    return node_f_synthesis


# ---------- G. 보고서 생성 ----------

def _eval_feedback(state):
    """Evaluator 가 미달 판정을 냈으면 재작성 지시로 바꾼다.
    eval_result 계약(nodes_eval.py): {"passed", "missing_items", "forced_pass", "feedback", ...}"""
    ev = state.get("eval_result")
    if not ev or ev.get("passed", True):
        return ""
    if ev.get("feedback"):  # nodes_eval._revision_note 가 만들어 둔 지시문
        return ev["feedback"]
    items = ev.get("missing_items", [])
    return "[재작성 지시] 품질 평가에서 아래 항목이 미달이었다. 이 항목만 고쳐라.\n" + "\n".join(
        f"- {i}" for i in items
    )


def _held_note(state, held):
    """보류 관점과 실패 사유(state['errors'])를 보고서 한계점에 쓰도록 넘긴다."""
    if not held:
        return ""
    lines = []
    for a in held:
        errs = [e["error"] for e in state.get("errors", []) if e.get("agent") == a]
        reason = f"실패 사유: {errs[-1]}" if errs else "실행 기록 없음"
        lines.append(f"- {WORKERS[a][2]}({a}): {reason} (시도 {len(errs)}회 실패)")
    return (
        "[판단보류 관점] 아래 관점은 워커가 근거를 확보하지 못했다(재시도 상한 도달).\n"
        + "\n".join(lines)
        + "\n해당 장에는 비교 판단을 쓰지 말고 '판단보류' 와 사유를 명시하며, 한계점에도 적는다."
    )


def _access_date_note():
    today = _dt.date.today().isoformat()
    return (
        f"[작성일] 오늘은 {today} 이다. REFERENCE 웹페이지의 '접근:' 날짜와 "
        f"한계점의 웹 검색 시점은 {today} 로 쓴다."
    )


def _length_rule(extra=""):
    rule = (
        f"[분량] 보고서 전체를 A4 {MAX_PAGES}장 이내로 쓴다 — 약 {int(MAX_REPORT_CHARS * 0.9):,}자를 목표로, "
        f"{MAX_REPORT_CHARS:,}자를 넘기지 않는다. "
        "앞의 '분량은 충분히 길어도 좋다' 보다 이 제한이 우선한다. "
        "수치·출처는 유지하고 반복 서술을 줄인다."
    )
    return rule + (" " + extra if extra else "")


def make_node_g(llm):
    """G. 보고서 생성. Evaluator 가 미달이면 재진입해서 재작성 지시를 받는다."""

    def node_g_report(state):
        held = held_agents(state)
        references = collect_references(state)

        # F 라벨만 넘기면 E 의 GB·%p·ms·W 수치와 각 관점 notes 가 보고서에 닿지 않는다.
        # REPORT_PROMPT 를 고치지 않고 {synthesis} 슬롯에 종합과 원자료를 함께 싣는다.
        perspective_block = {"관점 간 종합(F)": state.get("synthesis", {})}
        for a, (_, _, ko) in WORKERS.items():
            perspective_block[f"{ko} 원자료({a})"] = get_worker_output(state, a)

        def generate(extra_note):
            notes = [
                n for n in (
                    _eval_feedback(state), _held_note(state, held), _access_date_note(), _length_rule(extra_note)
                ) if n
            ]
            prompt = prompts.REPORT_PROMPT.format(
                analysis_background=config.ANALYSIS_BACKGROUND,
                selected_technologies=state.get("selected_technologies", {}),
                tech_research=state.get("tech_research", {}),
                synthesis=perspective_block,
                references=references,
                revision_note="\n\n".join(notes),
            )
            return llm.invoke(prompt).content

        report = generate("")
        shrinks = 0
        while len(report) > MAX_REPORT_CHARS and shrinks < MAX_SHRINK_RETRY:
            shrinks += 1
            report = generate(
                f"직전 초안이 {len(report):,}자로 제한을 넘었다. {MAX_REPORT_CHARS:,}자 이내로 다시 쓴다."
            )

        return {"final_report": report}

    return node_g_report
