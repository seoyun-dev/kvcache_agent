"""F, G, H. 평가종합/보고서생성/보고서검증 노드 - 04_agent_FGH_synthesis_report_validate.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import config, prompts
from src.schemas import Label, ValidationResult


def make_node_f(llm):
    """F. 평가 종합. 4관점(시장/이해관계자/도메인/TRL) 라벨을 모아 비교한다.

    labels 스키마를 여기서 고정해 만든다 — 공유 파일 schemas.py 의
    Synthesis.labels 가 dict[str, str](자유 형식 object)라 두 가지가 깨진다.
      1) OpenAI 기본 strict 구조화 출력(json_schema)이 400 으로 거부한다.
         "'required' ... must include every key in properties"
      2) method="function_calling" 으로 우회하면 호출마다 모양이 달라진다.
         실측: 1회차는 한국어 키 dict, 2회차는 list 가 돌아와 ValidationError.
    schemas.py 를 건드리지 않으려고 create_model 로 고정 스키마를 만들어
    LLM 에 넘기고, State 에는 기존과 똑같은 dict 모양으로 되돌려 넣는다.
    (근본 해결은 schemas.py 의 labels 를 고정 필드로 바꾸는 것 — 조 합의 필요)
    클래스 문으로 안 쓰는 이유: Jupyter 셀에서 정의한 클래스는
    inspect.getsource 가 못 읽어 '파일로 저장' 셀이 깨진다.
    """
    from pydantic import create_model

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
        tech_research = state.get("tech_research", {})
        trl_eval = {
            name: r.get("trl_assessment", {}) for name, r in tech_research.items()
        }
        prompt = prompts.SYNTHESIS_PROMPT.format(
            market_eval=state.get("market_eval", {}),
            stakeholder_eval=state.get("stakeholder_eval", {}),
            domain_eval=state.get("domain_eval", {}),
            trl_eval=trl_eval,
        )
        result = structured_llm.invoke(prompt)
        # State 모양은 기존 Synthesis.model_dump() 와 동일하게 유지한다.
        return {
            "synthesis": {
                "labels": result.labels.model_dump(),
                "conflicts": result.conflicts,
                "reasoning": result.reasoning,
            }
        }

    return node_f_synthesis


def make_node_g(llm):
    """G. 보고서 생성. H가 무효 판정을 내리면 재진입해서 revision_note를 받는다."""
    def node_g_report(state):
        validation = state.get("validation_result")
        revision_note = ""
        if validation and not validation.get("is_valid", True):
            items = validation.get("missing_items", [])
            absent = [i for i in items if not i.startswith("서열 표현")]
            violations = [i for i in items if i.startswith("서열 표현")]
            parts = []
            if absent:
                parts.append(
                    f"다음 장이 빠졌다: {', '.join(absent)}. 이번엔 반드시 포함하라."
                )
            if violations:
                parts.append(
                    f"다음이 본문에 들어 있다: {', '.join(violations)}. "
                    "해당 표현을 지우거나 중립 서술로 바꿔라. "
                    "단 금칙어를 설명하는 문장 자체도 쓰지 말 것."
                )
            revision_note = "[재작성 지시] " + " ".join(parts)

        all_references = (
            state.get("tech_references", [])
            + state.get("market_references", [])
            + state.get("stakeholder_references", [])
            + state.get("domain_references", [])
        )

        # F가 만든 라벨만 넘기면 E가 뽑은 GB·%p·ms·W 수치와 각 관점의 notes가
        # 보고서에 도달하지 않는다. REPORT_PROMPT(공유 파일)를 고치지 않고
        # {synthesis} 슬롯에 종합과 원자료를 함께 실어 보낸다.
        perspective_block = {
            "관점 간 종합(F)": state.get("synthesis", {}),
            "시장성 원자료(C)": state.get("market_eval", {}),
            "이해관계자 원자료(D)": state.get("stakeholder_eval", {}),
            "도메인 원자료(E)": state.get("domain_eval", {}),
        }

        prompt = prompts.REPORT_PROMPT.format(
            analysis_background=config.ANALYSIS_BACKGROUND,
            selected_technologies=state.get("selected_technologies", {}),
            tech_research=state.get("tech_research", {}),
            synthesis=perspective_block,
            references=all_references,
            revision_note=revision_note,
        )
        report_text = llm.invoke(prompt).content
        return {"final_report": report_text}

    return node_g_report


REQUIRED_CHAPTERS = ['SUMMARY', '시장', '이해관계자', '도메인', 'REFERENCE']
FORBIDDEN_WORDS = ['우수', '우월', '우위', '권장', '추천']
RANKING_CHECK_CHAPTERS = ['관점별 평가', '시사점']


def _split_chapters(report):
    """마크다운 헤더 기준으로 보고서를 {장 제목: 본문}으로 자른다."""
    import re

    chapters = {}
    title, buf = "(머리말)", []
    for line in report.split("\n"):
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if m:
            chapters[title] = "\n".join(buf)
            title, buf = m.group(1), []
        else:
            buf.append(line)
    chapters[title] = "\n".join(buf)
    return chapters


def node_h_validate(state):
    report = state.get("final_report", "")
    retry_count = state.get("retry_count", 0)
    chapters = _split_chapters(report)
    titles = list(chapters.keys())

    # 장 제목에서 찾는다. 본문에 "시장"이라는 낱말이 있다고 장이 있는 건 아니다.
    missing = [c for c in REQUIRED_CHAPTERS if not any(c in t for t in titles)]

    # 서열어는 지정한 장 안에서만 본다.
    scoped = "\n".join(
        body
        for t, body in chapters.items()
        if any(k in t for k in RANKING_CHECK_CHAPTERS)
    )
    forbidden_found = [w for w in FORBIDDEN_WORDS if w in scoped]
    if forbidden_found:
        missing.append(f"서열 표현 발견: {forbidden_found}")

    if not missing:
        result = ValidationResult(is_valid=True, missing_items=[], forced_pass=False)
        return {"validation_result": result.model_dump(), "retry_count": retry_count}

    new_retry_count = retry_count + 1
    if new_retry_count > config.MAX_RETRY_H:
        # 상한 도달. 통과시키되 검증 실패 사실을 보고서에 남긴다.
        # 이게 없으면 검증에 실패한 보고서가 통과 표시로 제출물이 되고,
        # 콘솔 print 말고는 어디에도 흔적이 남지 않는다.
        result = ValidationResult(is_valid=True, missing_items=missing, forced_pass=True)
        return {
            "validation_result": result.model_dump(),
            "retry_count": new_retry_count,
            "final_report": _append_audit_note(report, missing, new_retry_count),
        }

    result = ValidationResult(is_valid=False, missing_items=missing, forced_pass=False)
    return {"validation_result": result.model_dump(), "retry_count": new_retry_count}


def _append_audit_note(report, missing, retry_count):
    """forced_pass 시 검증 기록을 「한계점」 장 뒤(REFERENCE 앞)에 끼워 넣는다."""
    import re

    note = (
        "\n\n### 보고서 검증 기록 (자동 생성)\n\n"
        f"검증 노드(H)가 재작성 상한({config.MAX_RETRY_H}회)에 도달해 "
        f"아래 항목을 충족하지 못한 채 통과 처리했다. 재작성 시도 {retry_count}회.\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n"
    )
    for line in report.split("\n"):
        if re.match(r"^\s{0,3}#{1,6}\s+.*REFERENCE", line):
            return report.replace(line, note.strip() + "\n\n" + line, 1)
    return report + note


def route_after_h(state):
    """H 다음 조건부 엣지. graph.py의 add_conditional_edges가 이 함수를 쓴다."""
    validation = state.get("validation_result", {})
    if validation.get("is_valid", False):
        return "END"
    return "G"

