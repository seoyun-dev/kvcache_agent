"""F, G, H. 평가종합/보고서생성/보고서검증 노드 - 04_agent_FGH_synthesis_report_validate.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import config, prompts
from src.schemas import Synthesis, ValidationResult


def make_node_f(llm):
    """F. 평가 종합. 4관점(시장/이해관계자/도메인/TRL) 라벨을 모아 비교한다."""
    structured_llm = llm.with_structured_output(Synthesis)

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
        return {"synthesis": result.model_dump()}

    return node_f_synthesis

def make_node_g(llm):
    """G. 보고서 생성. H가 무효 판정을 내리면 재진입해서 revision_note를 받는다."""
    def node_g_report(state):
        validation = state.get("validation_result")
        revision_note = ""
        if validation and not validation.get("is_valid", True):
            missing = ", ".join(validation.get("missing_items", []))
            revision_note = (
                f"[재작성 지시] 이전 초안에서 다음이 빠졌다: {missing}. "
                "이번엔 반드시 포함하라."
            )

        all_references = (
            state.get("tech_references", [])
            + state.get("market_references", [])
            + state.get("stakeholder_references", [])
            + state.get("domain_references", [])
        )

        prompt = prompts.REPORT_PROMPT.format(
            analysis_background=config.ANALYSIS_BACKGROUND,
            selected_technologies=state.get("selected_technologies", {}),
            tech_research=state.get("tech_research", {}),
            synthesis=state.get("synthesis", {}),
            references=all_references,
            revision_note=revision_note,
        )
        report_text = llm.invoke(prompt).content
        return {"final_report": report_text}

    return node_g_report

REQUIRED_CHAPTERS = ['SUMMARY', '시장', '이해관계자', '도메인', 'REFERENCE']
FORBIDDEN_WORDS = ['우수', '우월', '우위', '권장', '추천']


def node_h_validate(state):
    report = state.get("final_report", "")
    retry_count = state.get("retry_count", 0)

    missing = [c for c in REQUIRED_CHAPTERS if c not in report]
    forbidden_found = [w for w in FORBIDDEN_WORDS if w in report]
    if forbidden_found:
        missing.append(f"서열 표현 발견: {forbidden_found}")

    if not missing:
        result = ValidationResult(is_valid=True, missing_items=[], forced_pass=False)
        return {"validation_result": result.model_dump(), "retry_count": retry_count}

    new_retry_count = retry_count + 1
    if new_retry_count > config.MAX_RETRY_H:
        result = ValidationResult(is_valid=True, missing_items=missing, forced_pass=True)
    else:
        result = ValidationResult(is_valid=False, missing_items=missing, forced_pass=False)

    return {"validation_result": result.model_dump(), "retry_count": new_retry_count}


def route_after_h(state):
    """H 다음 조건부 엣지. graph.py의 add_conditional_edges가 이 함수를 쓴다."""
    validation = state.get("validation_result", {})
    if validation.get("is_valid", False):
        return "END"
    return "G"
