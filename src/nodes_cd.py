"""C, D. 시장성/이해관계자 워커.

원래 02_agent_CD_market_stakeholder.ipynb가 생성하던 파일인데, Orchestrator-
Workers 전환 중에는 이 .py가 원본이다. 노트북 저장 셀을 다시 돌리면 아래
워커 래핑이 통째로 날아가니 전환이 끝날 때까지 실행하지 말 것.

수집·판정 로직은 건드리지 않았다. 바뀐 건 반환 모양뿐 - 관점별 State 키에
직접 쓰던 것을 WorkerResult 하나로 감싸 worker_results에 누적한다.
"""

from src import prompts
from src.node_utils import summarize_tech_research
from src.schemas import MarketEval, StakeholderEval
from src.worker_utils import (
    WorkerOutput,
    collect_source,
    find_evidence_gaps,
    gap_queries,
    run_worker,
    split_web_references,
    to_refs,
    web_search_tally,
)

TECHS = ("TurboQuant", "InfiniGen")


def _web_only_worker(agent, structured_llm, prompt_template, web_search_tool, templates):
    """C와 D는 소스가 웹 하나뿐이라 흐름이 완전히 같다 - 쿼리와 프롬프트만 다르다."""

    def work(state) -> WorkerOutput:
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        queries = [t.format(tech=tech) for tech in TECHS for t in templates]
        queries += gap_queries(state)  # 재조사 라운드면 빈 칸 겨냥 검색어 추가

        (search_results, hit_queries), source = collect_source(
            "web", lambda: web_search_tally(web_search_tool, queries), empty=("", [])
        )

        prompt = prompt_template.format(
            tech_context=tech_context, search_results=search_results
        )
        payload = structured_llm.invoke(prompt).model_dump()

        refs = to_refs(split_web_references(search_results, hit_queries), "web") if hit_queries else []
        searched = [s.kind for s in (source,) if s.status != "excluded"]
        return WorkerOutput(
            payload=payload,
            references=refs,
            gaps=find_evidence_gaps(payload, searched),
            sources=[source],
        )

    def node(state):
        return run_worker(agent, work, state)

    return node


def make_node_c(llm, web_search_tool):
    """C. 시장성 평가 워커."""
    # 대칭 질의: 템플릿 1벌을 두 기술에 기술명만 바꿔 던진다.
    QUERY_TEMPLATES = [
        "{tech} KV cache adoption production release notes 2024 2025 2026",
        "{tech} KV cache support merged llama.cpp MLX vLLM GitHub",
        "{tech} on-device edge deployment smartphone integration",
    ]
    return _web_only_worker(
        "C",
        llm.with_structured_output(MarketEval),
        prompts.MARKET_EVAL_PROMPT,
        web_search_tool,
        QUERY_TEMPLATES,
    )


def make_node_d(llm, web_search_tool):
    """D. 이해관계자 평가 워커."""
    # 대칭 질의 + 찬반 대칭: 기술마다 지지/비판/투자 3방향으로 검색.
    QUERY_TEMPLATES = [
        "{tech} KV cache developer feedback adoption benchmark results",
        "{tech} KV cache criticism limitation concern drawback",
        "{tech} KV cache investor analyst report coverage media 2024 2025 2026",
        "{tech} vs competing KV cache method comparison response",
    ]
    return _web_only_worker(
        "D",
        llm.with_structured_output(StakeholderEval),
        prompts.STAKEHOLDER_EVAL_PROMPT,
        web_search_tool,
        QUERY_TEMPLATES,
    )
