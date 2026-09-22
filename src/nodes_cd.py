"""C, D. 시장성/이해관계자 노드 - 02_agent_CD_market_stakeholder.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import prompts
from src.node_utils import run_web_search, summarize_tech_research
from src.schemas import MarketEval, StakeholderEval


def split_web_references(search_results, queries):
    """run_web_search 가 합쳐 준 문자열을 쿼리 단위로 되쪼개 Reference 목록을 만든다.

    이전 판은 검색 결과 전체를 200자로 잘라 {"source": "web_search"} 한 건으로
    뭉쳤다. 그러면 REFERENCE 장에 URL 이 한 줄도 안 남는다. 여기서는
    쿼리마다 한 건으로 나누고 본문에서 URL 을 뽑아 함께 싣는다.
    """
    import re

    blocks = {}
    current = None
    for line in search_results.split("\n"):
        m = re.match(r"^\[(.+?)\]\s?(.*)$", line)
        if m and m.group(1) in queries:
            current = m.group(1)
            blocks[current] = [m.group(2)]
        elif current is not None:
            blocks[current].append(line)

    refs = []
    for q in queries:
        body = "\n".join(blocks.get(q, []))
        urls = [
            u.rstrip(".,;:)}'\"")
            for u in re.findall(r"https?://\S+", body)
        ]
        seen = []
        for u in urls:
            if u not in seen:
                seen.append(u)
        refs.append(
            {
                "source": f"web_search: {q}",
                "detail": (
                    ("URL: " + " | ".join(seen[:3]) + " / ") if seen else "URL 없음 / "
                )
                + body.strip()[:300],
            }
        )
    return refs


def make_node_c(llm, web_search_tool):
    """C. 시장성 평가."""
    structured_llm = llm.with_structured_output(MarketEval)

    # 대칭 질의: 템플릿 1벌을 두 기술에 기술명만 바꿔 던진다.
    # 쿼리가 기술마다 다르면 입력 자체가 기울어, 프롬프트가 요구한
    # "두 기술 모두 값을 채운다"를 근거로 받쳐 줄 수 없다.
    QUERY_TEMPLATES = [
        "{tech} KV cache adoption production release notes",
        "{tech} KV cache support merged llama.cpp MLX vLLM",
    ]

    def node_c_market_eval(state):
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        queries = [
            t.format(tech=tech)
            for tech in ("TurboQuant", "InfiniGen")
            for t in QUERY_TEMPLATES
        ]
        search_results = run_web_search(web_search_tool, queries)
        prompt = prompts.MARKET_EVAL_PROMPT.format(
            tech_context=tech_context, search_results=search_results
        )
        result = structured_llm.invoke(prompt)
        return {
            "market_eval": result.model_dump(),
            "market_references": split_web_references(search_results, queries),
        }

    return node_c_market_eval


def make_node_d(llm, web_search_tool):
    """D. 이해관계자 평가."""
    structured_llm = llm.with_structured_output(StakeholderEval)

    # 대칭 질의 + 찬반 대칭. 이전 판은 TurboQuant 에 "opinion review" 를,
    # InfiniGen 에 "criticism" 을 물어 한쪽만 비판을 모으는 구조였다.
    # 지금은 두 기술 각각에 지지 방향 1개·비판 방향 1개를 같은 문형으로 던진다.
    QUERY_TEMPLATES = [
        "{tech} KV cache developer feedback adoption",
        "{tech} KV cache criticism limitation concern",
    ]

    def node_d_stakeholder_eval(state):
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        queries = [
            t.format(tech=tech)
            for tech in ("TurboQuant", "InfiniGen")
            for t in QUERY_TEMPLATES
        ]
        search_results = run_web_search(web_search_tool, queries)
        prompt = prompts.STAKEHOLDER_EVAL_PROMPT.format(
            tech_context=tech_context, search_results=search_results
        )
        result = structured_llm.invoke(prompt)
        return {
            "stakeholder_eval": result.model_dump(),
            "stakeholder_references": split_web_references(search_results, queries),
        }

    return node_d_stakeholder_eval

