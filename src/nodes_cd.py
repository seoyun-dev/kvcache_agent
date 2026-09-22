"""C, D. 시장성/이해관계자 노드 - 02_agent_CD_market_stakeholder.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import prompts
from src.node_utils import run_web_search, summarize_tech_research
from src.schemas import MarketEval, StakeholderEval


def make_node_c(llm, web_search_tool):
    """C. 시장성 평가."""
    structured_llm = llm.with_structured_output(MarketEval)

    def node_c_market_eval(state):
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        search_results = run_web_search(
            web_search_tool,
            [
                "TurboQuant KV cache quantization adoption production",
                "InfiniGen KV cache offloading adoption vLLM llama.cpp",
            ],
        )
        prompt = prompts.MARKET_EVAL_PROMPT.format(
            tech_context=tech_context, search_results=search_results
        )
        result = structured_llm.invoke(prompt)
        refs = [{"source": "web_search", "detail": search_results[:200]}]
        return {"market_eval": result.model_dump(), "market_references": refs}

    return node_c_market_eval

def make_node_d(llm, web_search_tool):
    """D. 이해관계자 평가."""
    structured_llm = llm.with_structured_output(StakeholderEval)

    def node_d_stakeholder_eval(state):
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        search_results = run_web_search(
            web_search_tool,
            [
                "TurboQuant developer opinion review",
                "InfiniGen competing method comparison criticism",
            ],
        )
        prompt = prompts.STAKEHOLDER_EVAL_PROMPT.format(
            tech_context=tech_context, search_results=search_results
        )
        result = structured_llm.invoke(prompt)
        refs = [{"source": "web_search", "detail": search_results[:200]}]
        return {"stakeholder_eval": result.model_dump(), "stakeholder_references": refs}

    return node_d_stakeholder_eval
