"""E. 도메인 평가 노드 - 03_agent_E_domain.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import prompts
from src.ingest import format_docs_for_prompt
from src.node_utils import summarize_tech_research
from src.schemas import DomainEval


def make_node_e(llm, domain_retriever):
    """E. 도메인 평가. tech_research는 재사용(재검색 안 함), 도메인 논문만 새로 검색."""
    structured_llm = llm.with_structured_output(DomainEval)

    def node_e_domain_eval(state):
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        queries = [
            "on-device LLM memory budget quantization accuracy tradeoff",
            "edge device memory bandwidth utilization inference latency",
            "flash memory offloading limited DRAM inference",
        ]
        all_docs = []
        for q in queries:
            all_docs.extend(domain_retriever.invoke(q))
        context = format_docs_for_prompt(all_docs, max_docs=10)

        prompt = prompts.DOMAIN_EVAL_PROMPT.format(
            tech_context=tech_context, context=context
        )
        result = structured_llm.invoke(prompt)
        refs = [
            {"source": d.metadata.get("source_name", "unknown"), "detail": d.page_content[:200]}
            for d in all_docs
        ]
        return {"domain_eval": result.model_dump(), "domain_references": refs}

    return node_e_domain_eval

