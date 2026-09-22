"""B. 기술조사 노드 - 01_agent_B_tech_research.ipynb에서 생성됨.
이 파일을 직접 고치지 말고, 노트북에서 고친 뒤 저장 셀을 다시 실행할 것."""

from src import prompts
from src.ingest import format_docs_for_prompt
from src.node_utils import run_web_search
from src.schemas import TechResearch


def make_node_b(llm, tech_retriever, web_search_tool):
    """B. 기술조사 노드를 만든다. llm/tech_retriever/web_search_tool을
    미리 받아 클로저로 갖고 있다가, 실제 노드 함수(node_b_tech_research)가
    호출될 때 사용한다."""
    structured_llm = llm.with_structured_output(TechResearch)

    def node_b_tech_research(state):
        queries = [
            "TurboQuant KV cache quantization bits distortion",
            "InfiniGen KV cache offloading prefetch speedup",
        ]
        all_docs = []
        for q in queries:
            all_docs.extend(tech_retriever.invoke(q))
        context = format_docs_for_prompt(all_docs, max_docs=10)

        deployment_search_results = run_web_search(
            web_search_tool,
            [
                "TurboQuant official runtime production deployment",
                "InfiniGen official runtime vLLM llama.cpp support",
            ],
        )

        prompt = prompts.TECH_RESEARCH_PROMPT.format(
            context=context, deployment_search_results=deployment_search_results
        )
        result = structured_llm.invoke(prompt)

        refs = [
            {"source": d.metadata.get("source_name", "unknown"), "detail": d.page_content[:200]}
            for d in all_docs
        ]
        refs.append({"source": "web_search(TRL 배포 확인용)",
                      "detail": deployment_search_results[:200]})
        return {
            "tech_research": {
                "TurboQuant": result.turboquant.model_dump(),
                "InfiniGen": result.infinigen.model_dump(),
            },
            "tech_references": refs,
        }

    return node_b_tech_research
