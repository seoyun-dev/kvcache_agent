"""E. 도메인 평가 워커.

원래 03_agent_E_domain.ipynb가 생성하던 파일인데, Orchestrator-Workers 전환
중에는 이 .py가 원본이다. 노트북 저장 셀을 다시 돌리면 아래 내용이 통째로
날아가니 전환이 끝날 때까지 실행하지 말 것.

E만 소스가 둘이다. 웹 검색을 'RAG가 실패하면 쓰는 폴백'으로 두지 않고 항상
같이 돌리는 이유는, 두 소스가 서로 다른 질문에 답하기 때문이다:

  RAG(도메인 논문 3편) - 기준 쪽. 8GB 기기에서 몇 W 나오는지, 비트폭별 전력,
                         지연 분포. 실측으로 GB 43건·ms 11건·W 5건이 있다.
  웹                   - 기술 쪽. TurboQuant/InfiniGen이 그 기기에서 얼마인지.

도메인 논문 3편에 두 기술의 이름이 각각 0회 나온다. 그래서 RAG는 '실패'하는 게
아니라 원래 기술 쪽 절반을 갖고 있지 않다. 웹으로 대체할 수도 없다 - 기준선이
없으면 기술 수치를 찾아와도 판정이 안 된다. 둘 다 있어야 한 칸이 채워진다.
"""

from src import prompts
from src.ingest import format_docs_for_prompt
from src.node_utils import summarize_tech_research
from src.schemas import DomainEval, SourceAttempt
from src.worker_utils import (
    WorkerOutput,
    collect_source,
    find_evidence_gaps,
    gap_queries,
    search_name,
    run_worker,
    split_web_references,
    to_refs,
    web_search_tally,
)

TECHS = ("TurboQuant", "InfiniGen")
PAPER_REF_LIMIT = 12


def make_node_e(llm, domain_retriever, web_search_tool=None):
    """E. 도메인 평가. tech_research는 재사용(재검색 안 함), 도메인 근거만 새로 모은다.

    web_search_tool에 기본값을 둔 건 노트북 03에서 리트리버만 꽂고 돌려보는
    경우를 위한 것이다. 그래프 경로에서는 graph.py가 항상 주입하므로 이 기본값은
    안 쓰인다. 미주입으로 돌면 RAG 단독으로 가되 그 사실이 sources에 excluded로
    남아, 근거가 적은 이유가 '웹을 안 봤기 때문'인지 '찾아도 없어서'인지 구분된다.
    """
    structured_llm = llm.with_structured_output(DomainEval)

    RAG_QUERIES = [
        "on-device LLM memory budget quantization accuracy tradeoff",
        "edge device memory bandwidth utilization inference latency",
        "flash memory offloading limited DRAM inference",
    ]

    # 전부 '측정값' 방향이다. C가 배포·채택 기록을, D가 사람의 발화를 이미
    # 가져갔으므로 E는 수치만 가져와야 한 근거가 두 관점에 중복 반영되지 않는다
    # (README 확증편향 방지의 '근거 소유권 분리').
    WEB_TEMPLATES = [
        "{tech} KV cache memory footprint GB measured mobile on-device",
        "{tech} tokens per second latency ms smartphone inference benchmark",
        "{tech} power consumption watts thermal throttling mobile SoC",
        "{tech} accuracy perplexity degradation bits quantization tradeoff",
    ]

    def _retrieve():
        # 쿼리별 결과를 전부 모으고 중복만 제거한다. max_docs를 고정값으로 두면
        # 안 된다 - 이전 판은 110청크를 모아놓고 앞 10개(= 쿼리 1 결과)만
        # 프롬프트에 넣어, 쿼리 2·3이 찾아온 메모리·지연 수치가 통째로
        # 버려졌다(실측: 들어간 10개 GB=0·ms=0 / 버려진 100개 GB=35·ms=10).
        # 그래놓고 E는 "근거 없음"이라고 정직하게 보고했다.
        docs, seen = [], set()
        for q in RAG_QUERIES:
            for d in domain_retriever.invoke(q):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    docs.append(d)
        return docs, len(docs)

    def work(state) -> WorkerOutput:
        tech_context = summarize_tech_research(state.get("tech_research", {}))
        sources = []

        docs, paper_source = collect_source("paper", _retrieve, empty=[])
        sources.append(paper_source)

        # 대칭 질의: 한쪽 기술만 비었다고 그쪽만 검색하면 검색 자원이 한 기술에
        # 쏠려 확증편향이 된다. 항상 두 기술 모두 같은 템플릿으로 던진다.
        web_queries = [t.format(tech=search_name(tech)) for tech in TECHS for t in WEB_TEMPLATES]
        web_queries += gap_queries(state)  # 재조사 라운드면 빈 칸 겨냥 검색어 추가
        if web_search_tool is None:
            web_text, hit_queries = "", []
            sources.append(
                SourceAttempt(
                    kind="web", status="excluded", attempts=0,
                    error="web_search_tool 미주입 - RAG 단독 실행",
                )
            )
        else:
            (web_text, hit_queries), web_source = collect_source(
                "web",
                lambda: web_search_tally(web_search_tool, web_queries),
                empty=("", []),
            )
            sources.append(web_source)

        context = format_docs_for_prompt(docs, max_docs=len(docs)) if docs else "(검색 결과 없음)"
        prompt = prompts.DOMAIN_EVAL_PROMPT.format(
            tech_context=tech_context,
            context=context,
            web_context=web_text or "(웹 검색 결과 없음)",
        )
        payload = structured_llm.invoke(prompt).model_dump()

        # 청크 단위로 다 실으면 State가 실행마다 수십 KB씩 분다. 문헌·페이지
        # 단위로 접고 상한을 건다 - REFERENCE 장에 필요한 건 출처이지 원문이 아니다.
        seen_pages, paper_items = set(), []
        for d in docs:
            key = (d.metadata.get("source_name", "unknown"), d.metadata.get("page", "?"))
            if key in seen_pages:
                continue
            seen_pages.add(key)
            paper_items.append({"source": f"{key[0]} p.{key[1]}", "detail": d.page_content})

        refs = to_refs(paper_items, "paper", limit=PAPER_REF_LIMIT)
        if hit_queries:
            refs += to_refs(split_web_references(web_text, hit_queries), "web")

        searched = [s.kind for s in sources if s.status != "excluded"]
        return WorkerOutput(
            payload=payload,
            references=refs,
            gaps=find_evidence_gaps(payload, searched),
            sources=sources,
        )

    def node_e_domain_eval(state):
        return run_worker("E", work, state)

    return node_e_domain_eval
