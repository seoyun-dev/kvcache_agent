"""
각 노트북의 코드 셀을 실제로 실행해서 문법·런타임 에러를 잡는다.
실제 PDF·API 키가 필요한 셀(리트리버 구축, 실제 LLM 호출, graph.invoke 등)은
건너뛴다 - 이 샌드박스엔 그게 없다. 나머지(함수 정의, 가짜 LLM 테스트, 파일
저장)는 전부 실제로 돌려서 검증한다.

주의: inspect.getsource()가 셀 소스를 제대로 찾으려면 linecache에 그 셀의
내용을 등록해둬야 한다 - 실제 Jupyter/IPython은 셀 실행마다 이걸 자동으로
해준다(그래서 노트북 안에서는 inspect.getsource가 잘 동작한다). 이 스크립트는
그 동작을 흉내낸다.
"""
import linecache
import sys
import traceback

import nbformat

SKIP_PATTERNS = [
    "build_tech_retriever(",
    "build_domain_retriever(",
    "TavilySearch(",
    "init_chat_model(",
    "graph.invoke(",
    "report_path.write_text",
]


def should_skip(src: str) -> bool:
    return any(p in src for p in SKIP_PATTERNS)


def run_notebook(path: str, ns: dict) -> bool:
    nb = nbformat.read(path, as_version=4)
    ok = True
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if should_skip(src):
            print(f"  [skip cell {i}] (실제 리소스 필요)")
            continue
        fname = f"<{path}::cell{i}>"
        # IPython이 셀마다 하는 걸 흉내: linecache에 이 "가짜 파일"의 내용을
        # 미리 등록해서 inspect.getsource()가 올바른 소스를 찾게 한다.
        linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
        try:
            exec(compile(src, fname, "exec"), ns)
        except Exception:
            print(f"  [FAIL cell {i}] {path}")
            print(src[:300])
            traceback.print_exc()
            ok = False
    return ok


if __name__ == "__main__":
    import os
    os.chdir("notebooks")  # 노트북들이 상대경로로 ../src를 참조하므로 cwd를 맞춘다
    sys.path.insert(0, ".")

    all_ok = True
    for nb_path in [
        "01_agent_B_tech_research.ipynb",
        "02_agent_CD_market_stakeholder.ipynb",
        "03_agent_E_domain.ipynb",
        "04_agent_FGH_synthesis_report_validate.ipynb",
    ]:
        print(f"=== {nb_path} ===")
        ns = {"__name__": "__notebook_cell__"}
        ok = run_notebook(nb_path, ns)
        all_ok = all_ok and ok
        print("  OK" if ok else "  FAILED")
        print()

    print("=== 05_full_graph_run.ipynb (05는 앞의 네 파일이 실제로 저장됐어야 함) ===")
    print("  (1차 통과: import만 확인. build_graph 호출 셀은 llm 등이 실제 리소스라")
    print("   여기선 당연히 NameError가 난다 - 아래 fake 주입 재검증이 진짜 결과)")
    ns5 = {"__name__": "__notebook_cell__"}
    run_notebook("05_full_graph_run.ipynb", ns5)

    # cell 7,8(진짜 리소스)을 스킵해서 llm/tech_retriever/... 가 없는 채로
    # cell 10(build_graph 정의+호출)이 실패했을 것이다. 진짜 fake 객체를 넣어서
    # notebook 05 "자신의" build_graph가 - 즉 01~04에서 새로 만든 nodes_b/cd/e/fgh를
    # 실제로 엮은 그래프가 - 처음부터 끝까지 도는지 별도로 검증한다.
    print("  -- 추가 검증: fake 객체로 notebook 05의 build_graph를 직접 돌려본다 --")
    ok5 = True
    try:
        sys.path.insert(0, "..")
        from langchain_core.documents import Document

        class FakeStructuredLLM:
            def __init__(self, output):
                self.output = output
            def invoke(self, prompt):
                return self.output

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        from src.schemas import (
            DomainCriteria, DomainEval, MarketEval, ResearchResult,
            StakeholderEval, Synthesis, TechResearch, TechStatus, TRLAssessment,
        )

        class FakeFullLLM:
            _n = {"c": 0}
            def with_structured_output(self, schema_cls):
                if schema_cls is TechResearch:
                    rr = ResearchResult(overview="o", scope="s", limitations="l",
                                         trl_assessment=TRLAssessment(trl_ondevice=4, trl_global=6, evidence_status="found"))
                    return FakeStructuredLLM(TechResearch(turboquant=rr, infinigen=rr))
                if schema_cls is MarketEval:
                    ts = TechStatus(turboquant="정식", infinigen="실험")
                    return FakeStructuredLLM(MarketEval(market_size_growth="추정 갈림", adoption_status=ts,
                                                          ecosystem_support=ts, standardization="있음", label="조건 의존"))
                if schema_cls is StakeholderEval:
                    ts = TechStatus(turboquant="채택했다고 말함", infinigen="조건부")
                    return FakeStructuredLLM(StakeholderEval(competing_camp_reaction=ts, developer_adoption=ts,
                                                               investor_coverage=ts, label="조건 의존"))
                if schema_cls is DomainEval:
                    dc = DomainCriteria(
                        memory_budget=TechStatus(turboquant="들어간다 (1GB)", infinigen="넘는다 (2GB)"),
                        accuracy=TechStatus(turboquant="나눠 보고 (-2%p)", infinigen="뭉쳐 보고 (-4%p)"),
                        latency=TechStatus(turboquant="이 조건 실측 (40ms)", infinigen="다른 조건 실측 (30ms)"),
                        power_thermal=TechStatus(turboquant="실측 있음 (3W)", infinigen="근거 없음"),
                    )
                    return FakeStructuredLLM(DomainEval(criteria=dc, reversal_detected=True,
                                                          reversal_criteria="메모리", label="조건 의존"))
                if schema_cls is Synthesis:
                    return FakeStructuredLLM(Synthesis(
                        labels={"market": "조건 의존", "stakeholder": "조건 의존", "domain": "조건 의존", "trl": "판단보류"},
                        conflicts=[], reasoning="가짜",
                    ))
                raise ValueError(schema_cls)

            def invoke(self, prompt):
                self._n["c"] += 1
                if self._n["c"] == 1:
                    return FakeMessage("# SUMMARY\n내용\n# 시장\n내용\n# 이해관계자\n내용\n# REFERENCE\n내용")
                return FakeMessage("# SUMMARY\n내용\n# 시장\n내용\n# 이해관계자\n내용\n# 도메인\n내용\n# REFERENCE\n내용")

        class FakeRetriever2:
            def invoke(self, query):
                return [Document(page_content=f"가짜 문서 for {query}", metadata={"source_name": "fake.pdf"})]

        class FakeWebSearchTool2:
            def invoke(self, args):
                return f"가짜 검색: {args['query']}"

        ns5["llm"] = FakeFullLLM()
        ns5["tech_retriever"] = FakeRetriever2()
        ns5["domain_retriever"] = FakeRetriever2()
        ns5["web_search_tool"] = FakeWebSearchTool2()

        # cell 10 (build_graph 정의 + 호출)과 cell 12 (mermaid)를 fake로 재실행
        nb5 = nbformat.read("05_full_graph_run.ipynb", as_version=4)
        code_cells = [c for c in nb5.cells if c.cell_type == "code"]
        exec(compile(code_cells[5].source, "<05::build_graph-fake>", "exec"), ns5)  # build_graph 정의+호출
        exec(compile(code_cells[6].source, "<05::mermaid-fake>", "exec"), ns5)  # mermaid

        real_result = ns5["graph"].invoke({}, config={"recursion_limit": 40})
        assert real_result["validation_result"]["is_valid"] is True
        assert "TurboQuant" in real_result["tech_research"]
        print("  notebook 05의 실제 build_graph, fake 데이터로 A~H 전체 실행 성공")
        print("  최종 validation_result:", real_result["validation_result"])
    except Exception:
        print("  [FAIL] notebook 05 build_graph 직접 검증 실패")
        traceback.print_exc()
        ok5 = False

    all_ok = all_ok and ok5
    print("  OK" if ok5 else "  FAILED")

    print()
    print("=== 전체 결과 ===")
    print("모두 통과" if all_ok else "실패 있음 - 위 로그 확인")
    sys.exit(0 if all_ok else 1)
