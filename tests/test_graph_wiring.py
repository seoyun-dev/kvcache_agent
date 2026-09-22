"""
API 키·실제 임베딩 모델 없이 그래프 배선(엣지, State 병합, H 루프)만 검증한다.
가짜 LLM/리트리버/웹서치 도구를 넣어서, 실제 LangChain 객체와 동일한
인터페이스(.invoke, .with_structured_output)만 흉내낸다.

실행: python -m tests.test_graph_wiring   (프로젝트 루트에서)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.graph import build_graph  # noqa: E402
from src.schemas import (  # noqa: E402
    DomainCriteria,
    DomainEval,
    MarketEval,
    ResearchResult,
    StakeholderEval,
    Synthesis,
    TechResearch,
    TechStatus,
    TRLAssessment,
)


class FakeStructuredLLM:
    """llm.with_structured_output(Schema).invoke(prompt) 흉내."""

    def __init__(self, schema_cls, fixed_output):
        self.schema_cls = schema_cls
        self.fixed_output = fixed_output

    def invoke(self, prompt):
        return self.fixed_output


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """3가지를 흉내낸다: .invoke() (일반 텍스트), .with_structured_output(Schema)."""

    _report_counter = {"n": 0}

    def with_structured_output(self, schema_cls):
        if schema_cls is TechResearch:
            rr = ResearchResult(
                overview="가짜 개요", scope="가짜 범위", limitations="가짜 한계",
                trl_assessment=TRLAssessment(trl_ondevice=4, trl_global=6, evidence_status="found"),
            )
            return FakeStructuredLLM(schema_cls, TechResearch(turboquant=rr, infinigen=rr))
        if schema_cls is MarketEval:
            ts = TechStatus(turboquant="정식", infinigen="실험")
            return FakeStructuredLLM(
                schema_cls,
                MarketEval(
                    market_size_growth="추정 갈림", adoption_status=ts, ecosystem_support=ts,
                    standardization="있음", label="조건 의존", notes="가짜",
                ),
            )
        if schema_cls is StakeholderEval:
            ts = TechStatus(turboquant="채택했다고 말함", infinigen="조건부")
            return FakeStructuredLLM(
                schema_cls,
                StakeholderEval(
                    competing_camp_reaction=ts, developer_adoption=ts, investor_coverage=ts,
                    label="조건 의존", notes="가짜",
                ),
            )
        if schema_cls is DomainEval:
            dc = DomainCriteria(
                memory_budget=TechStatus(turboquant="들어간다", infinigen="넘는다"),
                accuracy=TechStatus(turboquant="나눠 보고", infinigen="한쪽만 보고"),
                latency=TechStatus(turboquant="이 조건 실측", infinigen="다른 조건 실측"),
                power_thermal=TechStatus(turboquant="실측 있음", infinigen="근거 없음"),
            )
            return FakeStructuredLLM(
                schema_cls,
                DomainEval(
                    criteria=dc,
                    reversal_detected=True, reversal_criteria="메모리 예산",
                    label="조건 의존",
                    notes="메모리 예산 - TurboQuant: 가짜 근거. InfiniGen: 가짜 근거.",
                ),
            )
        if schema_cls is Synthesis:
            return FakeStructuredLLM(
                schema_cls,
                Synthesis(
                    labels={"market": "조건 의존", "stakeholder": "조건 의존",
                            "domain": "조건 의존", "trl": "판단보류"},
                    conflicts=["시장은 압축 쪽, 도메인은 확장 쪽을 일부 지지"],
                    reasoning="네 관점이 전부 같은 방향은 아니라 정렬 검사는 해당 없음. "
                              "도메인은 메모리 예산 기준으로 작동점 A/B 판정이 갈림.",
                ),
            )
        raise ValueError(f"예상 못 한 스키마: {schema_cls}")

    def invoke(self, prompt):
        # G(보고서 생성) 노드용 - 첫 호출은 REQUIRED_CHAPTERS 중 하나를 빠뜨려서
        # H -> G 재시도 루프가 실제로 도는지 검증하고, 두번째부터는 다 채운다.
        self._report_counter["n"] += 1
        if self._report_counter["n"] == 1:
            body = "# SUMMARY\n내용\n# 시장\n내용\n# 이해관계자\n내용\n# REFERENCE\n내용"
            # "도메인" 챕터를 일부러 빠뜨림
        else:
            body = "# SUMMARY\n내용\n# 시장\n내용\n# 이해관계자\n내용\n# 도메인\n내용\n# REFERENCE\n내용"
        return FakeMessage(body)


class FakeRetriever:
    def invoke(self, query):
        return [Document(page_content=f"가짜 문서 내용 for '{query}'", metadata={"source_name": "fake.pdf", "page": 1})]


class FakeWebSearchTool:
    def invoke(self, args):
        return f"가짜 웹 검색 결과: {args['query']}"


def run():
    llm = FakeLLM()
    graph = build_graph(
        llm=llm,
        tech_retriever=FakeRetriever(),
        domain_retriever=FakeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )

    result = graph.invoke({}, config={"recursion_limit": 40})

    # --- 검증 ---
    assert result["selected_technologies"]["SW"]["name"] == "TurboQuant", "A 노드 확인"
    assert result["target_domain"] == "OnDevice AI"
    assert "TurboQuant" in result["tech_research"], "B 노드 확인"
    assert result["market_eval"]["label"] == "조건 의존", "C 노드 확인"
    assert result["market_eval"]["adoption_status"]["turboquant"] == "정식", "TechStatus 중첩 구조 확인"
    assert result["stakeholder_eval"]["label"] == "조건 의존", "D 노드 확인"
    assert result["domain_eval"]["reversal_detected"] is True, "E 노드 확인"
    assert result["domain_eval"]["criteria"]["memory_budget"]["infinigen"] == "넘는다", "도메인 중첩 구조 확인"
    assert result["domain_eval"]["notes"], "domain_eval에 근거 서술(notes)이 채워지는지 확인"
    assert "conflicts" in result["synthesis"], "F 노드 확인"
    assert "도메인" in result["final_report"], "G가 H 피드백 받아 재생성했는지 확인"
    assert result["validation_result"]["is_valid"] is True, "H 최종 통과 확인"

    # references가 4개 키로 분리돼 각자 잘 채워지는지 (reducer 없이, 단독 작성)
    assert len(result["tech_references"]) == 3, f"B: RAG 2쿼리 + 웹서치 1건, got {len(result['tech_references'])}"
    assert len(result["market_references"]) == 1, "C 단독 작성 확인"
    assert len(result["stakeholder_references"]) == 1, "D 단독 작성 확인"
    assert len(result["domain_references"]) == 3, "E: RAG 3쿼리 확인"
    total_refs = (
        len(result["tech_references"]) + len(result["market_references"])
        + len(result["stakeholder_references"]) + len(result["domain_references"])
    )

    # retry_count가 실제로 1회 올라갔는지 (첫 보고서는 '도메인' 챕터 누락 -> 재시도)
    assert result["retry_count"] == 1, f"H->G 재시도 루프 확인, got {result['retry_count']}"

    print("모든 배선 테스트 통과.")
    print(f"  - references 총합(4키 분리): {total_refs}")
    print(f"  - retry_count: {result['retry_count']}")
    print(f"  - validation_result: {result['validation_result']}")


class AlwaysBadReportLLM(FakeLLM):
    """G가 몇 번을 재생성해도 계속 챕터가 빠진 보고서만 내놓는 경우 -
    H가 MAX_RETRY_H(2)를 넘기면 forced_pass=True로 강제 종료하는지 검증."""

    def invoke(self, prompt):
        return FakeMessage("# SUMMARY\n내용만 있고 다른 챕터는 계속 빠짐")


def run_forced_pass_scenario():
    llm = AlwaysBadReportLLM()
    graph = build_graph(
        llm=llm,
        tech_retriever=FakeRetriever(),
        domain_retriever=FakeRetriever(),
        web_search_tool=FakeWebSearchTool(),
    )
    result = graph.invoke({}, config={"recursion_limit": 40})

    assert result["validation_result"]["forced_pass"] is True, "상한 도달 시 강제 통과 확인"
    assert result["validation_result"]["is_valid"] is True, "forced_pass여도 is_valid=True로 END 라우팅"
    assert result["retry_count"] == 3, f"최초 1회 + 재시도 2회 = 3 확인, got {result['retry_count']}"
    assert len(result["validation_result"]["missing_items"]) > 0, "missing_items가 남아있어야 함(한계점 장 기재용)"

    print("forced_pass 시나리오 통과.")
    print(f"  - retry_count: {result['retry_count']}")
    print(f"  - validation_result: {result['validation_result']}")


if __name__ == "__main__":
    run()
    print()
    run_forced_pass_scenario()
