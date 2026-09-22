"""
여러 에이전트 노트북에서 공통으로 쓰는 작은 헬퍼 두 개.
그래프 로직이 아니라 순수 문자열 가공이라 노트북마다 중복하지 않고 여기 하나로 둔다.
"""


def summarize_tech_research(tech_research: dict) -> str:
    """B의 결과를 C/D/E 프롬프트에 넣을 짧은 요약으로 변환."""
    if not tech_research:
        return "(아직 기술조사 결과 없음)"
    lines = []
    for name, r in tech_research.items():
        # overview 만 넘기면 B 가 원문에서 뽑은 실험 환경·수치·한계가 C/D/E 에
        # 도달하지 못한다. 도메인 평가(E)는 그 수치가 유일한 기술별 근거다.
        lines.append(f"- {name}")
        lines.append(f"  개요: {r.get('overview', '')}")
        if r.get("scope"):
            lines.append(f"  적용 범위(실험 환경·모델·데이터셋): {r['scope']}")
        if r.get("limitations"):
            lines.append(f"  논문이 밝힌 한계: {r['limitations']}")
    return "\n".join(lines)


def run_web_search(web_search_tool, queries: list[str]) -> str:
    """쿼리 여러 개를 순서대로 검색해 하나의 문자열로 합친다.
    하나가 실패해도 나머지는 계속 진행한다(그래프 전체가 멈추지 않게)."""
    results = []
    for q in queries:
        try:
            r = web_search_tool.invoke({"query": q})
            results.append(f"[{q}] {r}")
        except Exception as e:  # noqa: BLE001
            results.append(f"[{q}] 검색 실패: {e}")
    return "\n".join(results)
