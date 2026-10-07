"""
여러 에이전트 노트북에서 공통으로 쓰는 작은 헬퍼들.
그래프 로직이 아니라 순수 가공이라 노트북마다 중복하지 않고 여기 하나로 둔다.

wrap_worker/latest_worker_result/get_worker_output/get_worker_references 네 개는
Orchestrator-Workers 전환(state.py의 WorkerResult/merge_results 참고)에 맞춰
C/D/E 노트북(02, 03)과 F/G 노트북(04 -> nodes_fg.py)이 공통으로 써야 하는 계약이다.
각 워커 함수는 (output: dict, references: list) 튜플만 리턴하면 되고, 나머지
agent 태그·status·ts 붙이기와 예외 흡수는 wrap_worker가 대신한다 - 워커 하나가
죽어도 그래프 전체가 멈추지 않고 "failed"로 기록된 채 Synthesizer까지 넘어간다
(가이드 B절 "일부 worker 작업 실패 시 재시도/제외" 요건).
"""
import datetime as _dt


def wrap_worker(agent: str):
    """C/D/E 노드 함수를 감싸 WorkerResult(state.py) 모양으로 통일한다.

    사용 예 (nodes_cd.py):
        @wrap_worker("C")
        def node_c_market_eval(state):
            ...
            return result.model_dump(), split_web_references(search_results, queries)
    """
    def decorator(fn):
        def wrapped(state):
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            try:
                output, references = fn(state)
                return {
                    "worker_results": [
                        {"agent": agent, "output": output, "references": references,
                         "status": "ok", "ts": ts}
                    ]
                }
            except Exception as e:  # noqa: BLE001 - 워커 실패를 그래프 중단이 아니라 상태로 흡수
                return {
                    "worker_results": [
                        {"agent": agent, "output": {}, "references": [], "status": "failed", "ts": ts}
                    ],
                    "errors": [{"agent": agent, "error": str(e), "ts": ts}],
                }
        return wrapped
    return decorator


def latest_worker_result(state: dict, agent: str) -> dict | None:
    """worker_results에서 해당 agent의 가장 최근 시도를 찾는다.
    merge_results가 append-only라 재시도 시 같은 agent가 두 번 등장할 수 있어
    마지막 것을 최신으로 본다."""
    matches = [r for r in state.get("worker_results", []) if r["agent"] == agent]
    return matches[-1] if matches else None


FAILED_WORKER_PLACEHOLDER = {
    "label": "판단보류",
    "notes": "워커 실행 실패(또는 재시도 상한 도달)로 근거를 확보하지 못함 - fallback 처리.",
}


def get_worker_output(state: dict, agent: str) -> dict:
    """Synthesizer(F)/보고서 생성(G)이 쓴다. 성공한 최신 결과가 없으면
    '판단보류' 플레이스홀더를 돌려줘 하류 노드가 KeyError로 죽지 않게 한다."""
    r = latest_worker_result(state, agent)
    return r["output"] if r and r["status"] == "ok" else FAILED_WORKER_PLACEHOLDER


def get_worker_references(state: dict, agent: str) -> list:
    r = latest_worker_result(state, agent)
    return r["references"] if r and r["status"] == "ok" else []



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
