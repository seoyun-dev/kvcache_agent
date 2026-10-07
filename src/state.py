"""
State Schema - Orchestrator-Workers 패턴, Layered State.

기존(고정 DAG) 버전은 market_eval/stakeholder_eval/domain_eval 처럼 노드마다
전용 키를 따로 둬서 "쓰는 노드가 정확히 1개"라 reducer가 필요 없었다. 지금은
C/D/E가 Orchestrator의 계획(plan)에 따라 동적으로 fan-out되고, 실패하면 같은
노드가 재시도로 "두 번째" 결과를 쓸 수 있어 더 이상 그 전제가 성립하지 않는다.
그래서 State를 두 층으로 나눈다.

  제어 (control)  : plan, orch_retry_count, eval_retry_count - 라우팅 결정에만 쓰고
                     보고서 본문에는 안 들어간다.
  페이로드 (payload): worker_results, synthesis, final_report, eval_result - 실제
                     내용. worker_results는 fan-out 동시쓰기가 생기는 유일한 키라
                     reducer(merge_results)를 둔다.

설계 근거(가이드 C절 7항목 대응)는 README "State Schema" 절에 정리했다 -
그 표의 항목 이름을 그대로 아래 필드 옆 주석에 달아 코드만 봐도 대응이
보이게 했다.
"""
from typing import Annotated, Literal, NotRequired, TypedDict
import operator


class SubTask(TypedDict):
    """Orchestrator가 수립하는 계획 한 칸. C/D/E 중 하나를 가리킨다.
    재개/복구: status만 보면 어디까지 끝났는지 알 수 있다 - 재진입 시 failed만
    pending으로 되돌려 그 태스크만 다시 Send한다(= 고정 3-way가 아닌 동적 fan-out)."""
    agent: Literal["C", "D", "E"]
    status: Literal["pending", "done", "failed"]
    reason: NotRequired[str]             # 관측성: 이 태스크를 왜 (다시) 띄웠나 - 트레이스에 그대로 보인다
    gap_fields: NotRequired[list[str]]   # 재조사 표적 - 워커가 이 칸만 겨냥한 검색어를 더한다
    reworked: NotRequired[bool]          # 종료 보장: 재조사는 태스크당 한 번


class WorkerResult(TypedDict):
    """C/D/E가 공통으로 리턴하는 결과 하나. 기존엔 market_eval/market_references처럼
    관점마다 키가 따로였는데, 여기서는 agent 태그로 구분되는 동일 구조 하나로
    합쳐서 Synthesizer가 획일적으로 집계할 수 있게 했다."""
    agent: str          # "C" | "D" | "E"
    output: dict         # schemas.MarketEval / StakeholderEval / DomainEval 의 model_dump()
    references: list     # [{"source":..., "detail":...}, ...]
    status: Literal["ok", "failed"]
    ts: str              # 상관(correlation): Synthesizer/Evaluator가 "어느 시도"인지 식별


def merge_results(a: list, b: list) -> list:
    """동시 처리: C/D/E가 같은 슈퍼스텝에서 병렬로 worker_results에 쓸 때 쓰는 reducer.
    append-only라 재시도로 같은 agent가 두 번 들어와도(1차 실패 + 2차 성공) 둘 다
    남는다 - "최신 시도가 무엇인지"는 node_utils.latest_worker_result가 뒤에서부터
    찾아 판단한다(지속성 비용: 워커 3개 고정이라 무한 증식하지 않는다)."""
    return a + b


class OrchestratorState(TypedDict, total=False):
    # 상관(correlation): LangSmith 실행(run_id)과 State를 잇는 키. main.py가
    # graph.invoke()에 config={"run_id": run_id}로 넘기는 값과 동일한 문자열을
    # 여기 같이 넣어둔다 - 아무 노드도 이 값을 읽거나 쓰지 않는다(그래서 다른
    # 담당자의 노드 코드는 안 건드려도 됨). 보고서/State만 보고도 LangSmith
    # 대시보드에서 해당 실행을 바로 찾아갈 수 있게 하기 위한 용도.
    trace_id: str

    # ---- 제어 (control) ----
    selected_technologies: dict
    target_domain: str
    plan: list[SubTask]
    orch_retry_count: int     # 종료 보장: Orchestrator<->Synthesizer 루프 상한 (config.MAX_RETRY_ORCH)
    eval_retry_count: int     # 종료 보장: G<->Evaluator 루프 상한 (config.MAX_RETRY_EVAL)

    # ---- 페이로드 (payload) ----
    tech_research: dict        # B 산출물 (고정 선행 노드라 워커 래핑 안 함)
    tech_references: list
    worker_results: Annotated[list[WorkerResult], merge_results]
    synthesis: dict            # Synthesizer(F) 산출물
    final_report: str          # G 산출물
    eval_result: dict          # Evaluator 산출물 - {"passed": bool, "missing_items": [...], "forced_pass": bool, ...}

    # 관측성: 사유를 담아 State 내부에 쌓는다(외부 트레이스가 아니라 바로 여기 담는
    # 이유는 route_after_synthesis가 재작업 여부를 이 값이 아니라 plan.status로
    # 판단하긴 하지만, "왜 실패했는지"는 보고서 한계점 장과 사람이 디버깅할 때
    # 여기서 바로 확인해야 하기 때문).
    errors: Annotated[list[dict], operator.add]
