"""
Orchestrator 노드 - 가이드 B절(Orchestrator-Workers Mandatory Items) 구현.

    - "서브 태스크 목록을 구조화된 형태로 State에 저장"      -> plan: list[SubTask]
    - "Workers는 계획이 수립된 이후에 결정 = Dynamic Fan-out" -> route_dynamic_fanout (Send)
    - "일부 worker 실패 시 재시도/제외 중 하나를 정해야 함"   -> Orchestrator<->Synthesizer 재진입 루프

Orchestrator는 두 가지 역할을 겸한다.
  1) 최초 진입: plan이 없으면 C/D/E 세 개짜리 초기 계획을 세운다.
  2) 재진입(Synthesizer가 되돌려보낸 경우): 1라운드 결과를 보고 다시 띄울 것만 고른다.
     - failed                -> 같은 태스크 재시도
     - done 인데 근거 공백   -> 빈 칸(evidence_gaps, partial 제외)만 겨냥한 재조사
     그래서 두 번째 Send 의 대상과 개수는 실행마다 다르다 - 실패·공백이 없으면 0건이고
     바로 G 로 간다. "동적 동작 실증(Orchestrator = dynamic fan-out)" 이 트레이스에
     보이는 지점이고, 각 태스크의 reason 에 왜 다시 띄웠는지가 남는다.
"""
from langgraph.types import Send

from . import config
from .node_utils import latest_worker_result


def hard_gap_fields(state, agent) -> list[str]:
    """이 워커의 최신 성공 결과에서 '근거 전무' 인 칸(partial 제외)의 항목명. 순서 유지·중복 제거."""
    r = latest_worker_result(state, agent)
    if not r or r["status"] != "ok":
        return []
    fields = []
    for g in r.get("evidence_gaps") or []:
        if not g.get("partial") and g.get("field") and g["field"] not in fields:
            fields.append(g["field"])
    return fields


def needs_rework(state, task) -> bool:
    """끝난 태스크인데 빈 칸이 남았고 아직 재조사를 안 했으면 True."""
    return task["status"] == "done" and not task.get("reworked") and bool(hard_gap_fields(state, task["agent"]))


def node_orchestrator(state) -> dict:
    plan = state.get("plan")

    if plan is None:
        # 최초 계획. 세 관점은 B 산출물에만 의존하고 서로 독립이라 한 번에 띄운다.
        # 두 번째 라운드부터는 결과를 보고 대상과 개수를 정한다(아래) - 그게 동적 fan-out 이다.
        return {"plan": [{"agent": a, "status": "pending", "reason": "초기 분해"} for a in ("C", "D", "E")]}

    # 재진입: 결과를 보고 다시 띄울 태스크를 고른다.
    #   failed            -> 같은 태스크 재시도 (fallback)
    #   done + 근거 공백  -> 빈 칸만 겨냥한 재조사 (태스크당 1회)
    #   done + 공백 없음  -> 그대로 둔다
    new_plan = []
    for task in plan:
        if task["status"] == "failed":
            new_plan.append({**task, "status": "pending", "reason": "실행 실패 재시도"})
        elif needs_rework(state, task):
            fields = hard_gap_fields(state, task["agent"])[: config.MAX_REWORK_FIELDS]
            new_plan.append({
                **task, "status": "pending", "reworked": True, "gap_fields": fields,
                "reason": f"근거 공백 재조사 ({', '.join(fields)})",
            })
        else:
            new_plan.append(task)
    return {
        "plan": new_plan,
        "orch_retry_count": state.get("orch_retry_count", 0) + 1,
    }


def route_dynamic_fanout(state):
    """Orchestrator 다음 조건부 엣지. pending 태스크만 Send 로 내보낸다.
    태스크 자체(재조사 표적 포함)를 같이 실어 보내 워커가 무엇을 겨냥할지 안다.
    1라운드는 3건, 2라운드는 실패·공백이 있던 것만 - 실행마다 대상과 개수가 달라진다."""
    pending = [t for t in state.get("plan", []) if t["status"] == "pending"]
    return [Send(task["agent"], {**state, "task": task}) for task in pending]


def route_after_synthesis(state):
    """Synthesizer 다음 조건부 엣지.
    실패 태스크나 근거 공백이 남은 태스크가 있고 라운드 상한(config.MAX_RETRY_ORCH)에
    아직 안 왔으면 Orchestrator 로 되돌아가 그것만 재디스패치한다.
    상한에 도달했거나 다 채워졌으면 G 로 - 그때도 실패로 남은 태스크는
    node_utils.get_worker_output 이 '판단보류' 로 채운다(제외 fallback)."""
    if state.get("orch_retry_count", 0) >= config.MAX_RETRY_ORCH:
        return "G"
    plan = state.get("plan", [])
    if any(t["status"] == "failed" or needs_rework(state, t) for t in plan):
        return "Orchestrator"
    return "G"
