"""워커 공통 실행 껍데기 (Orchestrator-Workers).

C/D/E 세 워커가 같은 봉투(WorkerResult)로 결과를 돌려주게 만드는 자리다.
노드별 수집·판정 로직은 nodes_cd.py / nodes_e.py가 갖고, 여기엔 재시도·제외·
공백 탐지처럼 세 워커가 똑같이 필요로 하는 것만 둔다.

핵심 규칙 하나: 워커는 예외를 밖으로 던지지 않는다. 하나가 raise하면 dynamic
fan-out 전체가 멈춘다. 실패는 status로 알리고, 재작업이냐 제외냐는
Orchestrator가 정한다 - 여기서는 신호만 만든다.
"""
from __future__ import annotations

import datetime as _dt
import re
import time
from typing import Any, Callable, NamedTuple

from src.schemas import (
    REF_DETAIL_MAX,
    EvidenceGap,
    SourceAttempt,
    SourceKind,
    WorkerRef,
    WorkerResult,
)

NO_EVIDENCE = "근거 없음"
TECH_KEYS = (("turboquant", "TurboQuant"), ("infinigen", "InfiniGen"))

# 단위가 붙은 수치만 '측정값'으로 센다. 그냥 숫자로 보면 작동점 라벨("16K 배치4")에
# 걸려서, 두 작동점 다 못 찾은 칸까지 부분 성공으로 오분류된다.
MEASUREMENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|MB|KB|ms|W\b|%p|%|bit|비트)", re.IGNORECASE)


class WorkerOutput(NamedTuple):
    """워커 본체가 run_worker에 돌려주는 것. 봉투에 담기기 직전 상태."""
    payload: dict
    references: list[WorkerRef]
    gaps: list[EvidenceGap]
    sources: list[SourceAttempt]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def attempt_no(state: dict, agent: str) -> int:
    """이 agent의 몇 번째 디스패치인지. merge_results가 append-only라
    worker_results에 남은 같은 agent 기록 수 + 1이 곧 시도 번호다."""
    prior = sum(1 for r in state.get("worker_results") or [] if r.get("agent") == agent)
    return prior + 1


def collect_source(
    kind: SourceKind,
    fn: Callable[[], tuple[Any, int]],
    *,
    empty: Any,
    max_attempts: int = 2,
) -> tuple[Any, SourceAttempt]:
    """소스 하나를 재시도하며 수집한다. fn은 (값, 건수)를 돌려준다.

    끝내 실패해도 예외를 올리지 않고 empty를 준다. 웹이 죽어도 논문 근거로
    부분 판정은 나와야 하기 때문 - 제외는 소스 단위이지 워커 단위가 아니다.
    """
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            value, count = fn()
            return value, SourceAttempt(
                kind=kind,
                status="ok" if attempt == 1 else "retried",
                attempts=attempt,
                doc_count=count,
            )
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_attempts:
                time.sleep(1.5 * attempt)
    return empty, SourceAttempt(
        kind=kind, status="excluded", attempts=max_attempts, error=last_error[:200]
    )


def to_refs(items: list[dict], kind: SourceKind, *, limit: int | None = None) -> list[WorkerRef]:
    """{source, detail} 목록을 WorkerRef로 바꾸면서 detail을 자르고 개수를 막는다.

    둘 다 지속성 비용 때문이다. 이 리스트는 reducer로 State에 누적돼 체크포인트
    마다 저장된다. RAG는 청크를 100건 넘게 물어오는데 그걸 통째로 실으면 State가
    실행마다 수십 KB씩 분다. 원문은 프롬프트로만 흘려보내고 State에는 출처와
    짧은 요약만 남긴다.
    """
    refs = [
        WorkerRef(
            kind=kind,
            source=str(item.get("source", "unknown")),
            detail=str(item.get("detail", ""))[:REF_DETAIL_MAX],
        )
        for item in items
    ]
    return refs[:limit] if limit else refs


def web_search_tally(tool: Any, queries: list[str]) -> tuple[tuple[str, list[str]], int]:
    """쿼리를 하나씩 던지고 성공한 것만 모아 (본문, 성공쿼리목록), 성공건수를 준다.

    node_utils.run_web_search를 안 쓰는 이유: 그쪽은 쿼리별 실패를
    "[q] 검색 실패: ..." 문자열로 바꿔 넣고 정상 반환한다. 한 건 실패로 나머지를
    버리지 않으려는 설계라 그 자체는 맞지만, 그대로 쓰면 '전부 실패'와 '전부 성공'이
    호출부에 똑같이 보여서 sources에 ok가 찍힌다 - 폴백 기록이 거짓말을 한다.

    여기서는 전멸일 때만 예외를 올려 collect_source가 excluded를 찍게 하고,
    실패한 쿼리의 에러 문자열은 프롬프트에서 뺀다. 그건 근거가 아니다.
    """
    kept_blocks, kept_queries = [], []
    for q in queries:
        try:
            kept_blocks.append(f"[{q}] {tool.invoke({'query': q})}")
            kept_queries.append(q)
        except Exception:  # noqa: BLE001
            continue
    if not kept_queries:
        raise RuntimeError(f"웹 검색 {len(queries)}건 전부 실패")
    return ("\n".join(kept_blocks), kept_queries), len(kept_queries)


def split_web_references(search_results: str, queries: list[str]) -> list[dict]:
    """run_web_search 가 합쳐 준 문자열을 쿼리 단위로 되쪼개 Reference 목록을 만든다.

    검색 결과 전체를 한 건으로 뭉치면 REFERENCE 장에 URL 이 한 줄도 안 남는다.
    쿼리마다 한 건으로 나누고 본문에서 URL 을 뽑아 함께 싣는다.
    """
    import re

    blocks: dict[str, list[str]] = {}
    current = None
    for line in search_results.split("\n"):
        m = re.match(r"^\[(.+?)\]\s?(.*)$", line)
        if m and m.group(1) in queries:
            current = m.group(1)
            blocks[current] = [m.group(2)]
        elif current is not None:
            blocks[current].append(line)

    refs = []
    for q in queries:
        body = "\n".join(blocks.get(q, []))
        urls = [u.rstrip(".,;:)}'\"") for u in re.findall(r"https?://\S+", body)]
        seen: list[str] = []
        for u in urls:
            if u not in seen:
                seen.append(u)
        refs.append(
            {
                "source": f"web_search: {q}",
                "detail": (("URL: " + " | ".join(seen[:3]) + " / ") if seen else "URL 없음 / ")
                + body.strip()[:300],
            }
        )
    return refs


def find_evidence_gaps(payload: dict, searched: list[str]) -> list[EvidenceGap]:
    """구조화 출력에서 '근거 없음'인 칸을 항목x기술 단위로 추린다.

    TechStatus 패턴({turboquant, infinigen})을 만나면 기술별로 쪼개고, 단일
    문자열 필드면 '공통'으로 묶는다. 세 워커의 스키마가 전부 이 두 모양이라
    한 함수로 덮인다.

    notes처럼 긴 서술 필드가 본문에 '근거 없음'을 포함하는 경우를 세지 않으려고
    평문 필드는 완전일치로만 본다. TechStatus 값은 "근거 없음 (찾아본 범위: ...)"
    같은 꼬리가 붙으므로 부분일치로 본다.

    한 칸이 작동점별로 갈리는 경우가 실제로 나온다 - "들어간다 (16K 배치4: 140MB;
    32K 배치1: 근거 없음)"처럼 절반만 찾은 값이다. 수치가 같이 들어 있으면
    partial로 표시한다. 재디스패치 표적으로는 똑같이 유효하지만 품질 평가가
    '근거 전무'로 읽으면 안 되기 때문.
    """
    gaps: list[EvidenceGap] = []

    def walk(node: Any, field: str) -> None:
        if isinstance(node, dict):
            if all(key in node for key, _ in TECH_KEYS):
                for key, tech in TECH_KEYS:
                    value = str(node[key])
                    if NO_EVIDENCE in value:
                        gaps.append(EvidenceGap(
                            field=field, tech=tech, searched=searched,
                            partial=bool(MEASUREMENT_RE.search(value)),
                        ))
                return
            for key, value in node.items():
                walk(value, key)
        elif isinstance(node, str) and node.strip() == NO_EVIDENCE:
            gaps.append(EvidenceGap(field=field, tech="공통", searched=searched))

    walk(payload, "")
    return gaps


def run_worker(agent: str, fn: Callable[[dict], WorkerOutput], state: dict) -> dict:
    """워커 1개를 실행해 State 패치(worker_results, 실패 시 errors)를 돌려준다.
    node_utils.wrap_worker의 자리를 대신하되 sources·evidence_gaps를 함께 싣는다.

    워커 단위 재시도를 여기서 걸지 않는 이유: 그건 Orchestrator의 일이다.
    Synthesizer가 plan의 해당 태스크를 failed로 돌리면 Orchestrator가 그것만
    재디스패치한다(config.MAX_RETRY_ORCH). 여기서 또 돌리면 재시도가 두 겹이
    되고 트레이스에서 어느 층이 돈 건지 안 보인다. 소스 단위 재시도
    (collect_source)만 워커 안에 둔다 - 그건 Tavily 한 번 흔들린 것을 전면
    재디스패치로 키우지 않으려는, 층이 다른 장치다.

    status는 ok/failed 두 값뿐이다. '근거를 찾았는가'는 evidence_gaps가 따로
    말한다 - 합치면 get_worker_output이 payload를 통째로 버린다.
    """
    started = time.monotonic()
    attempt = attempt_no(state, agent)
    ts = _dt.datetime.now().isoformat(timespec="seconds")

    try:
        out = fn(state)
        result = WorkerResult(
            agent=agent,
            output=out.payload,
            references=out.references,
            status="ok",
            ts=ts,
            attempt=attempt,
            elapsed_ms=_elapsed_ms(started),
            sources=out.sources,
            evidence_gaps=out.gaps,
        )
        return {"worker_results": [result.model_dump()]}
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        result = WorkerResult(
            agent=agent,
            output={},
            status="failed",
            ts=ts,
            attempt=attempt,
            error=error,
            elapsed_ms=_elapsed_ms(started),
        )
        # errors까지 같이 쓰는 이유: state.py가 관측성 계층을 외부 트레이스가 아니라
        # 이 키에 두기로 했고(보고서 한계점 장과 디버깅이 여기서 읽는다),
        # node_utils.wrap_worker도 같은 모양으로 쓴다. 한쪽만 빼면 비는 자리가 생긴다.
        return {"worker_results": [result.model_dump()], "errors": [{"agent": agent, "error": error, "ts": ts}]}
