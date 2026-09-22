"""
3-3절 State Schema.
LangGraph는 State 전체를 모든 노드에 전달하지만, 각 노드는 자기 담당 키만
읽고 쓴다 (3-1절 "State 분리 키" 원칙).

근거(references)도 관점별로 키를 나눴다 - B/C/D/E가 하나의 공유 리스트에
동시에 append하던 이전 버전과 달리, 지금은 각 키를 쓰는 노드가 정확히
1개씩이라 reducer가 필요 없다(동시 쓰기 충돌 자체가 발생하지 않음). REFERENCE
챕터에 쓸 통합 목록은 G가 네 리스트를 단순히 이어붙여서 만든다.
"""
from typing import TypedDict


class Reference(TypedDict, total=False):
    source: str       # 문서명 또는 URL
    detail: str        # 인용한 내용 요약


class GraphState(TypedDict, total=False):
    # A. 기술 선정
    selected_technologies: dict
    target_domain: str

    # B. 기술조사 (단독 작성 - reducer 불필요)
    tech_research: dict  # schemas.TechResearch 구조
    tech_references: list[Reference]

    # C. 시장성 평가 (단독 작성 - reducer 불필요)
    market_eval: dict  # schemas.MarketEval 구조
    market_references: list[Reference]

    # D. 이해관계자 평가 (단독 작성 - reducer 불필요)
    stakeholder_eval: dict  # schemas.StakeholderEval 구조
    stakeholder_references: list[Reference]

    # E. 도메인 평가 (단독 작성 - reducer 불필요)
    domain_eval: dict  # schemas.DomainEval 구조 (criteria: DomainCriteria 하나 + reversal_* 서술)
    domain_references: list[Reference]

    # F. 평가 종합
    synthesis: dict

    # G. 보고서 생성
    final_report: str

    # H. 보고서 검증
    validation_result: dict
    retry_count: int
