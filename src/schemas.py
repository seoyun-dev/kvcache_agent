"""
각 에이전트의 구조화 출력 스키마 (Pydantic).
2절 C절(평가 관점 및 기준)에서 가이드가 명시한 판정 항목을 그대로 필드로
옮긴다. TechStatus로 "두 기술 각각의 상태를 나란히 적는" 패턴을 공통화해서,
스키마 자체가 "어느 한쪽 편을 드는" 출력을 구조적으로 못 내게 만든다
(자유 텍스트로 몰아 쓰던 이전 버전 대비).
"""
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

Label = Literal["압축 유리 조건", "확장 유리 조건", "조건 의존", "차이 없음", "판단보류"]
EvidenceStatus = Literal["found", "not_found", "out_of_scope"]

T = TypeVar("T")


class TechStatus(BaseModel, Generic[T]):
    """한 판정 항목에 대해 TurboQuant/InfiniGen 각각의 상태를 나란히 기록.
    label 같은 단일 승자 필드가 없다 - 두 값을 나란히 두는 것 자체가
    "비교평가, 편들지 않기" 원칙을 스키마 레벨에서 강제한다.

    제네릭이라 두 가지로 쓰인다:
    - TechStatus[Literal["정식", ...]] 처럼 순수 카테고리만 필요한 자리
      (시장성 adoption_status 등)는 Pydantic이 옵션 밖 값을 바로 거부한다.
    - TechStatus[str] 처럼 카테고리 + 실측 수치를 같이 적어야 하는 자리
      (도메인 memory_budget 등, 예: "넘는다 (1.8GB)")는 자유 문자열을 허용한다.
    """
    turboquant: T
    infinigen: T


# ---------- B. 기술조사 (TRL은 여기 포함 - "기술 개요" 챕터에 들어감,
#             가이드 참고목차의 "관점별 평가"에는 TRL이 안 들어있음) ----------

class TRLAssessment(BaseModel):
    trl_ondevice: int = Field(ge=1, le=9, description="환경 게이트(TRL5) 적용한 온디바이스 기준 도달 단계")
    trl_global: int = Field(ge=1, le=9, description="환경 게이트 없이 판단한 전역 기준 도달 단계")
    evidence_status: EvidenceStatus
    note: str = Field(default="", description="온디바이스/전역 값이 다르면 그 차이를 설명")


class ResearchResult(BaseModel):
    overview: str = Field(description="기술 핵심 개요 2~3문장")
    scope: str = Field(description="실험에 쓰인 모델·데이터셋·환경 등 적용 범위")
    limitations: str = Field(description="논문이 스스로 밝힌 한계")
    trl_assessment: TRLAssessment


class TechResearch(BaseModel):
    turboquant: ResearchResult
    infinigen: ResearchResult


# ---------- C. 시장성 평가 ----------
# 가이드 C절: 시장 규모·성장성 / 상용화·채택 현황 / 생태계 지지

class MarketEval(BaseModel):
    market_size_growth: Literal["추정 갈림", "추정 모임", "근거 없음"] = Field(
        description="온디바이스 AI 시장 자체의 규모·성장률 추정 - 두 기술 공통 배경이라 기술별로 안 나눔"
    )
    adoption_status: TechStatus[Literal["정식", "실험", "예고만", "근거 없음"]] = Field(
        description=(
            "상용화/채택 현황. 벤더 공식 발표·릴리스노트만 근거로 인정한다. "
            "정식=기본 켜짐, 실험=preview이거나 기본값 꺼짐, 예고만=발표에만 "
            "있고 배포판에 없음. 서버용/온디바이스용이 다르면 notes에 둘 다 "
            "적고 이 필드값은 온디바이스 기준으로 낸다."
        )
    )
    ecosystem_support: TechStatus[Literal["본류 병합", "옵션", "포크만", "근거 없음"]] = Field(
        description=(
            "생태계 지지. llama.cpp·MLX-VLM·vLLM 세 런타임을 각각 확인해 "
            "가장 높은 값을 이 필드에 대표로 넣고, 나머지 두 곳 상태는 "
            "notes에 괄호로 적는다."
        )
    )
    standardization: Literal["있음", "근거 없음"] = Field(
        description=(
            "표준화 동향 - 공통 배경. 여러 회사가 함께 참여하는 규격·벤치마크"
            "(JEDEC·MLPerf 등)에 이름이 올라야 '있음'. 개별 회사 백서·블로그, "
            "GitHub 스타·포크 수는 근거로 세지 않는다."
        )
    )
    label: Label = Field(
        description=(
            "adoption_status·ecosystem_support 두 항목 각각에 대해 어느 기술 "
            "쪽이 더 앞선 값인지(예: adoption_status는 '정식'>'실험'>'예고만'>"
            "'근거 없음' 순) 먼저 개별 판정한다. 두 항목이 같은 기술을 가리키면 "
            "그 방향을 label로, 서로 다른 기술을 가리키면 '조건 의존', 둘 다 "
            "동률이면 '차이 없음', 둘 다 '근거 없음'이면 '판단보류'로 한다. "
            "market_size_growth·standardization은 공통 배경이라 이 판정에 안 쓴다."
        )
    )
    notes: str = Field(
        default="",
        description=(
            "adoption_status·ecosystem_support 각 태그의 구체적 근거(출처·날짜·"
            "핵심 문장)를 기술별로 정리한다. adoption_status가 서버용과 "
            "온디바이스용이 갈리면 둘 다 여기 적는다. ecosystem_support의 "
            "대표값 외 나머지 두 런타임 상태도 여기 괄호로 적는다. label이 "
            "'조건 의존'이면 두 항목 중 어느 쪽이 어느 기술을 가리켰는지 "
            "반드시 여기 밝힌다. 근거를 못 찾았으면 그 항목은 '근거 없음'이라고만 "
            "쓰고 열세로 서술하지 않는다."
        ),
    )


# ---------- D. 이해관계자 평가 ----------
# 가이드 C절: 경쟁 기술 진영 / 도입 기업·개발자 / 투자 업계

class StakeholderEval(BaseModel):
    competing_camp_reaction: TechStatus[Literal["대응기술 냈다고 밝힘", "한계 지적", "언급만", "근거 없음"]] = Field(
        description="경쟁 진영 반응 - 상대 진영의 논문·제품이 이 기술을 어떻게 다뤘는가. 둘 이상 언급이면 더 강한 쪽 값을 고른다."
    )
    developer_adoption: TechStatus[Literal["채택했다고 말함", "조건부", "안 쓴다고 말함", "근거 없음"]] = Field(
        description=(
            "도입 기업·개발자 발화. 시장성의 세 런타임(llama.cpp·MLX-VLM·vLLM) 중 "
            "온디바이스 도메인이므로 llama.cpp를 대표로 본다. 구현자가 지목한 "
            "채택 장벽(있다면)을 notes에 같이 적는다."
        )
    )
    investor_coverage: TechStatus[Literal["있음", "근거 없음"]] = Field(
        description="투자 업계·애널리스트 언급 - 기술 이름을 직접 지목한 분석만 센다. 보도자료 전재·벤더 발표는 세지 않는다."
    )
    label: Label = Field(
        description=(
            "competing_camp_reaction·developer_adoption·investor_coverage 세 "
            "항목을 하나씩 놓고, 그 항목의 실제 근거 내용이 TurboQuant·InfiniGen "
            "중 어느 쪽에 더 유리한 신호인지 먼저 개별 판단한다(단순 카테고리 "
            "순서가 아니라 실제 문맥 기준 - 예: '한계 지적'을 받은 쪽이 그 항목에서 "
            "불리). 세 항목의 방향이 전부 같으면 그 방향을 label로, 방향이 "
            "갈리면 '조건 의존', 판단 가능한 항목이 없으면 '판단보류'."
        )
    )
    notes: str = Field(
        default="",
        description=(
            "TurboQuant·InfiniGen 각 기술당 지지 근거 1건 + 비판 근거 1건씩, "
            "총 4건을 주체와 원문 요지를 함께 적는다(3개 항목별로 각각 채우는 "
            "게 아니라 기술 단위로 묶어서, 어느 항목에서 나온 근거인지는 괄호로 "
            "표시). 그 기술에 대한 지지·비판 중 한쪽이 없으면 '반대 근거 없음"
            "(찾아본 범위: ...)'이라고 쓴다. label이 '조건 의존'이면 세 항목 "
            "중 어느 것이 어느 기술 쪽으로 판단됐는지도 여기 명시한다."
        ),
    )


# ---------- E. 도메인 평가 ----------
# 가이드 C절 + 팀 설계: 메모리 예산 / 정확도 / 지연 / 전력·발열.
# 작동점 A·B(팀이 자체 추가한 세부 조건, 가이드 필수 요구사항 아님) 구분은
# 구조화 필드를 두 번 반복하지 않고 reversal_criteria 서술로 흡수한다 -
# 16칸을 기계적으로 채우게 하는 것보다 "달라지면 왜 달라지는지"가 보고서에
# 더 유용하고, 구조화 출력 실패 위험도 줄어든다.

class DomainCriteria(BaseModel):
    memory_budget: TechStatus[str] = Field(
        description="'들어간다 (1.2GB)'/'넘는다 (1.8GB, 예산 초과 0.3GB)'/'근거 없음' - "
                    "카테고리 뒤 괄호에 실제 GB 수치를 반드시 같이 적는다"
    )
    accuracy: TechStatus[str] = Field(
        description="'나눠 보고 (NIAH -2%p, GSM8K -6%p)'/'한쪽만 보고 (GSM8K, -4%p)'/"
                    "'뭉쳐 보고 (-4%p)'/'근거 없음' - 카테고리 뒤 괄호에 원본 대비 손실폭(%p)을 "
                    "반드시 같이 적는다. 카테고리만 쓰고 숫자를 안 쓰면 안 된다"
    )
    latency: TechStatus[str] = Field(
        description="'이 조건 실측 (토큰당 45ms)'/'다른 조건 실측 (배치8에서 30ms)'/"
                    "'근거 없음' - 카테고리 뒤 괄호에 실제 ms 수치를 반드시 같이 적는다"
    )
    power_thermal: TechStatus[str] = Field(
        description="'실측 있음 (평균 3.2W, 10분 후 20% 스로틀링)'/'근거 없음' - "
                    "실측 있음이면 괄호에 W·% 수치를 반드시 같이 적는다"
    )


class DomainEval(BaseModel):
    criteria: DomainCriteria = Field(
        description=(
            "네 항목 대표 판정. 작동점 A(16K·배치4)·B(32K·배치1) 전반에 걸쳐 "
            "성립하는 값을 적되, 둘 사이에 차이가 있으면 더 근거가 많은 쪽을 "
            "적고 그 차이는 reversal_criteria에 서술한다."
        )
    )
    reversal_detected: bool = Field(description="작동점 A/B 사이에 같은 기술의 판정 방향이 뒤집히는 항목이 있는가")
    reversal_criteria: str = Field(
        default="",
        description=(
            "뒤집힌다면 어떤 항목이, A에서는 어땠고 B에서는 어떻게 달랐는지, "
            "어떤 기준(메모리 예산 등) 때문인지 구체적으로 서술한다. 이건 두 "
            "기술의 순위 비교가 아니라 한 기술 내부의 조건별 차이다."
        ),
    )
    label: Label
    notes: str = Field(
        default="",
        description=(
            "criteria의 각 판정마다 왜 그렇게 판단했는지 근거를 정리한다. 예: "
            "'메모리 예산 - TurboQuant: 압축 후 1.2GB로 예산(1.5GB) 안에 들어감"
            "(논문 Table 1). InfiniGen: 호스트 DRAM 오프로딩이 전제인데 기준 "
            "기기엔 별도 호스트 DRAM이 없어 논문 조건 자체가 성립 안 함.' "
            "항목마다 근거 없으면 그것도 적는다."
        ),
    )


# ---------- F. 평가 종합 ----------

class Synthesis(BaseModel):
    labels: dict[str, str] = Field(description="{'market':..., 'stakeholder':..., 'domain':..., 'trl':...}")
    conflicts: list[str] = Field(default_factory=list, description="관점 간 상충 지점 목록, 없으면 빈 리스트")
    reasoning: str = Field(
        description=(
            "네 관점이 한 방향으로 몰렸다면 그 이유, "
            "도메인 평가가 조건별로 갈렸다면 어떤 기준 때문인지를 서술. "
            "둘 다 해당 안 되면 빈 문자열 대신 '해당 없음'이라고 명시."
        )
    )


# ---------- H. 보고서 검증 (규칙 기반, LLM 미사용) ----------

class ValidationResult(BaseModel):
    is_valid: bool
    missing_items: list[str] = Field(default_factory=list)
    forced_pass: bool = False
