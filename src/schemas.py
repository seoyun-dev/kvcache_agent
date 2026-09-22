"""
각 에이전트의 구조화 출력 스키마 (Pydantic).
2절 C절(평가 관점 및 기준)에서 가이드가 명시한 판정 항목을 그대로 필드로
옮긴다. TechStatus로 "두 기술 각각의 상태를 나란히 적는" 패턴을 공통화해서,
스키마 자체가 "어느 한쪽 편을 드는" 출력을 구조적으로 못 내게 만든다
(자유 텍스트로 몰아 쓰던 이전 버전 대비).
"""
from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["압축 유리 조건", "확장 유리 조건", "조건 의존", "차이 없음", "판단보류"]
EvidenceStatus = Literal["found", "not_found", "out_of_scope"]


class TechStatus(BaseModel):
    """한 판정 항목에 대해 TurboQuant/InfiniGen 각각의 상태를 나란히 기록.
    label 같은 단일 승자 필드가 없다 - 두 값을 나란히 두는 것 자체가
    "비교평가, 편들지 않기" 원칙을 스키마 레벨에서 강제한다."""
    turboquant: str
    infinigen: str


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
    adoption_status: TechStatus = Field(
        description="상용화/채택 현황. 값: '정식'(기본 켜짐)/'실험'(preview)/'예고만'/'근거 없음'"
    )
    ecosystem_support: TechStatus = Field(
        description="생태계 지지(llama.cpp/MLX-VLM/vLLM 등). 값: '본류 병합'/'옵션'/'포크만'/'근거 없음'"
    )
    standardization: Literal["있음", "근거 없음"] = Field(description="표준화 동향 - 공통 배경")
    label: Label
    notes: str = Field(
        default="",
        description=(
            "adoption_status·ecosystem_support 각 태그의 구체적 근거(출처·날짜·"
            "핵심 문장)를 기술별로 정리한다. 근거를 못 찾았으면 그 항목은 "
            "'근거 없음'이라고만 쓰고 열세로 서술하지 않는다."
        ),
    )


# ---------- D. 이해관계자 평가 ----------
# 가이드 C절: 경쟁 기술 진영 / 도입 기업·개발자 / 투자 업계

class StakeholderEval(BaseModel):
    competing_camp_reaction: TechStatus = Field(
        description="경쟁 진영 반응. 값: '대응기술 냈다고 밝힘'/'한계 지적'/'언급만'/'근거 없음'"
    )
    developer_adoption: TechStatus = Field(
        description="도입 기업·개발자 발화. 값: '채택했다고 말함'/'조건부'/'안 쓴다고 말함'/'근거 없음'"
    )
    investor_coverage: TechStatus = Field(
        description="투자 업계·애널리스트 언급. 값: '있음'/'근거 없음'"
    )
    label: Label
    notes: str = Field(default="", description="지지·비판 근거 각 1건. 반대쪽 없으면 '반대 근거 없음(찾아본 범위: ...)'")


# ---------- E. 도메인 평가 ----------
# 가이드 C절 + 팀 설계: 메모리 예산 / 정확도 / 지연 / 전력·발열.
# 작동점 A·B(팀이 자체 추가한 세부 조건, 가이드 필수 요구사항 아님) 구분은
# 구조화 필드를 두 번 반복하지 않고 reversal_criteria 서술로 흡수한다 -
# 16칸을 기계적으로 채우게 하는 것보다 "달라지면 왜 달라지는지"가 보고서에
# 더 유용하고, 구조화 출력 실패 위험도 줄어든다.

class DomainCriteria(BaseModel):
    memory_budget: TechStatus = Field(
        description="'들어간다 (1.2GB)'/'넘는다 (1.8GB, 예산 초과 0.3GB)'/'근거 없음' - "
                    "카테고리 뒤 괄호에 실제 GB 수치를 반드시 같이 적는다"
    )
    accuracy: TechStatus = Field(
        description="'나눠 보고 (NIAH -2%p, GSM8K -6%p)'/'한쪽만 보고 (GSM8K, -4%p)'/"
                    "'뭉쳐 보고 (-4%p)'/'근거 없음' - 카테고리 뒤 괄호에 원본 대비 손실폭(%p)을 "
                    "반드시 같이 적는다. 카테고리만 쓰고 숫자를 안 쓰면 안 된다"
    )
    latency: TechStatus = Field(
        description="'이 조건 실측 (토큰당 45ms)'/'다른 조건 실측 (배치8에서 30ms)'/"
                    "'근거 없음' - 카테고리 뒤 괄호에 실제 ms 수치를 반드시 같이 적는다"
    )
    power_thermal: TechStatus = Field(
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
