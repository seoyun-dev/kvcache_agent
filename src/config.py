"""
프로젝트 전역 설정.
- 문서 경로 (data/papers/ 밑에 실제 PDF를 넣어야 함 — 이 리포에는 안 들어있음)
- 모델 이름
- 청킹/재시도 파라미터
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
OUTPUT_DIR = BASE_DIR / "output"

# 1-2절 RAG 적용 대상 문서 (MVP: 5편, 93p)
# 파일명은 예시. 실제로는 아래 arxiv 링크에서 받아 이 이름으로 data/papers/ 밑에 저장할 것.
#   TurboQuant : https://arxiv.org/pdf/2504.19874
#   InfiniGen  : https://arxiv.org/pdf/2406.19707
#   On-Device 시스템평가 : https://arxiv.org/pdf/2505.15030
#   ELIB/MBU   : https://arxiv.org/pdf/2508.11269
#   LLM in a Flash : https://arxiv.org/pdf/2312.11514
TECH_PAPERS = {
    "TurboQuant": PAPERS_DIR / "turboquant.pdf",
    "InfiniGen": PAPERS_DIR / "infinigen.pdf",
}

DOMAIN_PAPERS = {
    "OnDeviceEval": PAPERS_DIR / "ondevice_eval.pdf",
    "ELIB_MBU": PAPERS_DIR / "elib_mbu.pdf",
    "LLMinFlash": PAPERS_DIR / "llm_in_flash.pdf",
}

# 1-3절 Embedding 모델
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# 3-2절 LLM (팀 실습 코드 관행: init_chat_model 사용)
LLM_MODEL = "gpt-4.1-mini"
LLM_PROVIDER = "openai"
LLM_TEMPERATURE = 0

# 3-1절 청킹 파라미터 (단일 계층, Parent-Child 아님)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# 검색 파라미터 — BM25+FAISS Hybrid(RRF), 리랭커 없음 (MVP)
RETRIEVER_TOP_K_EACH = 20  # BM25/Dense 각각 뽑는 개수
RETRIEVER_TOP_K_FINAL = 5  # RRF 병합 후 최종 사용 개수

# 3-1절 재시도 상한
MAX_RETRY_H = 2  # G<->H 보고서 검증 루프

# 3-2절 대상 기술
SELECTED_TECHNOLOGIES = {
    "SW": {
        "name": "TurboQuant",
        "camp": "압축(양자화)",
        "reason": (
            "정보이론적 하한에 근접하는 근사 최적 분포 왜곡률을 수학적으로 증명. "
            "재학습 없이 임의의 사전학습 모델에 적용 가능한 배포 단계 기법. "
            "ICLR 2026 채택, Apple MLX-VLM 실채택 등 현시점 화제성 뚜렷."
        ),
    },
    "HW": {
        "name": "InfiniGen",
        "camp": "메모리 오프로딩",
        "reason": (
            "다음 레이어 attention 패턴을 예측해 필요한 KV 항목만 선택적으로 프리페치. "
            "특수 하드웨어(CXL/PIM) 없이 소프트웨어 메커니즘만으로 host 메모리 오프로딩 "
            "문제를 해결 — HW 후보 중 유일하게 온디바이스 도메인 원리 논의가 가능."
        ),
    },
}
TARGET_DOMAIN = "OnDevice AI"

# 보고서 "분석 배경" 챕터용 고정 문단 (1-1절 내용 재사용).
# State 키로 만들지 않는 이유: 매 실행마다 새로 검색/생성할 이유가 없는
# 정적 배경 설명이라, config 상수로 두고 G가 그대로 갖다 쓰는 쪽이 더 싸다.
ANALYSIS_BACKGROUND = """2026년 들어 온디바이스 AI는 CES부터 WWDC까지 이어지는
5대 글로벌 컨퍼런스의 공통 핵심 의제로 자리잡았다. Apple은 파운데이션 모델
기반 개인 에이전트로 Siri를 재설계했고, Samsung은 연말까지 약 2억 대의
갤럭시 AI 폰에 실시간 번역·생성형 이미지 편집을 탑재하겠다고 발표했다.
IDC는 2025년 GenAI 스마트폰 출하량이 전체의 약 30%를 차지할 것으로 전망했다.

이 흐름의 근본 제약은 메모리다. 스마트폰·엣지 기기는 배터리·발열·제한된
RAM 안에서 LLM을 구동해야 하는데, 컨텍스트가 길어질수록 KV cache가 모델
가중치보다 더 큰 메모리를 요구하는 구조적 병목이 있다. 2026년 3월 Google이
TurboQuant를 ICLR 2026 채택 논문으로 발표한 다음날 삼성전자·SK하이닉스
주가가 하락했다는 보도가 나올 만큼, KV cache 압축 기술 하나가 실제 산업
이해관계에 영향을 미치는 단계에 와 있다. 본 보고서는 이 병목을 서로 다른
철학(압축 vs 메모리 계층 확장)으로 푸는 두 기술을 다관점에서 비교한다."""
