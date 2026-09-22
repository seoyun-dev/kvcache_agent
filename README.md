# KV Cache 최적화 기술 다관점 평가 — MVP 코드

3번 그래프 설계(안) 그대로 짠 LangGraph 멀티에이전트 코드다.
`A -> B -> {C,D,E} -> F -> G <-> H` 여덟 노드, BM25+FAISS Hybrid(RRF), 리랭커·근거검증(B'/E') 없는 MVP 스코프.

**노트북 기반으로 짜서 분업이 가능하다** — 사람마다 노드 하나(또는 묶음) 담당 노트북을 열어서
독립적으로 프롬프트 고치고 테스트하고, 마지막 셀에서 그 결과를 `src/nodes_*.py`로 저장한다.
누가 뭘 고치든 다른 사람 파일은 안 건드리니 동시에 작업해도 충돌이 없다.

## 구조

```
notebooks/
  01_agent_B_tech_research.ipynb              B (기술조사, RAG) → src/nodes_b.py 생성
  02_agent_CD_market_stakeholder.ipynb         C, D (시장/이해관계자, WebSearch) → src/nodes_cd.py 생성
  03_agent_E_domain.ipynb                      E (도메인, RAG) → src/nodes_e.py 생성
  04_agent_FGH_synthesis_report_validate.ipynb F,G,H (종합/보고서/검증) → src/nodes_fgh.py 생성
  05_full_graph_run.ipynb                      위 네 파일을 모아 그래프 조립 + 실제 실행
src/
  config.py       문서 경로, 모델 이름, 선정 기술, 재시도 상한 등 전역 설정
  state.py        3-3절 State Schema (TypedDict)
  schemas.py      각 에이전트 구조화 출력 (Pydantic) - TechStatus 공통 패턴
  ingest.py       전처리+검색 파이프라인 (3-4절 mermaid)
  prompts.py      2절 판정 기준을 옮긴 프롬프트
  node_utils.py   여러 노트북이 같이 쓰는 작은 헬퍼 2개 (웹검색 실행, tech_research 요약)
  nodes_b.py      노트북 01이 생성 (직접 고치지 말 것)
  nodes_cd.py     노트북 02가 생성
  nodes_e.py      노트북 03이 생성
  nodes_fgh.py    노트북 04가 생성
  graph.py        3-4절 메인 그래프 배선 - 위 4개 파일을 import해서 조립
main.py           노트북 없이 한 번에 끝까지 돌리는 진입점 (05번 노트북과 동일한 일)
tests/
  test_graph_wiring.py   가짜 LLM으로 그래프 배선만 검증 (API 키 불필요)
scripts/
  download_papers.sh     5개 PDF 다운로드 헬퍼
validate_notebooks.py    다섯 노트북을 한 번에 다시 실행해서 안 깨졌는지 확인하는 스크립트
```

## 분업 추천

| 노트북 | 뭘 담당하나 | 잘 맞는 사람 |
|---|---|---|
| 01 (B, 기술조사) | RAG 리트리버 + TRL 판정 프롬프트 | RAG 파이프라인 이해한 사람 |
| 02 (C, D 시장/이해관계자) | 웹서치 프롬프트 2개, 구조 단순 | 프롬프트 다듬는 데 익숙한 사람 |
| 03 (E, 도메인) | 01과 같은 RAG 패턴 재사용 | 01 담당자와 짝, 또는 그 다음으로 |
| 04 (F, G, H) | 종합/보고서/규칙기반 검증, 셋이 이어짐 | 전체 그림 잡고 있는 사람 |
| 05 (조립) | 넷 다 끝난 뒤 import해서 실행 | 아무나, 제일 먼저 끝난 사람 |

**각 노트북은 완전히 독립적으로 실행 가능하다.** 01~04 아무 순서로나, 동시에 열어서 작업해도 된다.
05번만 넷이 다 저장을 마친 뒤에 돌아간다(정확히는 05번 셀에서 그 시점까지 저장된 파일을 그대로 가져다 씀 — 넷 중 하나가 아직 저장 전이면 import 에러가 나면서 어느 파일이 없는지 바로 알려준다).

**노트북을 고쳤으면 그 노트북 맨 마지막 "파일로 저장" 셀을 다시 실행해야 한다.** 안 그러면 05번(또는 `main.py`)은 예전 버전을 계속 쓴다.

## 이미 검증된 것 (이 코드를 짠 샌드박스에서 실제로 돌려봄)

- `pip install -r requirements.txt` 기준 모든 import 정상 동작 확인
  (langgraph 1.2.12, langchain 1.4.2 — 실습 코드와 같은 최신 패턴:
  `init_chat_model`, `langchain_classic.retrievers.EnsembleRetriever` 등)
- **노트북 5개 전부**: `python validate_notebooks.py`로 각 노트북의 셀을 실제로 실행해서 확인.
  실제 PDF·API 키가 필요한 셀만 건너뛰고(리트리버 구축, 실제 LLM 호출), 나머지는 전부 돌려봄:
  - 01~04: 함수 정의 → 가짜 LLM/리트리버로 배선 테스트 → `src/nodes_*.py` 저장까지 성공
  - 05: 넷이 저장한 파일을 실제로 import해서 `build_graph()`로 조립 → 가짜 데이터로
    A→B→{C,D,E}→F→G→H 전체 실행까지 성공 (mermaid 출력도 3-4절 구조와 일치 확인)
- **그래프 배선**: `python -m tests.test_graph_wiring` 통과.
  - fan-out(B→C,D,E)·fan-in(C,D,E→F) 정상 동작
  - `tech_references`/`market_references`/`stakeholder_references`/`domain_references`가
    각 노드 전용 키로 분리되어 있어 reducer 없이도 충돌 없이 누적됨을 확인
  - H→G 재시도 루프가 실제로 챕터 누락을 잡아서 재생성시킴, 상한(2회) 도달 시 `forced_pass=True`로 강제 종료
- 청킹 로직: TurboQuant 논문 실제 텍스트로 `RecursiveCharacterTextSplitter` 동작 확인
- BM25+FAISS+RRF 하이브리드 검색: 가짜 임베딩으로 `EnsembleRetriever` 배선 확인
- PDF 파일이 없을 때 에러 메시지가 명확한지 확인 (`data/papers/`에 뭘 넣어야 하는지 바로 알려줌)

## 아직 못 돌려본 것 (이 샌드박스가 인터넷이 막혀 있어서)

1. **실제 PDF 다운로드** — arxiv.org 접근이 이 개발 환경에서 막혀 있음.
   `scripts/download_papers.sh`를 팀 로컬에서 실행해서 `data/papers/`를 채울 것.
2. **Qwen3-Embedding-0.6B 실제 다운로드·추론** — huggingface.co 접근이 막혀
   있어서 `HuggingFaceEmbeddings(...)`를 실제로 호출해본 적은 없음. import
   자체는 되고 API 사용법도 표준 패턴이라 크게 걱정할 부분은 아니지만, 처음
   실행할 때 모델(약 1.2GB) 다운로드로 몇 분 걸릴 수 있다.
3. **실제 OpenAI/Tavily API 호출** — 키가 없어서 구조화 출력(`with_structured_output`)이
   실제 모델 응답에서도 스키마대로 잘 나오는지는 못 봤다. 프롬프트가 스키마
   설명과 어긋나면 첫 실행에서 파싱 에러가 날 수 있음 — 이건 각자 담당 노트북의
   "5. 실제 LLM 테스트" 셀에서 바로 확인 가능하고, 보통 프롬프트 문구를 조금
   다듬는 정도로 해결됨. **에러가 나면 그 노드 담당자만 자기 노트북을 열어
   고치면 되지, 다른 사람 작업을 기다릴 필요가 없다** — 이게 노트북 분리 구조의 핵심 이점.
4. **PDFPlumberLoader가 실제 논문 PDF(2단 레이아웃 포함)를 얼마나 깔끔하게
   뽑는지** — 텍스트만 뽑아서 청킹하는 로직은 검증했지만, 실제 바이너리 PDF를
   태워본 적은 없다.

## 실행 순서

```bash
cd kv-cache-agent
pip install -r requirements.txt --break-system-packages   # 필요시
cp .env.example .env                                       # OPENAI_API_KEY, TAVILY_API_KEY 채우기
bash scripts/download_papers.sh                             # data/papers/ 채우기
```

그다음 둘 중 하나:

**(A) 노트북으로, 분업해서** — 각자 01~04번 중 담당 노트북을 열어서 위에서 아래로 순서대로 실행(Run All).
"4. 배선 테스트"까지는 API 키 없이도 확인 가능하다. "5. 실제 LLM 테스트"부터는 `.env`가 채워져 있어야 실제로 돈다.
마지막 "파일로 저장" 셀까지 실행하면 `src/nodes_*.py`가 만들어진다(또는 갱신된다).
넷 다 끝나면 `05_full_graph_run.ipynb`를 열어서 위에서 아래로 실행.

**(B) 한 번에** — 노트북 없이 터미널에서:
```bash
python validate_notebooks.py    # 배선 재확인 (선택, API 키 불필요)
python main.py                   # 실제 실행 (이미 만들어진 src/nodes_*.py를 그대로 씀)
```

성공하면 `output/final_report.md`에 보고서가 생성된다.

## 시간 없을 때 자를 수 있는 곳

- E(도메인) 노드를 건너뛰면 그래프가 더 빨리 돈다 — `05_full_graph_run.ipynb`(또는 `main.py`)의
  그래프 조립 부분에서 `g.add_node("E", ...)`와 `g.add_edge("B","E")`/`g.add_edge("E","F")`를
  주석 처리. F의 `domain_eval` 참조는 `.get(..., {})` 기본값이 이미 안전하게 처리하므로 에러는 안 남.
- `config.MAX_RETRY_H = 0`으로 낮추면 H가 한 번 실패해도 바로 END로 감
  (재시도 루프 자체를 끄는 효과, 디버깅 사이클을 줄이고 싶을 때).

## 설계로 정한 것 — TRL 7~9는 B가 함부로 안 매긴다

B(기술조사)는 원문 논문만 RAG로 보고, TRL 7~9(공식 배포·상용 지원) 판정에만
아주 좁은 웹 검색 2건을 보조로 쓴다. 1~6단계는 논문이 자기 실험을 스스로
보고하니 판단 가능하지만, 7~9는 논문이 쓰인 "이후"에 일어나는 일이라 원문에
없을 수밖에 없다. C가 하는 "채택 현황" 검색과 질문 형태가 달라(C는 두 기술
비교, B는 기술별 개별 배포 사실) 중복이 아니고, 그래프 순서상 B가 C보다 먼저
끝나 C 결과를 재사용할 수도 없다. `01_agent_B_tech_research.ipynb`의
TECH_RESEARCH_PROMPT에 "원문·검색 결과에 배포 사실이 직접 없으면 7~9로
매기지 말고 not_found로 남겨라"를 명시했다.

## 시간 남으면 이어붙일 것 (3-2절 "시간 남으면 추가" 표 그대로)

- **B'/E' (LLM-as-Judge 근거검증)**: `01_agent_B_tech_research.ipynb`(또는 `03_...E_domain.ipynb`)에
  `make_node_b_verify(llm)` 같은 함수를 같은 패턴으로 추가하고 저장한 뒤,
  `src/graph.py`에서 B→B'→{C,D,E} 조건부 엣지로 바꾸면 된다.
  `route_after_h`와 똑같은 패턴(조건부 함수 + `add_conditional_edges`)을 재사용하면 됨.
- **리랭커(`bge-reranker-v2-m3`)**: `src/ingest.py`의 `build_hybrid_retriever`에서
  `EnsembleRetriever` 결과를 받아 CrossEncoder로 한 번 더 정렬하는 단계만
  추가하면 됨. 리트리버 인터페이스(`.invoke(query) -> list[Document]`)는
  그대로 유지하면서 감싸면(wrapper) 01·03 노트북은 안 건드려도 된다.
- **QJL, FlexGen 문서 추가**: `src/config.py`의 `TECH_PAPERS`에 두 줄만 추가.
