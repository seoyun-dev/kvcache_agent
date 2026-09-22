"""
전처리 + 검색 파이프라인 (3-4절 mermaid 그대로).

전처리: PdfPlumber 텍스트 추출 -> RecursiveCharacterTextSplitter(단일 계층)
       -> Qwen3-Embedding-0.6B 임베딩 -> FAISS / BM25 인덱스 병렬 구축
검색  : BM25 top-20 + FAISS Dense top-20 -> RRF 병합 -> top-5 (리랭커 없음, MVP)

주의: PdfPlumber는 2단(two-column) 레이아웃에서 좌우 단이 섞일 수 있어
      layout=True로 로드한다. 표는 병합 셀이 빈 칸으로 나올 수 있고,
      그림/차트 안의 수치는 아예 추출되지 않는다 - 이건 알려진 한계로
      보고서 "한계점" 장에 명시할 것 (design doc 3-1절 참고).
"""
from __future__ import annotations

from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Qwen3-Embedding-0.6B 로더. 모듈 레벨에 캐시해서 여러 번 안 불러온다.

    주의: 이 함수는 실제로 huggingface.co에서 모델 가중치를 내려받는다.
    개발 샌드박스에 인터넷이 막혀 있으면 여기서 실패한다 - 팀 로컬 환경에서
    돌릴 것.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def load_and_split(paths: dict[str, Path]) -> list[Document]:
    """PDF 경로 dict를 받아 로드 + 단일 계층 청킹."""
    docs: list[Document] = []
    missing = []
    for name, path in paths.items():
        if not path.exists():
            missing.append(str(path))
            continue
        loader = PDFPlumberLoader(str(path), text_kwargs={"layout": True})
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_name"] = name
        docs.extend(loaded)

    if missing:
        print(
            "[ingest] 다음 PDF가 없어서 건너뜀 (data/papers/ 에 실제 파일을 "
            f"넣어야 함): {missing}"
        )
    if not docs:
        raise FileNotFoundError(
            "인덱싱할 문서가 하나도 없다. config.py의 TECH_PAPERS/DOMAIN_PAPERS "
            "경로에 실제 PDF를 넣었는지 확인할 것."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_hybrid_retriever(chunks: list[Document]) -> EnsembleRetriever:
    """BM25 + FAISS(Dense) 앙상블 (RRF는 EnsembleRetriever 내부 처리)."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.RETRIEVER_TOP_K_EACH

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    dense = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_TOP_K_EACH},
    )

    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def build_tech_retriever() -> EnsembleRetriever:
    """B(기술조사)가 쓰는 리트리버 — TurboQuant, InfiniGen 원문."""
    chunks = load_and_split(config.TECH_PAPERS)
    return build_hybrid_retriever(chunks)


def build_domain_retriever() -> EnsembleRetriever:
    """E(도메인평가)가 쓰는 리트리버 — 도메인 평가 논문 3편.
    tech_research는 여기서 재검색하지 않고 State에서 바로 읽어와 프롬프트에
    컨텍스트로 넣는다 (3-2절: "tech_research 재사용 + 도메인 문서 검색 병행")."""
    chunks = load_and_split(config.DOMAIN_PAPERS)
    return build_hybrid_retriever(chunks)


def format_docs_for_prompt(docs: list[Document], max_docs: int = 5) -> str:
    """검색 결과를 XML 유사 포맷으로 (실습 rag/utils.py의 format_docs 패턴)."""
    parts = []
    for d in docs[:max_docs]:
        source = d.metadata.get("source_name", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        parts.append(f"<document><source>{source}</source><page>{page}</page>"
                      f"<content>{d.page_content}</content></document>")
    return "\n".join(parts)
