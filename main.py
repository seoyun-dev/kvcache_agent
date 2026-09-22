"""
실행 진입점 (노트북 없이 한 번에 끝까지 돌리고 싶을 때 쓴다).

    python main.py

.env에 OPENAI_API_KEY, TAVILY_API_KEY가 있어야 한다.
data/papers/ 밑에 config.py에 적힌 파일명으로 실제 PDF 5편이 있어야 한다.
notebooks/01~04번을 먼저 한 번씩 끝까지 돌려서 src/nodes_b.py 등 4개
파일을 만들어둬야 한다(이 리포에는 이미 만들어서 넣어뒀다).

노트북으로 단계별로 보면서 돌리고 싶으면 notebooks/05_full_graph_run.ipynb를
대신 쓸 것 - 이 파일과 똑같은 일을 셀 단위로 나눠서 한다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

from src import config
from src.graph import build_graph
from src.ingest import build_domain_retriever, build_tech_retriever

load_dotenv()


def main():
    for key in ["OPENAI_API_KEY", "TAVILY_API_KEY"]:
        if not os.environ.get(key):
            raise RuntimeError(f"{key} 가 .env에 없다. .env.example 참고.")

    print("[main] 임베딩 인덱스 구축 중 (Qwen3-Embedding-0.6B 다운로드가 처음엔 시간 걸림)...")
    tech_retriever = build_tech_retriever()
    domain_retriever = build_domain_retriever()

    llm = init_chat_model(
        config.LLM_MODEL, model_provider=config.LLM_PROVIDER, temperature=config.LLM_TEMPERATURE
    )
    llm_full = init_chat_model(
        config.LLM_MODEL_FULL, model_provider=config.LLM_PROVIDER, temperature=config.LLM_TEMPERATURE
    )
    web_search_tool = TavilySearch(max_results=5)

    print("[main] 그래프 컴파일...")
    graph = build_graph(llm, llm_full, tech_retriever, domain_retriever, web_search_tool)

    print("[main] 실행 시작 (A -> B -> {C,D,E} -> F -> G <-> H)...")
    result = graph.invoke({}, config={"recursion_limit": 40})

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUTPUT_DIR / "final_report.md"
    report_path.write_text(result.get("final_report", "(보고서 생성 실패)"), encoding="utf-8")

    print(f"[main] 완료. 보고서: {report_path}")
    print(f"[main] validation_result: {result.get('validation_result')}")


if __name__ == "__main__":
    main()
