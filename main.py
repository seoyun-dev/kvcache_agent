"""
실행 진입점 (노트북 없이 한 번에 끝까지 돌리고 싶을 때 쓴다).

    python main.py

.env에 OPENAI_API_KEY, TAVILY_API_KEY가 있어야 한다.
data/papers/ 밑에 config.py에 적힌 파일명으로 실제 PDF 5편이 있어야 한다.
Orchestrator-Workers 전환 후에는 nodes_b.py, nodes_cd.py, nodes_e.py,
nodes_fg.py, nodes_eval.py 다섯 개가 필요하다 (nodes_fgh.py가 F/G/H로
합쳐져 있던 걸 nodes_fg.py + nodes_eval.py로 나눴다). graph.py의
ImportError 메시지에 각 파일이 갖춰야 할 모양이 적혀 있다.

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

    if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
        print(f"[main] LangSmith 추적 켜짐 (project={os.environ.get('LANGCHAIN_PROJECT')}) - "
              "동적 fan-out/재시도 trace는 smith.langchain.com에서 캡처할 것.")
    else:
        print("[main] LangSmith 추적 꺼져 있음 - Deliverables의 트레이스 캡처가 필요하면 "
              ".env에 LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY/LANGCHAIN_PROJECT를 채울 것 (.env.example 참고).")

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

    print("[main] 실행 시작 (A -> B -> Orchestrator =(동적 fan-out)=> {C,D,E} -> Synthesizer -> G <-> Evaluator)...")
    result = graph.invoke({}, config={"recursion_limit": 50})

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUTPUT_DIR / "final_report.md"
    report_path.write_text(result.get("final_report", "(보고서 생성 실패)"), encoding="utf-8")

    print(f"[main] 완료. 보고서: {report_path}")
    print(f"[main] plan: {result.get('plan')}")
    print(f"[main] eval_result: {result.get('eval_result')}")


if __name__ == "__main__":
    main()
