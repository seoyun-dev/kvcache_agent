#!/usr/bin/env bash
# 팀 로컬 환경에서 실행할 것 (이 개발 샌드박스는 arxiv.org 접근이 막혀 있어서
# 여기선 못 돌림). data/papers/ 밑에 config.py가 기대하는 파일명으로 받는다.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/papers

declare -A PAPERS=(
  ["turboquant.pdf"]="https://arxiv.org/pdf/2504.19874"
  ["infinigen.pdf"]="https://arxiv.org/pdf/2406.19707"
  ["ondevice_eval.pdf"]="https://arxiv.org/pdf/2505.15030"
  ["elib_mbu.pdf"]="https://arxiv.org/pdf/2508.11269"
  ["llm_in_flash.pdf"]="https://arxiv.org/pdf/2312.11514"
)

for name in "${!PAPERS[@]}"; do
  url="${PAPERS[$name]}"
  echo "받는 중: $name <- $url"
  curl -sL "$url" -o "data/papers/$name"
done

echo "완료. data/papers/ 안에 5개 PDF가 있어야 한다:"
ls -la data/papers/
