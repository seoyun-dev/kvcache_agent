"""Evaluator. 보고서 품질 평가 노드 (가이드 D절) - 기존 nodes_fgh.py의 H를 확장 이전.

평가 방식은 Hybrid(가이드 3안) = 규칙(1안) + LLM Judge(2안).
네 평가 항목마다 규칙 검사와 Judge 판정을 둘 다 돌려, 둘 다 통과해야 그 항목을
통과로 본다. 규칙은 결정적이라 형식·서열어처럼 "있다/없다"로 가를 수 있는 것을
잡고, Judge는 규칙이 못 보는 내용 수준(주장이 근거에 실제로 받쳐지는지, 금칙어
없이 돌려 말한 우열 판정)을 본다.

  항목            | 규칙 (결정적)                               | LLM Judge (내용)
  Groundedness    | 본문 [n] 인용이 REFERENCE에 실재 / 관점 장마다 인용 존재 | 주장이 워커 근거로 추적되는가
  중립성          | 서열 금칙어 / 필수 장·개조식·[소결] 형식        | 우회적 우열 판정·전반 추천 결론
  편향 통제       | 출처 수 하한 / 단일 출처 쏠림 / 한쪽 기술만 인용 | 한 기술에 유리한 근거만 골랐는가
  관점 커버리지   | 4관점(TRL·시장·이해관계자·도메인) 장 + 두 기술 모두 언급 | 각 관점이 두 기술을 실질적으로 다루는가

미달이면 route_after_eval이 G로 되돌린다(G<->Evaluator 루프). 상한은
config.MAX_RETRY_EVAL이고, 도달하면 통과시키되 미달 항목을 보고서에 감사 기록으로
남긴다(forced_pass) - 검증 실패 보고서가 통과 표시로 제출되는 일을 막는다.
Judge 호출이 실패하면 그래프를 멈추지 않고 규칙 결과만으로 판정한 뒤 errors에 남긴다.
"""
import json
import re

from src import config
from src.node_utils import get_worker_output, get_worker_references

TECHS = ["TurboQuant", "InfiniGen"]
WORKER_AGENTS = {"C": "시장성", "D": "이해관계자", "E": "도메인"}

# ── 관점 커버리지: 가이드의 4관점 -> 보고서 장 제목 키워드 ──
# 기술 성숙도(TRL)는 전담 워커 없이 B 산출물로 "3. 기술 개요" 장에 들어간다.
PERSPECTIVE_CHAPTERS = {
    "기술 성숙도": "기술 개요",
    "시장성": "시장성 평가",
    "이해관계자": "이해관계자 평가",
    "도메인 적용": "도메인 평가",
}
REQUIRED_CHAPTERS = ["SUMMARY", *PERSPECTIVE_CHAPTERS.values(), "시사점", "한계점", "REFERENCE"]
EVAL_CHAPTERS = ["시장성 평가", "이해관계자 평가", "도메인 평가"]  # 4장 하위 3개

# ── 중립성 ──
FORBIDDEN_WORDS = ["우수", "우월", "우위", "낫다", "권장", "추천"]
RANKING_CHECK_CHAPTERS = ["SUMMARY", "관점별 평가", *EVAL_CHAPTERS, "시사점"]

# ── 형식 (기존 H 임계값 승계) ──
MIN_SUMMARY_BLOCKS = 3
MIN_BULLET_LINES = 15

# ── Groundedness / 편향 통제 ──
MIN_NUMBERED_REFS = 5        # REFERENCE 번호 항목 하한 (고정 논문 5건)
MIN_WEB_REFS = 2             # 논문 외 출처 하한 - 논문만으로 시장·이해관계자를 쓰면 근거 편중
MAX_SINGLE_REF_SHARE = 0.5   # 4장 인용 중 한 출처가 차지하는 비율 상한

CRITERIA = ["groundedness", "neutrality", "bias", "coverage"]
CRITERIA_KO = {
    "groundedness": "Groundedness",
    "neutrality": "중립성",
    "bias": "편향 통제",
    "coverage": "관점 커버리지",
}

EVIDENCE_CHAR_LIMIT = 12000  # Judge 프롬프트에 싣는 근거 상한(토큰 비용 제어)

JUDGE_PROMPT = """당신은 KV cache 최적화 기술 평가 보고서의 품질 심사관이다.
보고서를 고치지 말고, 아래 네 항목을 [근거 자료]와 대조해 판정만 하라.
이 보고서의 목적은 두 기술(TurboQuant, InfiniGen)의 다관점 "비교"이지 우열 판정이 아니다.

1. groundedness: 보고서의 수치·사실 주장이 [근거 자료]로 추적되는가.
   - 근거 자료에 없는 수치·제품명·날짜·발화가 단정적으로 쓰였으면 issue로 적는다.
   - "근거 없음"이라고 명시한 항목은 위반이 아니다.
2. neutrality: 특정 기술 추천이나 전반적 우열 판정이 없는가.
   - "~에 유리한 조건", "~에서 제약이 크다" 같은 조건부·항목 단위 비교는 허용한다.
   - 금칙어가 없어도 "결론적으로 X를 택해야 한다", "X가 전반적으로 앞선다"처럼
     보고서 전체 수준의 우열·선택 결론을 내리면 위반이다.
3. bias: 단일 출처나 한 기술에 유리한 근거로 쏠리지 않았는가.
   - 한 기술에는 비판·한계를 적고 다른 기술에는 적지 않는 비대칭,
     한쪽 기술의 "근거 없음"을 열세로 해석한 서술을 찾는다.
4. coverage: 기술 성숙도(TRL), 시장성, 이해관계자, 도메인 적용 네 관점이
   각각 두 기술을 모두 실질적으로 다루는가 (장 제목만 있고 내용이 비면 위반).

각 항목은 passed(bool)와 issues(위반 사례 목록)를 채운다. issue 하나는
  quote  : 위반이 있는 보고서 문장을 **글자 그대로** 짧게 인용 (요약·의역 금지)
  problem: 무엇이 문제이고 어떻게 고쳐야 하는지 (G가 그대로 받아 재작성한다)
groundedness는 수치를 지목할 때 그 수치를 근거 자료에서 실제로 찾아본 뒤에만 issue로 적는다.
위반이 없으면 passed=true, issues=[] 로 둔다. 사소한 문체 문제는 위반으로 보지 않는다.

[근거 자료 - 워커 산출물과 참고문헌 후보]
{evidence}

[평가 대상 보고서]
{report}"""


# ───────────────────────── 보고서 파싱 ─────────────────────────

def _split_chapters(report):
    """마크다운 헤더 기준으로 보고서를 {장 제목: 본문}으로 자른다.

    본문은 **다음 같은 단계 이상 헤더 전까지** 전부다 — 하위 절(##·###)을 포함한다.
    처음 판은 모든 헤더에서 끊어서 '3. 기술 개요' 처럼 바로 아래 ## 로 시작하는 장의
    본문이 비고(TRL 없음 오탐), '4.2 이해관계자 평가' 의 인용이 4.2.1~4.2.3 에만 있어
    0건으로 읽혔다(Groundedness 오탐). 편향 통제 검사도 같은 본문을 읽어 4.2 하위 절이
    빠진 채 통과했었다.
    """
    lines = report.split("\n")
    heads = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))

    chapters = {"(머리말)": "\n".join(lines[: heads[0][0]] if heads else lines)}
    for k, (i, level, title) in enumerate(heads):
        end = len(lines)
        for j, lv, _ in heads[k + 1:]:
            if lv <= level:
                end = j
                break
        body = "\n".join(lines[i + 1:end])
        # 같은 제목이 두 번 나오면(예: 기술별 '## TurboQuant') 이어 붙인다 — 덮어쓰면 앞 것이 사라진다.
        chapters[title] = chapters[title] + "\n" + body if title in chapters else body
    return chapters


def _body_of(chapters, keywords):
    """제목에 keywords 중 하나라도 들어간 장들의 본문을 합친다."""
    return "\n".join(b for t, b in chapters.items() if any(k in t for k in keywords))


def _citations(text):
    """본문의 [n], [n, m], [n-m] 인용 번호를 전부 뽑는다. [소결] 같은 비숫자 괄호는 무시."""
    nums = []
    for group in re.findall(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]", text):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[\-–]", part):
                lo, hi = (int(x) for x in re.split(r"\s*[\-–]\s*", part))
                nums.extend(range(lo, hi + 1))
            else:
                nums.append(int(part))
    return nums


def _reference_entries(chapters):
    """REFERENCE 장의 {번호: 항목 첫 줄}."""
    ref_body = _body_of(chapters, ["REFERENCE"])
    return {
        int(m.group(1)): m.group(2).strip()
        for m in re.finditer(r"^\s*\[(\d+)\]\s*(.*)$", ref_body, re.M)
    }


# ───────────────────────── 규칙 검사 (1안) ─────────────────────────

def _rule_groundedness(chapters):
    issues = []
    refs = _reference_entries(chapters)
    body = "\n".join(b for t, b in chapters.items() if "REFERENCE" not in t)
    dangling = sorted({n for n in _citations(body) if n not in refs})
    if dangling:
        issues.append(f"본문 인용 {dangling}이 REFERENCE에 없다. 없는 번호를 지우거나 REFERENCE에 항목을 추가하라.")
    # URL 이 항목 둘째 줄로 넘어가는 형식도 있어서 항목 전체(다음 [n] 전까지)를 본다.
    ref_body = _body_of(chapters, ["REFERENCE"])
    blocks = re.split(r"(?m)^\s*(?=\[\d+\])", ref_body)
    homonyms = sorted(
        int(m.group(1))
        for blk in blocks
        if (m := re.match(r"\[(\d+)\]", blk)) and any(k in blk for k in config.HOMONYM_URL_MARKERS)
    )
    if homonyms:
        issues.append(
            f"REFERENCE {homonyms}는 이름만 같은 다른 프로젝트다(예: 3D 장면 생성기 InfiniGen). "
            "그 항목과 본문 인용을 지우고, 근거가 남지 않으면 '근거 없음'으로 써라."
        )
    for ch in EVAL_CHAPTERS:
        if not _citations(_body_of(chapters, [ch])):
            issues.append(f"'{ch}' 장에 [n] 인용이 하나도 없다. 근거 문장 끝에 REFERENCE 번호를 붙여라.")
    return issues


def _rule_neutrality(report, chapters):
    issues = []
    scoped = _body_of(chapters, RANKING_CHECK_CHAPTERS)
    found = [w for w in FORBIDDEN_WORDS if w in scoped]
    if found:
        issues.append(f"서열 표현 발견: {found}. 조건부 서술(~에 유리한 조건, ~에서 제약이 크다)로 바꿔라.")

    # 형식 규칙은 기존 H 승계. 개조식·[소결] 구조가 깨지면 두 기술을 나란히 적는
    # 대칭 비교가 무너져 한쪽만 서술하는 줄글이 나오므로 중립성 쪽에 묶는다.
    n_concl = report.count("[소결]")
    if n_concl < MIN_SUMMARY_BLOCKS:
        issues.append(f"[소결] 블록이 {n_concl}개뿐이다. 4.1/4.2/4.3 각 장 끝에 하나씩 총 {MIN_SUMMARY_BLOCKS}개를 넣어라.")
    bullets = sum(
        1 for line in _body_of(chapters, EVAL_CHAPTERS).split("\n")
        if line.lstrip().startswith("- ")
    )
    if bullets < MIN_BULLET_LINES:
        issues.append(
            f"관점별 평가가 개조식이 아니다('- ' 줄 {bullets}개, {MIN_BULLET_LINES}개 이상 필요). "
            "4장 줄글을 'TurboQuant: / InfiniGen: / →' 3줄 개조식으로 바꿔라."
        )
    return issues


def _rule_bias(chapters):
    issues = []
    refs = _reference_entries(chapters)
    if len(refs) < MIN_NUMBERED_REFS:
        issues.append(f"REFERENCE 번호 항목이 {len(refs)}개뿐이다. 확정 논문 5건을 포함해 {MIN_NUMBERED_REFS}개 이상 적어라.")
    web = [n for n in refs if n > 5]  # [1]~[5]는 REPORT_PROMPT가 고정한 논문 5건
    if len(web) < MIN_WEB_REFS:
        issues.append(
            f"논문 외 출처가 {len(web)}건뿐이다(하한 {MIN_WEB_REFS}). "
            "시장성·이해관계자 근거로 쓴 웹 출처를 [6]번부터 REFERENCE에 넣고 본문에서 인용하라."
        )

    eval_body = _body_of(chapters, EVAL_CHAPTERS)
    cited = _citations(eval_body)
    if cited:
        top = max(set(cited), key=cited.count)
        share = cited.count(top) / len(cited)
        if share > MAX_SINGLE_REF_SHARE:
            issues.append(
                f"4장 인용 {len(cited)}건 중 [{top}]이 {share:.0%}를 차지한다(상한 {MAX_SINGLE_REF_SHARE:.0%}). "
                "다른 출처의 근거도 함께 인용하라."
            )

    # 대칭 인용: 한 기술 줄에만 출처가 붙고 다른 기술 줄은 근거 없이 서술되면 편중.
    per_tech = {t: 0 for t in TECHS}
    for line in eval_body.split("\n"):
        for t in TECHS:
            if re.match(rf"^\s*-\s*\**{t}\**\s*:", line) and _citations(line):
                per_tech[t] += 1
    lo, hi = min(per_tech.values()), max(per_tech.values())
    if hi >= 3 and lo == 0:
        empty = [t for t, n in per_tech.items() if n == 0]
        issues.append(f"4장에서 {empty} 줄에는 인용이 하나도 없다({per_tech}). 두 기술 모두 근거를 붙여라.")
    return issues


def _rule_coverage(chapters):
    issues = []
    titles = list(chapters.keys())
    missing = [c for c in REQUIRED_CHAPTERS if not any(c in t for t in titles)]
    if missing:
        issues.append(f"다음 장이 빠졌다: {missing}. 지정된 장 제목을 글자 그대로 써라.")
    for view, ch in PERSPECTIVE_CHAPTERS.items():
        body = _body_of(chapters, [ch])
        if not body.strip():
            continue  # 장 누락은 위에서 이미 잡음
        absent = [t for t in TECHS if t not in body]
        if absent:
            issues.append(f"{view} 관점('{ch}' 장)에 {absent} 서술이 없다. 두 기술을 나란히 다뤄라.")
    if "TRL" not in _body_of(chapters, [PERSPECTIVE_CHAPTERS["기술 성숙도"]]):
        issues.append("기술 성숙도 관점: '기술 개요' 장에 TRL 판정이 없다. trl_ondevice/trl_global 값을 적어라.")
    return issues


URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _norm_url(u):
    return u.rstrip(".,;:'\"").rstrip("/").lower()


def _rule_reference_urls(chapters, evidence_urls):
    """REFERENCE 웹 항목([6]~)의 URL 이 워커가 모은 근거에 있는지, 같은 URL 이 두 번 나오지 않는지.

    2026-10-07 run efde822e 에서 [6] 「mlx-vlm Releases」가 REPORT_PROMPT 의 예시를 그대로
    옮긴 것이었다 - 근거 어디에도 없는 URL 을 4곳에 인용했고 Judge 도 통과시켰다.
    evidence_urls 가 None 이면(State 없이 단독 검사) 출처 대조는 건너뛴다.
    """
    issues = []
    ref_body = _body_of(chapters, ["REFERENCE"])
    blocks = [b for b in re.split(r"(?m)^\s*(?=\[\d+\])", ref_body) if re.match(r"\[\d+\]", b)]
    seen, dup, unknown = {}, [], []
    # [1]~[5] 는 REPORT_PROMPT 가 고정한 논문이다. 웹 항목이 그 URL 을 다시 적으면 중복으로만 본다.
    fixed = {
        _norm_url(u) for blk in blocks
        if int(re.match(r"\[(\d+)\]", blk).group(1)) <= 5 for u in URL_RE.findall(blk)
    }
    for blk in blocks:
        n = int(re.match(r"\[(\d+)\]", blk).group(1))
        for u in URL_RE.findall(blk):
            key = _norm_url(u)
            if key in seen and seen[key] != n:
                dup.append((seen[key], n))
            seen.setdefault(key, n)
            if n > 5 and evidence_urls is not None and key not in evidence_urls and key not in fixed:
                unknown.append(n)
    if dup:
        issues.append(f"REFERENCE 에 같은 URL 이 중복된다 {sorted(set(dup))}. 하나만 남기고 본문 인용 번호를 합쳐라.")
    if unknown:
        issues.append(
            f"REFERENCE {sorted(set(unknown))}의 URL 이 워커가 모은 근거에 없다(지어낸 출처). "
            "그 항목과 본문 인용을 지우고, 근거가 남지 않으면 '근거 없음'으로 써라."
        )
    return issues


def evidence_url_set(state):
    """워커·B 가 실제로 모은 참고문헌 후보의 URL 집합."""
    texts = [json.dumps(state.get("tech_references", []), ensure_ascii=False)]
    for agent in WORKER_AGENTS:
        texts.append(json.dumps(get_worker_references(state, agent), ensure_ascii=False))
    return {_norm_url(u) for t in texts for u in URL_RE.findall(t)}


def run_rule_checks(report, evidence_urls=None):
    """네 항목별 규칙 위반 목록. State와 무관한 순수 함수라 단독 테스트가 가능하다.
    evidence_urls 를 주면 REFERENCE 의 웹 URL 이 근거에 실재하는지도 본다."""
    chapters = _split_chapters(report)
    return {
        "groundedness": _rule_groundedness(chapters) + _rule_reference_urls(chapters, evidence_urls),
        "neutrality": _rule_neutrality(report, chapters),
        "bias": _rule_bias(chapters),
        "coverage": _rule_coverage(chapters),
    }


# ───────────────────────── LLM Judge (2안) ─────────────────────────

def _evidence_parts(state):
    """Judge가 대조할 근거: B의 기술조사 + C/D/E 최신 성공 결과 + 각 참고문헌 후보.
    실패한 워커는 get_worker_output이 '판단보류' 플레이스홀더를 돌려주므로,
    Judge는 그 관점의 '근거 없음' 서술을 hallucination으로 오판하지 않는다."""
    parts = [f"## 기술조사(B)\n{state.get('tech_research', {})}",
             f"### 참고문헌(B)\n{state.get('tech_references', [])}"]
    for agent, view in WORKER_AGENTS.items():
        parts.append(f"## {view}({agent})\n{get_worker_output(state, agent)}")
        parts.append(f"### 참고문헌({agent})\n{get_worker_references(state, agent)}")
    return parts


def build_evidence(state):
    """Judge 프롬프트용 근거. 앞쪽(B) 자료가 길어 뒤쪽 워커 근거가 잘려 나가지
    않도록 칸마다 균등하게 자른다."""
    parts = _evidence_parts(state)
    per_part = EVIDENCE_CHAR_LIMIT // len(parts)
    return "\n\n".join(p[:per_part] for p in parts)


# 단위가 붙었거나 소수인 수치만 대조한다. "TRL 5"의 5처럼 단위 없는 정수는
# 근거 어디에나 우연히 있을 수 있어 대조 근거로 못 쓴다.
_UNIT_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|MB|KB|%p|%|ms|W|K|배|x|비트|bit)|\d+\.\d+", re.I)


def _norm(text):
    return re.sub(r"\s+", "", text).lower()


def _is_false_grounding_alarm(quote, evidence_norm):
    """Judge가 '근거에 없다'고 지목한 문장의 수치가 실제로는 근거에 전부 있으면 오탐.
    LLM Judge는 비결정적이라 같은 수치를 한 번은 통과, 한 번은 위반으로 판정한다
    (실측: 근거에 있는 '1.1GB', '0.2%p'를 위반 보고서에서만 지목). 수치 대조는
    결정적인 규칙이 맡아 Judge의 오탐을 걷어낸다 - Hybrid의 규칙 쪽 역할."""
    nums = _UNIT_NUMBER.findall(re.sub(r"\[\d+\]", "", quote))  # [3] 인용 번호 제외
    return bool(nums) and all(_norm(n) in evidence_norm for n in nums)


def _make_judge(llm):
    """구조화 출력 Judge. 클래스 문 대신 create_model을 쓰는 건 nodes_fgh.make_node_f와
    같은 이유 - 노트북 셀에서 정의한 클래스는 '파일로 저장' 셀이 소스를 못 읽는다."""
    from pydantic import create_model

    issue = create_model("JudgeIssue", quote=(str, ...), problem=(str, ...))
    criterion = create_model("JudgeCriterion", passed=(bool, ...), issues=(list[issue], ...))
    verdict = create_model("JudgeVerdict", **{c: (criterion, ...) for c in CRITERIA})
    return llm.with_structured_output(verdict)


# ───────────────────────── 노드 ─────────────────────────

def make_node_evaluator(llm):
    judge = _make_judge(llm)

    def node_evaluator(state):
        report = state.get("final_report", "")
        retry_count = state.get("eval_retry_count", 0)

        rule = run_rule_checks(report, evidence_url_set(state))

        errors = []
        try:
            v = judge.invoke(JUDGE_PROMPT.format(evidence=build_evidence(state), report=report))
            judge_result = {
                c: {"passed": getattr(v, c).passed,
                    "issues": [f"'{i.quote}' — {i.problem}" for i in getattr(v, c).issues],
                    "dropped": []}
                for c in CRITERIA
            }
            # Groundedness 오탐 제거: 지목된 수치가 근거에 실재하면 버린다.
            evidence_norm = _norm("\n".join(_evidence_parts(state)))
            g = getattr(v, "groundedness")
            kept = [f"'{i.quote}' — {i.problem}" for i in g.issues if not _is_false_grounding_alarm(i.quote, evidence_norm)]
            dropped = [f"'{i.quote}' — {i.problem}" for i in g.issues if _is_false_grounding_alarm(i.quote, evidence_norm)]
            if dropped:
                judge_result["groundedness"] = {
                    # 지목된 issue가 전부 오탐이었으면 통과로 되돌린다.
                    "passed": g.passed or not kept,
                    "issues": kept,
                    "dropped": dropped,
                }
            judge_ok = True
        except Exception as e:  # noqa: BLE001 - Judge 장애로 그래프를 멈추지 않는다(규칙만으로 판정)
            judge_result = {c: {"passed": True, "issues": [], "dropped": []} for c in CRITERIA}
            judge_ok = False
            errors.append({"agent": "Evaluator", "error": f"LLM Judge 실패, 규칙 검사만 반영: {e}"})

        criteria = {}
        missing = []
        for c in CRITERIA:
            j = judge_result[c]
            passed = not rule[c] and j["passed"]
            criteria[c] = {
                "passed": passed,
                "rule_issues": rule[c],
                "judge_passed": j["passed"],
                "judge_issues": j["issues"],
                "judge_dropped": j["dropped"],  # 관측성: 수치 대조로 걸러낸 Judge 오탐
            }
            # Judge가 passed=false인데 issues를 비워 보내면 G가 고칠 게 없으므로 항목명이라도 남긴다.
            judge_issues = j["issues"] or ([] if j["passed"] else ["(Judge 미달, 세부 사유 없음)"])
            missing += [f"[{CRITERIA_KO[c]}] {i}" for i in rule[c] + (judge_issues if not j["passed"] else [])]

        passed = all(criteria[c]["passed"] for c in CRITERIA)
        result = {
            "passed": passed,
            "missing_items": missing,
            "forced_pass": False,
            "criteria": criteria,
            "judge_ok": judge_ok,
            "attempt": retry_count + 1,
            "feedback": _revision_note(missing),
        }
        update = {"eval_result": result}
        if errors:
            update["errors"] = errors

        if passed:
            update["eval_retry_count"] = retry_count
            return update

        new_retry_count = retry_count + 1
        update["eval_retry_count"] = new_retry_count
        if new_retry_count > config.MAX_RETRY_EVAL:
            # 상한 도달. 통과시키되 미달 항목을 보고서에 남긴다(종료 보장 + 감사 추적).
            result["forced_pass"] = True
            update["final_report"] = _append_audit_note(report, missing, new_retry_count)
        return update

    return node_evaluator


def _revision_note(missing):
    """G 재진입 시 프롬프트 {revision_note}에 그대로 넣을 수 있는 재작성 지시문."""
    if not missing:
        return ""
    return "[재작성 지시 - 품질 평가 미달 항목]\n" + "\n".join(f"- {m}" for m in missing)


def _append_audit_note(report, missing, retry_count):
    """forced_pass 시 평가 기록을 REFERENCE 앞에 끼워 넣는다."""
    note = (
        "\n\n### 보고서 품질 평가 기록 (자동 생성)\n\n"
        f"품질 평가 노드가 재작성 상한({config.MAX_RETRY_EVAL}회)에 도달해 "
        f"아래 항목을 충족하지 못한 채 통과 처리했다. 평가 시도 {retry_count}회.\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n"
    )
    for line in report.split("\n"):
        if re.match(r"^\s{0,3}#{1,6}\s+.*REFERENCE", line):
            return report.replace(line, note.strip() + "\n\n" + line, 1)
    return report + note


def route_after_eval(state):
    """Evaluator 다음 조건부 엣지. graph.py의 add_conditional_edges가 쓴다.
    forced_pass도 passed=False로 남지만 상한 도달이므로 END로 보낸다."""
    r = state.get("eval_result", {})
    if r.get("passed") or r.get("forced_pass"):
        return "END"
    return "G"
