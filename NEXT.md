# 다음 세션 — 2026-08-26

전체 상태는 `STATUS.md`. 이 파일은 **바로 이어서 할 일**만.

## 0. 환경 복구 (5분)

```bash
cd ~/MyGIthub/organic-agent/organic-agent-ai
git branch --show-current          # photoselect/chore/2-pipeline-scaffold 여야 한다
photoselect/scripts/spike/setup_ollama.sh          # Ollama 기동 + gemma3:12b 확인
PY=photoselect/scripts/spike/.venv/bin/python
$PY -m pytest photoselect/tests -q                 # 14 passed
```

작업 트리에 **커밋 안 된 변경**이 있다 (PR #11 이후 오늘 한 것):
`STATUS.md` `NEXT.md` `photoselect/{config,llm/*,analyze/vlm.py,analyze/runners/arniqa.py,draft/*,store.py,__main__.py,tests/*}`
`photoselect/scripts/{arniqa_regressors.py,spike/vlm_compare.py}` + spike에서 꺼내 온 사본들.
→ 오늘 첫 일로 커밋 (아래 §3).

## 1. 모델 크기 비교 — 12B vs 7B vs 27B (30분+, 다운로드 20GB)

어제 결론: 프롬프트로는 scene 74%를 못 올린다(1차 재실행 250칸 동일, 변경안 둘 다 하락).
남은 손잡이가 모델 크기다. 채점은 **자동**(`vlm_compare.py`가 1차 채점의 정정값을 정답표로 씀).

```bash
cd photoselect/scripts/spike
ollama pull qwen2.5vl:7b        # ~6GB   — 원가 판단용 (7B로 내려도 되나)
ollama pull gemma3:27b          # ~17GB  — 품질 상한 (tech-stack의 "31B 비교" 자리)

# 같은 50장(시드 0) · 1차 프롬프트 · 결과 자동 채점
.venv/bin/python vlm_tag.py --n 50 --prompt v1 --model qwen2.5vl:7b --out out/vlm_tags_qwen7b.csv
.venv/bin/python vlm_compare.py --v1 out/vlm_tags.csv --v2 out/vlm_tags_qwen7b.csv --grade out/vlm_grade.csv
.venv/bin/python vlm_tag.py --n 50 --prompt v1 --model gemma3:27b --out out/vlm_tags_g27b.csv
.venv/bin/python vlm_compare.py --v1 out/vlm_tags.csv --v2 out/vlm_tags_g27b.csv --grade out/vlm_grade.csv
```

읽는 법: 표의 "2차" 열이 그 모델. **scene ≥ 85%면 27B 채택 검토**(g6e L40S 48GB, ~$1.9/h).
7B가 12B와 ±3%p 안이면 7B로 내려 원가 절감. 속도(s/장)도 함께 기록 — `vlm_tag.py`가 찍어 준다.
`--prompt v1`을 빼먹으면 안 된다 — 모듈 프롬프트도 지금은 1차 원문이지만 명시가 안전하다.

결과는 `STATUS.md` 1.1 VLM 줄과 `photoselect/docs/tech-stack.md` §2 모델 표에 기록.

## 2. 갤러리 재분석 v0.3 (23분, 백그라운드)

프롬프트를 1차 원문으로 되돌렸으므로 `MODEL_VERSION`이 0.3이다. 갤러리(2)의 캡션이 v2 프롬프트
것이라 갱신이 필요하다. 1번 돌리는 동안은 Ollama가 바쁘니 **1번 끝나고**.

```bash
$PY -m photoselect analyze --gallery "dataset1/류지혜고객님 (2)" --force > photoselect/out/analyze-v03.log 2>&1 &
```

## 3. 커밋 (어제 것 — 30분)

이슈 #2는 PR #11로 닫힌다. 어제 추가분은 같은 브랜치에 커밋 하나로 얹는다.

```bash
git add STATUS.md NEXT.md photoselect/config.py photoselect/store.py photoselect/__main__.py \
        photoselect/llm photoselect/draft photoselect/analyze/vlm.py photoselect/analyze/runners/arniqa.py \
        photoselect/tests photoselect/scripts/arniqa_regressors.py
git commit -m "[photoselect] chore: ARNIQA spaq · 정밀도 기준 이유 문장 · LLM(C) 모듈 · VLM 프롬프트 1차 확정

- ARNIQA 회귀기 kadid10k→spaq (실사 회귀기와 상관 0.16~0.34 vs 0.9)
- 이유 문장: AXIS_PRECISION ≥ 0.85 축만 취향 근거로
- llm/: Bedrock Haiku 4.5 structured output — 이유 일괄·피드백 번역, 가드레일 테스트, 실호출 검증
  (ap-northeast-2에 Mantle 엔드포인트 없음 → AnthropicBedrock)
- VLM 프롬프트 v2·v3 기각, 1차 원문 확정 (재실행 250칸 동일)
- STATUS.md 신설"
git push
```

`photoselect/scripts/spike/{vlm_tag,vlm_review,manifest,runners/*}.py`는 chore/1 사본이라 **여기서 커밋하지
않는다.** `vlm_compare.py`와 `vlm_tag.py`의 `--prompt` 플래그는 chore/1로 가져가야 한다 — chore/1 PR 낼 때.

## 4. 그다음 (외부 의존)

| 항목 | 막힌 곳 | 준비된 것 |
|---|---|---|
| 온보딩 파일럿 2~3명 | 사람 섭외 | `review` 페이지 = 수집 도구. 한 사람 20분 |
| DbStore | wes V22 | `store.py` 매핑 주석, `docs/schema-proposal.md` |
| EC2·Lambda·Terraform | 인프라 repo | `handler.py` 스텁 |

## 어제 배운 것 (다음 판단에 쓸 것)

- **측정 잡음부터 잰다.** 프롬프트 비교 전에 "같은 프롬프트 재실행"을 안 했으면 v2 하락을 잡음으로
  오해했을 것이다. 결과: 재실행 0칸 차이 → 하락은 실재.
- **작은 표본으로 프롬프트를 튜닝하지 않는다.** 캡션 문장 하나가 scene을 12%p 흔든다.
  50장 채점표에 맞추는 건 study/02 step4의 승자의 저주 그대로다.
- **가드레일은 코드로.** Bedrock이 죽어도(어제 Mantle 연결 실패) 초안은 템플릿으로 나갔다.
