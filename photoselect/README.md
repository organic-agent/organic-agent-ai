# photoselect — AI 셀렉터 서버

설계: `docs/plan.md`(단일 소스) · 로드맵: `docs/roadmap.md` · 스택: `docs/tech-stack.md`.
이 README는 **코드가 지금 무엇을 할 수 있는가**만 적는다.

## 두 단계

| | 상태 | 무엇 |
|---|---|---|
| **1단계 사진학 추천** | ✅ 동작 | 경량 3종 + VLM 태그 + 클러스터 → 사진학 prior + 장면 커버리지 + MMR → 초안 K장. **실제 데이터셋으로 테스트한다.** |
| **2단계 취향 반영** | ✅ 코드 동작 · 데이터 대기 | 선택·별점·온보딩 쌍 → Bradley-Terry → λ·conf → 같은 초안 코드가 개인화된다. 데이터는 `review` 페이지로 만들거나(팀원 채점) 실제 고객 데이터(DB 연결 후). |

**분기가 없다.** `evidence`가 비어 있으면 1단계, 채워지면 2단계다. 같은 `draft/job.py`다.

## 구조 (tech-stack.md §3 — embedder와 같은 모양)

```
photoselect/
├── __main__.py   로컬 CLI            handler.py   Lambda (B)      config.py   설정·손잡이
├── store.py      저장소 인터페이스     gallery.py   갤러리 소스     axes.py     고정 축 어휘(계약)
├── analyze/  A   job.py · runners/(faces·laion·arniqa) · vlm.py · cluster.py · represent.py
├── draft/    B   job.py · evidence.py · preference.py(BT+conf) · scoring.py · rerank.py(커버리지·MMR)
├── review/       html.py  초안 검수 페이지 → evidence.json
├── llm/      C   (미구현 — Bedrock 텍스트)
└── tests/
```

**DB가 아직 없다**(wes V22 협의 중). 그래서 `store.Store`가 인터페이스이고 지금은 `LocalStore`
(out/ 아래 JSONL·npy)가 붙어 있다. `DbStore`는 컬럼 매핑만 주석으로 있다 — 스키마가 머지되면
그것만 채우고 나머지는 안 바뀐다.

## 실행

torch·mediapipe가 있는 venv가 필요하다. 스파이크 venv를 그대로 쓴다.

```bash
cd organic-agent-ai
PY=photoselect/scripts/spike/.venv/bin/python

$PY -m photoselect analyze --list                                      # 갤러리 목록
$PY -m photoselect analyze --gallery "dataset1/류지혜고객님 (2)"          # A (VLM은 Ollama 떠 있을 때)
$PY -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)"          # B — 1단계 (evidence 없음)
$PY -m photoselect review  --gallery "dataset1/류지혜고객님 (2)"          # HTML 검수 페이지
# 페이지에서 담기/별점/쌍 비교 → '내보내기' → out/<갤러리>/evidence.json 으로 저장
$PY -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)"          # B — 2단계 (evidence 반영, round 2)
$PY -m pytest photoselect/tests -q
```

VLM은 `photoselect/scripts/spike/setup_ollama.sh`로 띄운다. 없으면 `--no-vlm` — 태그가 기본값으로
채워져 커버리지·이유 문장이 빈약해지지만 나머지는 돈다.

## 출력 (`out/<갤러리>/`)

| 파일 | DB 대응 |
|---|---|
| `analysis.jsonl` | `photo_analysis` 행 |
| `embeddings.npy` + `embedding_ids.json` | `photos.embedding` |
| `evidence.json` | `photo_selection_items` + `photo_ratings` + `pair_comparison_events` (읽기 전용) |
| `recommendations.jsonl` | `ai_recommendations` (round별 누적) |
| `review-rN.html` | 프론트 대신 |

## 손잡이 (`config.py`)

전부 `study/01-recsys/lab`에서 합성 데이터로 정한 값이고 **실제 갤러리에서 다시 정할 대상**이다.
`analyze`가 찍는 `similarityProfile`로 `cluster_threshold`를, `review` 페이지를 눈으로 보고
`lambda_mmr`·`scene_cap_ratio`를 정한다. `lambda_k`는 쌍 비교 데이터가 생기면 재추정한다.

## 접근 규칙 (CLAUDE.md) — 인터페이스로 강제

`Store`에 `photo_selections.status`를 만지는 메서드가 없고, `photo_selection_items`·
`photo_ratings`는 읽기 메서드만 있다. 쓰기는 `write_analysis`·`write_recommendations` 둘뿐.
이미지는 로컬 모델(경량 3종·Ollama/vLLM)만 본다.
