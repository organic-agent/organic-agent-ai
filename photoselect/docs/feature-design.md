# photoselect 기능 설계 — AI 셀렉터 (고객 화면 기준)

`plan.md`(2026-08-22판)의 워크플로우를 **고객이 보는 순서**로 명세한다. 단계마다 입력·
처리·산출·UI 계약을 적는다. 사진 진단(구 기능 B)은 범위 밖으로 보류 — 이 문서에서 제거.

---

## 0. 전수 분석 (고객 무관, 업로드 직후 GPU 배치)

고객이 갤러리에 들어오기 전에 끝나 있어야 하는 전처리. 상세는 `plan.md` §3-A.

### 고정 축 어휘 (VLM 출력 계약)

모델을 바꿔도 유지되는 enum. VLM에는 이 목록 중 택일을 강제하고(structured output 또는
후처리 매핑), 매핑 실패는 `unknown`.

| 축 | 값 |
|---|---|
| `scene` | `prep`(준비) · `entrance`(입장) · `vow`(서약/예식) · `ring` · `kiss` · `family` · `group`(단체) · `bouquet` · `walk`(퇴장/행진) · `snap`(스냅/자유) · `detail`(소품/공간) · `unknown` |
| `framing` | `closeup` · `half`(상반신) · `full`(전신) · `wide`(풍경형/원경) |
| `lighting` | `natural` · `backlit` · `indoor` · `flash` · `lowlight` |
| `expression` | `smile` · `laugh` · `serious` · `candid`(자연스러움) · `eyes_closed` · `none`(얼굴 없음) |
| `subjects` | `bride` · `groom` · `couple` · `family` · `friends` · `none` |

- `caption`: 한국어 1문장, 고객에게 보여도 되는 톤("야외 자연광 아래 두 분이 마주 보며 웃는 컷").
- 축은 **쌍 비교·선호 분포·커버리지·이유 템플릿**이 전부 이 어휘에 의존하므로, 값 추가는
  문서 갱신 + `model_version` 변경과 함께만.

## 1. 진입 — "AI 추천" 버튼

- 위치: 셀렉 화면. 누르면 **동의 고지** 1회: "지금까지 고르신 사진과 별점을 추천에 사용합니다."
- wes가 실행 조건 검증(갤러리 OPEN · 미제출 · `photo_analysis` 완료 · 활성 잡 없음) 후
  Lambda B 호출. 분석 미완료면 "사진 분석 중 (N분 남음)" 안내.

### 증거량 판정

```
evidence = 별점 수 + 수동 선택 수 + 쌍 비교 수 + 👍/👎 수
evidence ≥ E_min (초기값 15) → 2단계 건너뛰고 3단계
evidence <  E_min            → 2단계 온보딩
```

## 2. 온보딩 — 쌍 비교 (부족할 때만, 1~2분)

- 화면: 사진 2장 나란히 + "어느 쪽이 더 마음에 드세요?" + 건너뛰기. 8~12쌍 진행바.
- 쌍 생성 규칙 (`plan.md` §3-B): 같은 `scene` · **한 축만 다름** · `technical_pct`·
  `aesthetic_pct` 차이 ≤ 15pt · 서로 다른 클러스터. 축별 1~2쌍, 축 순서는 무작위.
- 저장: `pair_comparison_events(selection_id, photo_a, photo_b, chosen, axis)`.
  건너뛰기는 저장하지 않음.
- 이 로그가 주지표(홀드아웃 정확도)의 데이터셋이다 — 쌍의 `axis`를 반드시 남긴다.

## 3. 1차 초안 — 추천 목록 제시 (30초 내)

### 처리 (Lambda B, 결정적)

1. 선호 분포 계산: 별점(가중 = score−3)·선택(+1)·쌍 비교(chosen +1 / other −1)·👍/👎를
   축별 태그 카운트로 집계 → 정규화. 선호 벡터 = 선택·별점 4↑ 사진의 DINOv2 평균.
2. λ·conf(axis) 계산.
3. 장면 커버리지 배분: 목표 장수 K를 `scene` 분포 비례 + 최소 1장으로 배분.
4. 장면별로 `score` 순 그리디 + 클러스터당 1장 + MMR → K장.
5. `ai_recommendations(round=1)` 저장 + 템플릿 이유. 잡 상태 전이와 단일 트랜잭션.
6. Lambda C: 이유 문장 LLM 1회 일괄 → `reason` 갱신 (실패 시 템플릿 유지).

### UI 계약

- 추천은 **별도 패널**("AI 셀렉터 제안 N장"). 고객 선택함은 건드리지 않는다.
- 카드: 사진 + 이유 1줄 + `👍 담기` / `👎 아니요` / `비슷한 컷 N장` (같은 클러스터 차순위 펼침).
- 상단: 커버리지 요약 1줄("입장 8 · 서약 12 · 가족 10 · 스냅 20 …") + **"전체 담기"** 버튼.
- 담기 = `photo_selection_items`에 MANUAL 행 생성 + `accepted_at`, `accept_mode`.
  전체 담기는 `accept_mode='bulk'`.
- 담았다가 선택함에서 빼면 `unselected_at` 기록 (후회 신호).

## 4. 2차 셀렉 — 피드백과 재제시 (5초 내)

- 입력: 패널의 👍/👎 누적 + 자연어 입력창("가족 사진 더 보여주세요", "너무 클로즈업이
  많아요") + 그 사이 추가된 별점.
- 자연어 → Lambda C(Bedrock Haiku, 텍스트만) → `[{axis, tag, delta}]`. 예:
  "가족 더" → `{scene, family, +0.3}`, "클로즈업 많아요" → `{framing, closeup, −0.3}`.
  번역 실패·빈 결과면 무시.
- 재계산: 3단계 1~5 반복, `round+1`. 이미 담은 사진은 고정(불가침), 👎는 후보 제외,
  이유는 템플릿(LLM 호출 없음).
- "다시 추천" 버튼으로 명시 트리거. 라운드 수 제한 없음(지표로 관찰).

## 5. 종료

- 고객이 선택함을 확정·제출 — AI와 무관한 기존 wes 흐름. `photo_selections.status`는
  AI가 절대 바꾸지 않는다.
- 제출 시점에 `ai_recommendations`로 수락률·nDCG·후회율·라운드 수 계산 가능 (wes 또는
  배치 리포트).

## 이유 문장 템플릿 (C 폴백이자 2차 기본)

```
[{scene_ko}] {lighting_ko}에서 찍힌 {framing_ko} 컷이에요.
{pref_axis_ko}을(를) 선호하시는 것 같아 골랐어요.   # conf(axis) 최대 축, λ>0.4일 때만
비슷한 컷 {n}장 중 {rank_reason_ko} 이 사진을 골랐어요.   # cluster_size>1일 때만
```
LLM 문장화 프롬프트 규칙: 점수 숫자 언급 금지 · 단정 대신 제안 톤 · 신호에 없는 내용 금지 ·
photo_id 불일치 거부.

## 계약 요약

| 대상 | 내용 |
|---|---|
| wes → Lambda B | `{"selectionId", "jobId", "mode": "draft"|"refine", "feedbackText"?}` |
| `ai_recommendations` | 제시·반응 로그 — `plan.md` §5 |
| `pair_comparison_events` | 온보딩 로그 |
| `photo_selection_items` | AI는 **쓰지 않음**. 고객이 담을 때 wes가 MANUAL 생성 |
| `photo_ratings` | 읽기 허용 (동의 고지 전제) |
| 프론트 | 추천 패널 · 쌍 비교 화면 · 자연어 입력창 · 동의 고지 — organic-agent-test-web 트랙 협의 |
