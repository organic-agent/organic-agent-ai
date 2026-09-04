# 분석 배치 A 감사 — 갤러리 1 실데이터 기준 모델별 유효성

- 대상: 로컬 Postgres(`wes-postgres-local`) `gallery_id=1`, 822장, 전부 `model_version=photoselect-a-0.3`
  (`ai_analysis_jobs` #5, 5,709초 — 그중 VLM 5,468초 = 96%)
- 검토 기준: 각 모델의 산출값이 ① 이 갤러리에서 **변별력**이 있는가, ② 추천(배치 B)에서
  **실제로 읽히는가**, ③ 읽힌다면 **결과를 바꾸는가**. 셋 중 하나라도 아니면 "의미 없음"으로 분류.
- 작성일: 2026-08-29. 재현 SQL은 말미.

## 요약 판정

| 모델 | 산출 컬럼 | 변별력 | 소비처 | 결과 영향 | 판정 |
|---|---|---|---|---|---|
| ARNIQA(spaq) | `technical_score` → `technical_pct` | ✅ IQR 0.08, 백분위 0~100 | prior 50%, 대표 선정, 근거 | ✅ 추천 60장 평균 백분위 75 | **유효** |
| LAION Aesthetic v2 | `aesthetic_score` → `aesthetic_pct` | ⚠️ 원점수 5.04~6.37, IQR 0.31 (좁음) — 백분위로 펴서 씀. ARNIQA와 상관 −0.05(독립) | prior 50%, 근거 | ✅ 추천 평균 백분위 80 | **유효, 단 절대값 의미 없음** |
| DINOv2 (embedder) | `embedding` | ✅ 이웃 유사도 p50 0.887 / p90 0.974 | 버스트 클러스터, MMR, "담은 사진과 비슷" | ✅ 301 클러스터, 대표-형제 tech 격차 7.9pt | **유효** |
| MediaPipe Face Landmarker | `face_count`, `eyes_open` | ❌ 81%(668장)에서 얼굴 0 — 검출기 한계 | 대표 선정 1순위 키, 형제 탈락 사유 | ❌ 대표 301개 중 `eyes_open` 사유 5개, "눈 감김" 근거 1회 | **사실상 무효** |
| MediaPipe (부산물) | `smile`, `max_face_ratio` | — | **없음** (저장만) | — | **무의미 (dead data)** |
| VLM gemma3:12b — `scene` | `scene` | ⚠️ 정밀도 0.74. `detail` 82장이 전부 인물 컷, 스튜디오인데 `walk` 174장 | 장면 쿼터, 파생 쌍의 "같은 장면" 조건 | ✅ 커버리지 분포가 쿼터대로 나옴 | **유효하나 라벨 오류 큼** |
| VLM — `framing` | `framing` | ✅ full 596 / half 213 / closeup 10 — 얼굴 검출 결과와 정확히 상관 | 취향 피처, 근거 인용(정밀도 0.92) | ⏸ 증거 0이라 아직 미작동 | **유효 (잠재)** |
| VLM — `lighting` | `lighting` | ❌ 실내 스튜디오 촬영인데 `natural` 716 / `indoor` 94 | 취향 피처만 (근거 인용 불가, 정밀도 0.84) | ⏸ | **무의미 — 이 갤러리에서 라벨이 틀림** |
| VLM — `expression` | `expression` | ⚠️ smile 636(77%) 편중. `eyes_closed` 4장 | 취향 피처, 근거 인용 | ⏸ | **유효하나 저정보** |
| VLM — `subjects` | `subjects` | ✅ 정밀도 0.98, 분포 자연스러움 | 취향 피처, rarity, 근거 인용 | ⏸ | **유효 (잠재)** |
| VLM — `caption` | `caption` | ⚠️ 822장 중 유니크 513, 같은 문장 22회 반복. 칭찬어 13건(금지 규칙 위반) | 근거 문장 `moment` 재료 (LLM 경로에서만) | 낮음 | **비용 대비 낮음** |
| CLIP ViT-L/14 (LAION 내부) | (DB에 안 씀) | — | LAION 점수 계산용으로만 | — | 임베딩은 폐기 — 재활용 여지 |

핵심: **런타임의 96%를 쓰는 VLM에서 실제로 추천을 바꾸는 축은 `scene` 하나**이고, 그마저 라벨
오류가 눈에 띈다. 얼굴 모델은 꺼진 것과 같다. 취향 피처(framing·expression·subjects)는
증거(쌍비교·피드백) 생산자가 아직 없어 잠재 가치만 있다.

## 모델별 근거

### 1. ARNIQA(spaq) — 유효

- 원점수 0.365~0.765, Q1 0.581 / Q3 0.660. 백분위로 변환해 0~100 전 구간 사용.
- `technical_pct=50`(결측 대체값) 0건 → 결측 없음.
- 클러스터 대표(rank 0)와 형제 평균의 tech 백분위 격차 7.9pt — 얼굴 신호가 죽은 상태에서
  실질적으로 대표를 고르는 키.
- 추천 60장 평균 technical_pct 75. prior 절반이 이 값이므로 결과에 직접 작용.

### 2. LAION Aesthetic v2 — 유효하지만 원점수는 뭉쳐 있음

- 원점수 5.04~6.37, IQR 0.31 — AVA 1~10 척도의 13%만 쓴다. 웨딩 스튜디오 사진은 미학
  분포가 좁아서 절대값은 정보가 없고 순위 백분위만 의미가 있다(설계대로).
- ARNIQA 백분위와의 상관 −0.05 → 두 축이 독립. 합쳐서 prior를 만드는 근거는 성립.
- 추천 60장 평균 aesthetic_pct 80.
- 주의: `docs/paper-digest.md`가 경고한 대로 "SAC/AI-art 편향 — 수백 장 후보 세밀 순위엔
  약함". IQR이 이렇게 좁으면 백분위 차 5pt(형제 탈락 사유 임계)가 원점수 0.05 차이 —
  노이즈 수준일 가능성. 형제 탈락 사유 "인상이 약함"의 신뢰도는 낮게 봐야 한다.

### 3. DINOv2 임베딩(embedder) — 유효

- 인접 8장 내 코사인: p25 0.744 / p50 0.887 / p75 0.951 / p90 0.974. 임계 0.96은
  p75~p90 사이 — 버스트만 묶고 장면 전체는 안 묶는 위치. 301 클러스터, 단독 152, 최대 21장.
- `cluster_threshold=0.96`은 CLIP으로 맞춘 값인데 DINOv2 분포에서도 우연히 비슷한 위치에
  있다. 그래도 DINOv2 기준으로 재확정하고 주석을 고쳐야 한다(`config.py:90-93`).

### 4. MediaPipe Face Landmarker — 사실상 무효

- `face_count=0` 668/822(81%). VLM이 `couple`로 본 475장 중 440장이 0.
- 프레이밍과 정확히 상관: `full` 569/596 = 0, `half` 96/213 = 0, `closeup` 0/10 = 0.
- 원인(실측, `scripts/spike/face_probe.py`): Face Landmarker 내부 검출기가 BlazeFace
  **short-range**(128×128 입력, 셀카 거리용). 전신 컷의 얼굴 폭은 프레임의 7~10% → 128px
  안에서 ~9px. 원본 4608px로 올려도 0장 — 해상도 문제가 아님.
  - 풀레인지 검출기(`mp.solutions.face_detection(model_selection=1)`): 같은 사진 전부 검출.
  - 무작위 60장: 현재 방식 0장 = 49, 풀레인지 ≥1장 = 47, 풀레인지→얼굴 크롭→Landmarker로
    eyes_open 확보 = 41 (18% → 68%).
- 결과 영향: 대표 선정 키 1순위(눈 뜸)가 81%에서 0으로 무력화. `rank_reason_code='eyes_open'`
  대표 5/301, 추천 근거의 "눈 감김" 언급 1/60.
- 검출된 154장에서도 eyes_open 평균 0.65, 0.5 미만 37장(24%) — 작은 얼굴의 blendshape는
  노이즈가 크다(`faces.py:4-6` 자체 경고: "거울 반전만으로 0.6 흔들림"). 검출을 고쳐도
  절대 임계가 아니라 버스트 내 상대 비교로만 써야 한다.
- 대안 신호: VLM `expression=eyes_closed`(정밀도 0.90)가 이 갤러리에서 4장을 잡았고, 그
  4장은 모두 MediaPipe가 얼굴을 찾은 사진이다. 얼굴 모델 없이도 "눈 감김 회피"의 최소
  기능은 VLM 태그로 대체 가능.

### 5. MediaPipe `smile`, `max_face_ratio` — dead data

- `grep` 결과 `analyze/job.py`에서 저장하는 것 외에 읽는 코드가 없다.
- `smile` 평균 0.28인데 VLM은 636장을 `smile`로 태그 — 두 신호가 안 맞고, 어느 쪽도
  검증되지 않았다. 지금은 계산·저장 비용만 든다.

### 6. VLM `scene` — 유효하나 라벨 오류

- 분포: snap 292 / prep 185 / walk 174 / detail 82 / unknown 36 / bouquet 28 / kiss 18 /
  vow 5 / family 1 / entrance 1.
- 의심 라벨:
  - `detail` 82장 **전부** subjects가 couple/bride/groom — 디테일 컷(반지·소품)이 아니라
    인물 컷. 샘플 확인: "창가에 앉아 웨딩드레스를 입은 신부" 캡션이 `detail`.
  - `walk` 174장 — 이 갤러리는 스튜디오 촬영(흰 호리존, 해변 배경 세트). "행진" 장면이
    21%일 수 없다. "걸어가는 모습"을 walk로 매핑한 것으로 보임.
  - `family` 1 / `entrance` 1 / `vow` 5 — 스튜디오에 없는 장면이 소수 섞임.
- 그런데도 커버리지 쿼터는 이 라벨 위에서 돈다: 추천 30장이 snap 8 / prep 6 / walk 5 /
  detail 3 / … 로 분배됨. 라벨이 틀려도 "임베딩과 다른 기준의 다양성"은 확보하지만,
  근거 문장에 "행진 장면은 6장이면 충분해요"처럼 **틀린 장면 이름이 그대로 노출**된다
  (2라운드 rank 2, 4, 7). 정밀도 0.74로 근거 인용이 금지된 축인데 coverage 사유 경로
  (`draft/job.py:112-115`)는 이 게이트를 안 거친다 — 코드 결함.
- 축 어휘가 "본식" 기준(entrance·vow·ring·walk)이라 스튜디오/야외 스냅 갤러리에는 맞는
  값이 없다. 갤러리 유형별 어휘 또는 `studio`·`outdoor` 값 추가가 필요.

### 7. VLM `lighting` — 이 갤러리에서 무의미

- `natural` 716(87%) / `indoor` 94. 실내 스튜디오 세트에서 `natural`이 87%면 축이
  "밝고 부드러운 조명"을 natural로 읽는 것. 스튜디오 조명(flash/indoor)을 못 가른다.
- 소비처는 취향 피처뿐이고 정밀도 0.84로 근거 인용도 불가. 피처로 쓰여도 87%가 같은
  값이면 BT 회귀에 정보가 없다.

### 8. VLM `expression`, `subjects`, `framing` — 유효하지만 아직 소비자가 없음

- `subjects` 분포(couple 478 / bride 161 / groom 182 / friends 1)와 `framing`은 얼굴
  검출 결과·캡션과 일관 — 믿을 만하다.
- `expression`은 smile 77% 편중. 취향 피처로서 정보량이 낮다.
- 세 축의 소비처는 ① BT 취향 학습 ② 근거 인용 ③ rarity(subjects×expression 5% 이하).
  ①②는 `evidence {pair:0, rating:0, selection:0}`(두 라운드 모두)이라 아직 한 번도
  작동하지 않았다. 담은 사진 11장은 잡 #32(PENDING) 이후에 반영된다.

### 9. VLM `caption` — 비용 대비 낮음

- 유니크 513/822. "신부가 부케를 들고 미소짓고 있습니다." 22회, "신랑이 의자에 앉아
  정면을 응시하고 있다." 22회 — 버스트 안에서는 사실상 같은 문장.
- 프롬프트가 금지한 칭찬어("아름답다"·"인상적") 13건.
- 소비처는 LLM 근거 다듬기의 `moment` 재료뿐. Lambda 경로(LLM 없음)에선 아예 안 읽힌다.

### 10. CLIP 임베딩 — 폐기되는 계산

- `laion.embed()`가 사진당 CLIP ViT-L/14 768d를 계산하지만 DB 모드에선 DINOv2가 있어
  버린다(`analyze/job.py:124`). LAION 점수용이라 계산 자체는 필요.
- 이 벡터로 5축 zero-shot 태깅이 가능하다 — VLM(5,468초)의 저비용 대체 후보. repo에
  실험 없음.

## 권장 조치 (우선순위)

1. **`scene` coverage 사유에 정밀도 게이트 적용** — `draft/job.py:112-115`가
   `reason_min_precision`을 무시해 틀린 장면 이름이 고객에게 노출됨. 코드 한 줄.
2. **얼굴 검출 2단계화 또는 제거 결정** — 풀레인지 검출 → 크롭 → Landmarker(실측 18%→68%).
   또는 얼굴 모델을 빼고 `expression=eyes_closed`로 대체. 어느 쪽이든 `smile`·
   `max_face_ratio` 계산·저장은 중단.
3. **CLIP zero-shot vs VLM 정밀도 비교 스파이크** — 같은 50장 채점셋. subjects·framing이
   CLIP으로 충분하면 VLM은 scene·caption만 남기거나 제거. 런타임 96% 절감 후보.
4. **scene 어휘를 갤러리 유형에 맞게** — `studio`/`outdoor` 추가 또는 유형별 어휘.
   `detail` 정의를 프롬프트에서 "인물 없음"으로 못 박기(단, 프롬프트 수정 시 정확도
   재측정 필수 — `vlm.py:6-11`).
5. `lighting` 축은 취향 피처에서 제외 검토(정보량 없음). `cluster_threshold`를 DINOv2
   기준으로 재확정.

## 재현 SQL

```sql
-- 얼굴 검출 결손
SELECT framing, face_boxes->>'face_count' fc, count(*) FROM photos p
JOIN photo_analysis a ON a.photo_id=p.id WHERE p.gallery_id=1 GROUP BY 1,2 ORDER BY 1,2;

-- 원점수 분포·독립성
SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY (sub_scores->>'aesthetic_score')::float),
       percentile_cont(0.75) WITHIN GROUP (ORDER BY (sub_scores->>'aesthetic_score')::float),
       corr(technical_pct, aesthetic_pct)
FROM photos p JOIN photo_analysis a ON a.photo_id=p.id WHERE p.gallery_id=1;

-- 장면 라벨 의심
SELECT scene, subjects, count(*) FROM photos p JOIN photo_analysis a ON a.photo_id=p.id
WHERE p.gallery_id=1 AND scene IN ('detail','walk') GROUP BY 1,2 ORDER BY 1,3 DESC;

-- 추천에서 실제 쓰인 사유
SELECT round, score_breakdown->>'primary_reason', count(*) FROM ai_recommendations
WHERE selection_id=17 GROUP BY 1,2 ORDER BY 1,3 DESC;
```

얼굴 검출 실측 스크립트: `scripts/spike/face_probe.py` (해상도별 Landmarker vs BlazeFace 비교)
(`scripts/spike/.venv`, mediapipe 0.10.14, 무작위 60장 seed 0).
