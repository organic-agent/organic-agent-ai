# 모델 스파이크 리포트 (#1) — 진행 중

하네스: `photoselect/scripts/spike/` (사용법은 그 README). 이 문서는 실측이 쌓일 때마다
갱신하고, 완료 시 tech-stack.md §2 표를 확정치로 바꾼다.

## 확정된 것

### Bedrock ap-northeast-2 가용성 (2026-08-20, `bedrock_check.py`)

plan.md §6 최우선 확인 사항 해소:

- **opus-5·haiku-4.5는 온디맨드 미제공.** `apac.` 프로필에도 없다.
- **`global.` 크로스 리전 프로필(ACTIVE)로 호출 가능**: `global.anthropic.claude-opus-5`,
  `global.anthropic.claude-haiku-4-5-20251001-v1:0` → tech-stack.md §5 반영 완료.
- 온디맨드는 claude-3.5-sonnet(2024-06)·claude-3-haiku 등 구세대뿐.
- 후속 확인: `global.` 라우팅은 해외 리전 처리 가능 — 미리보기 이미지를 보내는 B-2(진단)의
  데이터 위치 정책 검토 필요. A-5는 텍스트 신호만 보내므로 무방.

### 1차 스모크 실측 (M4 Pro CPU, 미리보기 1600px 가정, dataset2 10장)

| 러너 | 단계 | 평균 s/장 | 3,000장 직렬 환산 |
|---|---|---|---|
| MediaPipe faces | ③ | 0.127 | 6.4분 |
| LAION aesthetic v2 | ② | 0.199 | 9.9분 |
| ARNIQA | ① | 0.369 | 18.4분 |
| **합계** | | **~0.70** | **~35분** |

- **직렬로는 15분 예산 초과 확정** — 프로세스 풀 병렬화가 전제다 (tech-stack.md §4의
  예상과 일치). Lambda 10GB(≈6 vCPU)에서 재실측 필요 — M4 Pro와 코어 성능이 다르다.
- 점수 범위 정상: ARNIQA scale_score ≈ 0.49~0.52, LAION ≈ 5.9~6.1 (AVA 1~10 축).
  10장 샘플이라 분산이 작다 — 전수 실행에서 갤러리 내 백분위의 변별력 확인이 본 게임.

## 관찰 (검증 필요)

- **작은 얼굴의 blink 신호 신뢰도**: 2인 컷(얼굴 bbox 비율 ~0.036)에서 eyes_open 최소값이
  0.2대로 낮게 나온다 — 실제 눈 감김인지 blendshape가 작은 얼굴에서 불안정한 건지 확인
  필요. 불안정하면 0-1을 얼굴 크롭 후 재검출로 바꾼다 (원가 상승).

## 남은 작업

- [ ] 골든셋 라벨 — 작가 최종 셀렉을 manifest의 `selected`에 채우기 (라벨 소스 확보가 선결)
- [ ] dataset1/dataset2 전수 실행 → AUC·recall@K (report.py)
- [ ] ①② 후보 확장 실행 — 1순위(ARNIQA·LAION)의 골든셋 상관이 부족할 때만
  (HyperIQA/LIQE/MUSIQ, Charm)
- [ ] HEIC(dataset1 스냅) 경로 검증 — pillow-heif 로드 확인
- [ ] 라이선스 원문 재확인 (채택 직전) + 버스트 셀렉 특허(US 10,671,895) 저촉 검토
- [ ] torch.hub ref를 커밋 SHA로 고정 (`runners/arniqa.py`)
