# 모델 스파이크 하네스 (#1)

`../dataset` 골든셋으로 단계별 후보 모델을 비교·실측하는 로컬 전용 스크립트.
**서비스 코드가 아니다** — 여기서는 편의상 런타임 다운로드(고정 버전)를 허용하고,
결과가 확정되면 채택 모델만 서비스 이미지에 번들한다 (tech-stack.md §2 원칙).

## 설치

```bash
cd photoselect/scripts/spike
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 사용 흐름

```bash
# 1. 매니페스트 생성 — 데이터셋 스캔 → photo_id,path,group,selected(빈칸) CSV
python manifest.py build --root ../../../../dataset --out out/manifest.csv

# 2. (수동) 골든셋 라벨 — 작가 최종 셀렉에 해당하는 행의 selected를 1로 채운다

# 3. 점수 실행 — 러너별 점수 + 사진당 처리 시간 적재 (중단 후 재실행 시 이어서)
python run.py --manifest out/manifest.csv --runners faces,laion_aesthetic,arniqa \
              --out out/scores.csv --limit 200

# 4. 리포트 — 라벨 대비 분리도(AUC)·recall@K·러너별 속도 요약
python report.py --manifest out/manifest.csv --scores out/scores.csv

# Bedrock ap-northeast-2 가용성 확인 (AWS 자격 증명 필요)
python bedrock_check.py
```

## 러너

| 러너 | 단계 | 모델 | 조달 (고정 버전) |
|---|---|---|---|
| `faces` | ③ 얼굴 신호 | MediaPipe Face Landmarker | `.task` v1 고정 URL → `weights/` |
| `arniqa` | ① 기술 품질 | ARNIQA (ResNet-50) | torch.hub `miccunifi/ARNIQA` 고정 태그 |
| `laion_aesthetic` | ② 미학 | CLIP ViT-L/14 + MLP | open_clip(openai) + MLP `.pth` 고정 커밋 |

추가 후보(HyperIQA·LIQE·MUSIQ·Charm·HSEmotion)는 1순위 모델의 골든셋 상관이
부족한 축에만 러너를 추가한다 (plan.md §3).

## 평가 지표 (metrics.py)

골든셋 라벨은 "작가가 최종 셀렉했는가"의 **이진 라벨**이므로 MOS 상관(SRCC) 대신:

- **AUC** — 점수가 선택/미선택을 얼마나 가르는가 (러너별·그룹별)
- **recall@K** — 점수 상위 K장 ∩ 작가 셀렉 (K = 그룹별 실제 셀렉 장수)
- 러너 간 순위 상관(SRCC) — 점수 축들이 서로 얼마나 겹치는지

## 출력

- `out/scores.csv` — photo_id × 러너별 점수·서브 신호·처리 시간(초)
- `out/report.md` — 지표 요약 + 사진당 평균 처리 시간 → tech-stack.md §2 표 갱신 재료
