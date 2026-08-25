# 모델 스파이크 하네스 (#1)

`../dataset` 골든셋으로 단계별 후보 모델을 비교·실측하는 로컬 전용 스크립트.
**서비스 코드가 아니다** — 여기서는 편의상 런타임 다운로드(고정 버전)를 허용하고,
결과가 확정되면 채택 모델만 서비스 이미지에 번들한다 (tech-stack.md §2 원칙).

> **2026-08-25 — 러너 이관.** `runners/`(faces·laion·arniqa·common)의 정본은 이제
> `photoselect/analyze/runners/`다. 여기 사본은 스파이크 스크립트 호환용으로 남겨 두되
> 고칠 일이 있으면 photoselect 쪽을 고친다. 실제 파이프라인 실행은 `photoselect/README.md`.

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

---

## VLM 고정 축 태그 (A-4/A-5) — 로컬 선행 검증

경량 3종과 달리 VLM은 **품질을 사람이 봐야** 안다. GPU EC2를 세우기 전에 맥북에서
Ollama로 먼저 돌려 **축 어휘 계약이 실제 사진에서 성립하는지**만 확인한다.

```bash
# 1) 로컬 환경 준비 — 기동 + 모델 확인. 몇 번을 돌려도 안전하다
./setup_ollama.sh              # 처음이면 --install, 비교군까지면 --compare
                               # 상태만 보려면 --status, 내리려면 --stop

# 2) 태그 뽑기 — 데이터셋별 균등 표본, 시드 고정(모델 비교 시 같은 사진이 쓰인다)
.venv/bin/python vlm_tag.py --manifest out/manifest.csv --n 50

# 3) 채점 시트 생성 → 브라우저에서 축마다 ✓/✗
.venv/bin/python vlm_review.py --tags out/vlm_tags.csv
open out/vlm_review.html

# 4) 축별 정확도 집계 (채점 시트가 내려준 CSV로)
.venv/bin/python vlm_review.py score --grade ~/Downloads/vlm_grade.csv
```

| 스크립트 | 역할 |
|---|---|
| `setup_ollama.sh` | Ollama 기동·모델 준비. **설치/기동 명령의 정본** — 바꿀 일이 있으면 여기만 고친다 |
| `vlm_tag.py` | Ollama structured outputs로 **enum을 강제**해 5축 태그 + 한국어 캡션. 축별 분포·속도·어휘 커버리지 요약 |
| `vlm_review.py` | 사진과 태그를 나란히 놓은 로컬 HTML 채점 시트 생성 + 축별 정확도 집계 |

**읽을 때의 단서 (반드시 함께 적을 것)**

- 여기 정확도는 **로컬 4bit 실행의 하한**이다. 프로덕션은 L4 + FP16 + vLLM이라 같지 않다.
- 속도는 프로덕션 목표(1,000장 ≤ 6분)와 **직접 비교하지 않는다**. vLLM 연속 배치가 없다.
- enum은 스키마가 강제하므로 `unknown`/`none`은 파싱 실패가 아니라 **모델의 실제 선택**이다.
  이 비율이 높으면 축 어휘를 손볼 신호다.
- 이미지는 로컬에서만 처리된다 — 외부 전송 없음 (CLAUDE.md 데이터 원칙).
