# photoselect 기술 스택 — 모델·인프라·서버

`plan.md`(2026-08-22판)·`feature-design.md`를 구현하는 ML 모델, 패키지, 인프라, 서버 구성의
확정안. 2026-08-19판에서 바뀐 핵심: **전수 분석이 Lambda CPU → GPU EC2 온디맨드**, **VLM
자체 호스팅 추가**, Bedrock은 텍스트 전용.

---

## 1. 전체 그림 — 이미지 하나, 실행 모양 셋

```
photoselect 컨테이너 이미지 (단일, GPU/CPU 공용 베이스)
 ├─ A analyze : 전수 분석     GPU EC2 온디맨드 · DB 큐(ai_analysis_jobs) 소비  ← 갤러리당 1회
 ├─ B draft   : 초안·재계산   Lambda EVENT {"selectionId","jobId","mode"}       ← 요청당
 └─ C reasons : LLM 표현·번역 Lambda (B에서 호출)                                ← 요청당
```

- wes(Kotlin/Spring)가 실행 조건 검증·트리거·인증·EC2 start를 소유. 이 서버는 HTTP를
  외부에 노출하지 않는다.
- B·C는 ML 추론 없음 → 같은 코드베이스지만 Lambda 이미지는 torch 없이 슬림 빌드(멀티스테이지).

## 2. ML 모델 스택

| 단계 | 모델 | 패키지 | 조달 | 라이선스 | 실행 |
|---|---|---|---|---|---|
| A-1 얼굴/눈/미소 | MediaPipe Face Landmarker | `mediapipe` | `.task` 번들 | Apache-2.0 | GPU EC2 (CPU 추론) |
| A-2 기술 품질 | ARNIQA | `torch` + vendored | 고정 커밋 가중치 번들 | Apache-2.0 | GPU |
| A-3 미학 | LAION Aesthetic v2 (CLIP ViT-L/14 + MLP) | `open_clip_torch` | 번들 | Apache-2.0 | GPU |
| A-4/5 **태그·캡션** | **오픈 VLM — 후보: Gemma 3 12B, Qwen2.5-VL 7B** (스파이크 후 확정; 31B 비교) | **`vllm`** | HF 가중치를 AMI에 굽기 | Gemma ToU / Apache-2.0 — 채택 전 원문 확인 | GPU (vLLM 배치, 이미지당 ≤0.3s 목표) |
| A-6 클러스터 | pgvector(DINOv2 768d) union-find | `psycopg` | — | — | CPU |
| A-7 대표 선정 | 결정적 규칙 | `numpy` | — | — | CPU |
| B 선호·점수·MMR | 결정적 코드 | `numpy`, pgvector 쿼리 | — | — | Lambda |
| C 이유·번역 | Claude Haiku 4.5 (Bedrock, **텍스트만**) | `anthropic[bedrock]` | API | — | Lambda |

- 런타임 다운로드 금지. VLM 가중치(수십 GB)는 AMI EBS에 미리 두고, 경량 3종은 이미지 번들.
- `MODEL_VERSIONS` 상수 → `photo_analysis.model_version`. VLM 교체는 전수 재적재 대상.
- VLM 출력은 structured output(vLLM guided decoding, enum 강제) → 고정 축 어휘 계약 유지.

## 3. Python 스택

| 용도 | 패키지 |
|---|---|
| 런타임 | Python 3.12 |
| 추론 (A) | `torch`(CUDA 휠), `vllm`, `open_clip_torch`, `mediapipe`, `numpy`, `pillow` |
| DB | `psycopg[binary]`, `pgvector` |
| AWS | `boto3` |
| LLM (C) | `anthropic[bedrock]` — `AnthropicBedrockMantle(aws_region="ap-northeast-2")` |
| 테스트 | `pytest` |

requirements는 `requirements-gpu.txt`(A) / `requirements.txt`(B·C)로 분리, 전부 핀 고정.

## 4. 인프라 (Terraform — `../organic-agent-infra`)

| 리소스 | 구성 | 비고 |
|---|---|---|
| **EC2 GPU (A)** | **g6.xlarge (L4 24GB, ~$0.8/h)** 로 시작. 31B 채택 시 g6e.xlarge(L40S 48GB, ~$1.9/h). 평시 **stopped** | AMI: CUDA + vLLM + 가중치. systemd 서비스가 부팅 시 `analyze` 루프 실행 |
| 기동/정지 | wes가 `ec2:StartInstances` → 인스턴스가 잡 없으면 `shutdown` (self stop) | 기동 2~3분 + 모델 로드 1~2분. AMI 가중치로 로드 단축 |
| 잡 큐 | `ai_analysis_jobs` 테이블 (큐 서비스 없음) | RUNNING 타임아웃(20분) → PENDING 재큐 |
| Lambda (B) | 슬림 이미지, 1~2GB, timeout 1분 | 초안 30초·재계산 5초 목표 |
| Lambda (C) | B와 동일 이미지·핸들러 분기 | Bedrock 호출 |
| S3 | previews 읽기 전용 | 원본·HEIC 접근 없음 |
| RDS | EC2·Lambda 인바운드 보안그룹 | 스키마는 wes Flyway |
| Parameter Store | DB 접속 등 | `config.py` 단일 통로 |
| IAM | EC2: `s3:GetObject`, `ssm:GetParameter`, `ec2:StopInstances`(self) · Lambda: `bedrock:InvokeModel`, `ssm` · wes: `ec2:StartInstances` | 최소 권한 |
| CloudWatch | 잡당 처리량·VLM 이미지/초·기동 시간 | 10분 예산 알람 |

**10분 예산 초과 시 대응 순서**: ① VLM 배치 크기·해상도 조정 → ② 12B→7B → ③ AMI 가중치·
인스턴스 warm pool로 기동 단축 → ④ 소마 기간 한정 상시 가동 → ⑤ 갤러리 분할(멱등 재개).

## 5. LLM (Bedrock) 구성

| 용도 | 모델 ID | 비고 |
|---|---|---|
| C 이유 문장 일괄 (1차 초안 시 1회) | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | 텍스트만, structured output, 100장 ≈ 수천 토큰 |
| C 자연어 피드백 → `{axis, tag, delta}` | 동일 | 2차 셀렉 5초 예산 내 (~1초) |

- 리전: ap-northeast-2 온디맨드에 Claude 5/Haiku 4.5 없음 → **`global.` 크로스 리전 프로필**
  (2026-08-20 확인). **이미지는 보내지 않으므로** 국외 라우팅 무방 — 이미지 처리는 전부
  자체 GPU에서.
- structured outputs 필수, 시스템 프롬프트(톤 규칙·축 어휘) prompt caching, 재시도 멱등.

## 6. 서버/모듈 구조

```
photoselect/
├── handler.py           # Lambda 엔트리 (draft / refine / reasons)
├── __main__.py          # 로컬 CLI: python -m photoselect analyze|draft|refine ...
├── analyze/             # A — GPU EC2
│   ├── loop.py          #   ai_analysis_jobs 소비 루프 + self stop
│   ├── faces.py  quality.py  aesthetic.py
│   ├── vlm.py           #   vLLM 태그+캡션 (enum 강제, 후처리 매핑)
│   └── cluster.py       #   union-find + 대표 선정
├── draft/               # B — Lambda
│   ├── evidence.py      #   별점·선택·쌍·👍/👎 집계 → 선호 분포·벡터·λ
│   ├── pairs.py         #   온보딩 쌍 생성
│   ├── score.py         #   prior + 취향 + 커버리지 + MMR 그리디
│   └── reasons_tpl.py   #   템플릿 이유
├── llm/                 # C
│   ├── bedrock.py       #   Mantle 래퍼, structured outputs
│   ├── reasons.py       #   일괄 문장화 + photo_id 가드레일
│   └── feedback.py      #   자연어 → {axis, tag, delta}
├── job.py  db.py  config.py  vocab.py   # vocab = 고정 축 enum 단일 소스
├── models/  tests/  Dockerfile  Dockerfile.gpu  requirements*.txt
```

- `db.py`가 접근 규칙을 강제: `photo_selections.status` 미접근, `photo_selection_items`
  **읽기 전용**, `ai_recommendations`·`pair_comparison_events`·`photo_analysis`만 쓰기.

## 7. 로컬 개발

```bash
../organic-agent-server/wes/scripts/db-tunnel.sh
python -m photoselect analyze --gallery-id 1 --job-id 1 --device mps   # Mac: 소형 VLM으로 스모크
python -m photoselect draft   --selection-id 1 --job-id 2
python -m photoselect refine  --selection-id 1 --job-id 3 --feedback "가족 사진 더"
pytest
```

- 트랙 B 실험(쌍 비교·홀드아웃 정확도)은 `scripts/spike/` 하네스 — GPU 없이 실행.
- LLM 테스트는 recorded fixture.

## 8. 스파이크 확인 목록

1. VLM 처리량: 12B/7B + vLLM, 1,000장 ≤ 6분 (L4)
2. 12B vs 31B 고정 축 정확도·캡션 품질 (100장, 팀 검수)
3. EC2 기동→모델 로드→첫 추론까지 시간 (AMI 가중치 전후)
4. 홀드아웃 쌍 정확도 첫 수치 (사람 10명 × 120쌍 — `plan.md` §6-A)
5. VLM 라이선스 원문 (Gemma ToU 상업 조건)
