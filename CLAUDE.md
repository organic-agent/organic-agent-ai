# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`organic-agent-ai`는 웨딩 사진 셀렉 서비스(wes)의 **AI 서버**다. 부부의 셀렉·보정 요청·앨범
구성 과정에서 작가의 업무 시간을 줄이는 AI 기능을 담당한다. AI는 초안·구조화·제안만 하고,
최종 결정(사진 선택·제출·보정 확정)은 언제나 사람이 한다.

> **현재 상태: 설계 확정(2026-08-22), 스파이크 진행 중.** 현행 설계의 단일 소스는
> **`photoselect/docs/plan.md`**, 실행 계획은 `photoselect/docs/roadmap.md`. 루트 `docs/plan.md`는
> 08-17판 제품 배경 기록이다. 이 문서는 요약과 작업 규칙만 담는다.

## 현재 범위 — AI 셀렉터 단일 기능 (마감: 소마 최종점검 2026-11-27)

결제 주체는 작가/스튜디오, 결제 근거는 **작가 업무 시간 절감**. 그러나 AI가 대화하는 상대는
**고객(신랑신부)**이다 — 고객이 빨리·납득하며 골라야 작가 납기와 클레임이 줄기 때문.

- **AI 셀렉터**: 정답을 맞추는 기능이 아니라 **설득해서 빨리 고르게 하는** 기능. 고객 취향
  (별점·선택·온보딩 쌍 비교)으로 추천+이유를 제시하고, 👍/👎·자연어 피드백으로 재추천(2차 셀렉).
- 점수 = **사진학 prior + 신뢰도(λ) 가중 취향** + 장면 커버리지 + MMR. 취향 증거가 적거나
  애매하면 자동으로 사진학이 우선. 품질 점수로 **하드 필터링 금지**.
- **보류**: P1 앨범 에디터, P2 보정 요청 구조화, 사진 진단 — 최종점검 이후.

공통 인프라: 업로드 직후 갤러리당 1회 전수 분석 — 경량 3종(MediaPipe·ARNIQA·LAION) +
**오픈 VLM 고정 축 태그·캡션** + pgvector 클러스터. **원가·데이터 원칙 — 전수 처리는 자체 GPU
(이미지 외부 전송 없음), Bedrock LLM은 텍스트 전용 소규모 호출(이유 문장·피드백 번역)에만.**

## 시스템에서의 위치

메인 백엔드는 sibling repo `../organic-agent-server/wes` (Kotlin/Spring)다. 실행 모양은 세 가지:

| 워크로드 | 모양 | 예 |
|---|---|---|
| A 전수 분석 (갤러리당 1회) | **GPU EC2 온디맨드** — wes가 start, 인스턴스가 `ai_analysis_jobs`(DB가 큐)를 소비 후 self stop | 경량 점수·VLM 태그·클러스터 |
| B 초안·재계산 (요청당) | Lambda EVENT — `handler.py` / `__main__.py` 동일 코드 | 1차 초안 30초, 2차 재계산 5초 |
| C LLM (요청당) | Lambda, Bedrock 텍스트 호출 | 이유 문장 일괄, 자연어 피드백 번역 |

- wes 백엔드가 실행 조건을 검증하고 트리거한다. B 이벤트는
  `{"selectionId": N, "jobId": M, "mode": "draft"|"refine"}`.
- **공유 Postgres(RDS)를 직접 읽고 쓴다.** HTTP payload로 데이터를 나르지 않는다.
- 이미지는 원본이 아니라 embedder가 만든 **미리보기 파생본**(`photos.preview_key`, EXIF
  회전·리사이즈 JPEG)을 S3에서 읽는다. HEIC 디코드는 이 서버의 일이 아니다.

## 기술 스택 (확정)

- Python 3.12, embedder와 같은 모듈 구조(`handler.py` / `__main__.py` / `job.py` / `db.py` /
  `config.py`) + `analyze/`(A) `draft/`(B) `llm/`(C) — 상세 `photoselect/docs/tech-stack.md`
- VLM: 오픈 모델(Gemma 3 12B / Qwen2.5-VL 7B 후보, 31B 비교 후 확정) + `vllm`, g6(L4) EC2
- `anthropic[bedrock]` — `AnthropicBedrockMantle(aws_region="ap-northeast-2")`, 서울 온디맨드에
  Claude 5/Haiku 4.5 없음 → **`global.` 크로스 리전 프로필** (텍스트만 보내므로 무방)
- `psycopg[binary]` + pgvector, `boto3` (S3·Parameter Store·EC2)
- 배포: ECR 이미지(Lambda 슬림 / GPU 풀), AMI(가중치 포함), Terraform은 `../organic-agent-infra`

## DB 계약 — 스키마는 wes가 소유한다

**이 repo는 마이그레이션을 만들지 않는다.** 모든 스키마 변경은 wes의 Flyway
(`../organic-agent-server/wes/src/main/resources/db/migration/`, 규칙은 그 repo의
`.claude/rules/migration.md`)에서 한다. 기능별 계약 테이블 목록은 `photoselect/docs/plan.md` §5.

지켜야 할 접근 규칙:

- `photo_selections.status`는 절대 바꾸지 않는다 — 제출은 부부만, wes API로만
- `photo_selection_items`는 **읽기 전용**. AI 추천은 `ai_recommendations`에만 쓰고, 고객이
  담은 것만 wes가 MANUAL로 승격한다
- `photo_ratings`는 **읽기 허용** (2026-08-22 정책 변경) — "AI 추천" 클릭 = 동의, UI 고지 전제.
  쓰기 금지
- 이미지(미리보기 포함)를 외부 API로 보내지 않는다 — VLM은 자체 GPU, Bedrock은 텍스트만

파이프라인 불변식은 `.claude/rules/pipeline.md`, Bedrock 호출 규칙은
`.claude/rules/bedrock.md`가 강제한다. 이슈·브랜치·커밋·PR 규칙은
`.claude/rules/git-workflow.md` (모듈 prefix + organic-agent-server 방식).

## 빌드 & 실행 (스캐폴드 후 이 형태를 유지한다)

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m photoselect analyze --gallery-id 1 --job-id 1     # A 로컬 실행 (RDS 터널 필요)
python -m photoselect draft --selection-id 1 --job-id 2     # B
pytest                                           # 테스트
```

RDS는 퍼블릭 접근이 없다. 로컬 실행 전 wes의 스크립트로 SSM 포트 포워딩을 연다:
`../organic-agent-server/wes/scripts/db-tunnel.sh` (기본 15432).

## 구현 순서와 리스크

`photoselect/docs/roadmap.md`(순서)·`photoselect/docs/plan.md` §8(리스크)을 따른다. 최우선:
**홀드아웃 쌍 예측 정확도 첫 수치(2주 내, GPU 불필요)** 와 VLM 처리량(1,000장 ≤ 6분) 실측.
평가는 정답 라벨이 아니라 쌍 비교 정확도·수락률·nDCG·후회율로 한다.

## 참조 저장소 (sibling)

- `../organic-agent-server/wes` — 메인 백엔드. 도메인 규칙·컨벤션은 그 repo의 `CLAUDE.md`와
  `.claude/rules/`를 따른다. embedder가 이 서버의 배치 구조 원형이다.
- `../organic-agent-test-web` — 프론트 (P1 앨범 에디터의 본체)
- `../organic-agent-infra` — Terraform (VPC/RDS/S3/Lambda)
- `../dataset` — 평가용 사진 데이터
