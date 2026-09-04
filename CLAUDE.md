# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`organic-agent-ai`는 웨딩 사진 셀렉 서비스(wes)의 **AI 서버**다. 부부의 셀렉·보정 요청·앨범
구성 과정에서 작가의 업무 시간을 줄이는 AI 기능을 담당한다. AI는 초안·구조화·제안만 하고,
최종 결정(사진 선택·제출·보정 확정)은 언제나 사람이 한다.

> 제품 우선순위·아키텍처·설계는 `docs/plan.md`가 단일 소스다. **현재 구성**(모듈·잡·계약)은
> `docs/embedder-photoselect-architecture.md`에 있다. 이 문서는 요약과 작업 규칙만 담는다.

## 제품 우선순위 (인터뷰 3건 기반 — 근거는 docs/plan.md §1)

결제 주체는 작가/스튜디오이고, 결제 근거는 **작가 업무 시간 절감**이다. 이 축으로 정렬:

1. **P1 앨범 미리보기·에디터** — 본체는 웹 에디터(wes/프론트). AI 몫은 자동 레이아웃 초안(배치)과
   크롭 안전 감지용 데이터(바운딩 박스 — 런타임 AI 호출 없음)
2. **P2 보정 요청 구조화** — 모호한 보정 요청을 작가가 등록한 가능/불가 메뉴에 대조해 구조화된
   요청(사진·부위·강도·참고)으로 변환. **첫 번째 진짜 LLM 기능**
3. **P3 AI 1차 셀렉 초안** — 목표 장수까지 초안 채우기. 품질 점수로 하드 필터링 금지
   (작가가 의도한 흔들림·눈감김 컷 보호)

공통 인프라: embedder 확장으로 사진별 얼굴/피사체 바운딩 박스 + 장면 태그를 갤러리당 1회 적재.
**원가 원칙 — 갤러리 전수 처리는 경량 모델, LLM은 사용자 요청 단위 소규모 호출에만.**

## 시스템에서의 위치

메인 백엔드는 sibling repo `../organic-agent-server/wes` (Kotlin/Spring)다. 실행 모양은 두 가지:

| 워크로드 | 모양 | 예 |
|---|---|---|
| 배치 (갤러리/앨범당 1회) | embedder 패턴 Lambda — `handler.py`(EVENT) / `__main__.py`(로컬 CLI) 동일 코드 | 임베딩·미리보기, 사진 분석·폴더화, 자동 레이아웃 초안 |
| 사용자 기능 (저장된 숫자 + 미리보기 몇 장 + LLM) | **wes가 직접** (Kotlin, Bedrock) — 이 repo에 없다 | 폴더별 추천, 비교샷, 보정 요청 구조화 |

경계는 **"갤러리 전수에 torch 모델 추론이 필요한가"**다. 그렇다면 이 repo, 아니면 wes
(wes `docs/plans/ai-feature-migration-to-wes.md`, 2026-09-04). wes도 미리보기를 읽어 Bedrock에 보낸다.

- wes 백엔드가 실행 조건을 검증하고 트리거한다. 배치는 `InvocationType.EVENT`, 이벤트는
  `{"galleryId": N, "jobId": M}`.
- **공유 Postgres(RDS)를 직접 읽고 쓴다.** HTTP payload로 데이터를 나르지 않는다.
- 이미지는 원본이 아니라 embedder가 만든 **미리보기 파생본**(`photos.preview_key`, EXIF
  회전·리사이즈 JPEG)을 S3에서 읽는다. HEIC 디코드는 이 서버의 일이 아니다.

## 기술 스택 (확정)

- Python 3.12, embedder와 같은 모듈 구조(`handler.py` / `__main__.py` / `job.py` / `db.py` /
  `config.py` + `bedrock.py`)
- `anthropic[bedrock]` — `AnthropicBedrockMantle(aws_region="ap-northeast-2")`,
  모델 ID는 `anthropic.` 접두사 (예: `anthropic.claude-opus-5`)
- `psycopg[binary]` + pgvector, `boto3` (S3·Parameter Store)
- 배포: ECR 컨테이너 이미지 Lambda, Terraform은 `../organic-agent-infra`

## DB 계약 — 스키마는 wes가 소유한다

**이 repo는 마이그레이션을 만들지 않는다.** 모든 스키마 변경은 wes의 Flyway
(`../organic-agent-server/wes/src/main/resources/db/migration/`, 규칙은 그 repo의
`.claude/rules/migration.md`)에서 한다. 기능별 계약 테이블 목록은 `docs/plan.md` §4.

지켜야 할 접근 규칙:

- `photo_selections.status`는 절대 바꾸지 않는다 — 제출은 부부만, wes API로만
- `photo_selection_items`는 `source='AI'` 행만 쓴다. MANUAL 행은 읽기 전용
- `photo_ratings`는 **접근 금지** — 개인 취향 신호, 정책상 AI 입력에서 제외

이슈·브랜치·커밋·PR 규칙은 `.claude/rules/git-workflow.md` (모듈 prefix + organic-agent-server 방식).

## 빌드 & 실행

두 모듈 모두 embedder 패턴이다 — 패키지 바로 아래에 실행 코드, `__main__.py`(로컬 CLI)와 `handler.py`/`worker.py`(배포)가
같은 `run()`을 부른다.

```bash
# embedder — 갤러리당 1회: 미리보기 PUT → DINOv3 → photo_analysis
cd embedder && python -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m embedder --gallery-id 1 [--force]
.venv/bin/python -m pytest tests -q                         # 34

# photoselect — 폴더화: SCORE(사진별 점수, torch) → CATEGORIZE(그룹·이름, torch 없음)
cd photoselect && python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps
.venv/bin/python -m photoselect analyze --db --gallery 12 --llm   # = score → categorize (wes FULL)
.venv/bin/python -m photoselect worker --llm                       # ai_analysis_jobs 폴링
.venv/bin/python -m pytest tests -q                                 # 24
```

로컬 E2E는 wes 쪽 스크립트가 감싼다: `../organic-agent-server/wes/scripts/local-worker.sh --llm`(워커),
`local-ai.sh <galleryId>`(임베딩 → 분석 한 번에). RDS는 퍼블릭 접근이 없다. 직접 붙을 때는 wes의
`scripts/db-tunnel.sh`로 SSM 포트 포워딩을 연다(기본 15432).

## 구현 순서와 리스크

`docs/plan.md` §6(순서)·§7(리스크)을 따른다. 최우선 확인 사항: ap-northeast-2 Bedrock 모델
가용성(미제공이면 크로스 리전 프로필 `apac.anthropic.…`), 배치 Lambda 15분 제한 실측.

## 참조 저장소 (sibling)

- `../organic-agent-server/wes` — 메인 백엔드. 도메인 규칙·컨벤션은 그 repo의 `CLAUDE.md`와
  `.claude/rules/`를 따른다. embedder는 그 repo에서 이 repo의 `embedder/` 모듈로 이관됐다(#20) —
  배치 구조(handler/__main__ 동일 코드)의 원형이자 photoselect의 선행 단계(preview·DINOv3 적재)다.
- `../organic-agent-test-web` — 프론트 (P1 앨범 에디터의 본체)
- `../organic-agent-infra` — Terraform (VPC/RDS/S3/Lambda)
- `../dataset` — 평가용 사진 데이터
