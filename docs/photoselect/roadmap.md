# photoselect 개발 로드맵 — 단계·산출물·이슈

`plan.md` §6(구현 순서)·`feature-design.md`·`tech-stack.md`를 실행 단위로 편성한 문서.
각 단계는 GitHub 이슈 1개와 1:1이고, 이슈 본문에 태스크 체크리스트와 산출물이 있다.
작업 규칙(이슈·브랜치·커밋·PR)은 `.claude/rules/git-workflow.md`.

## 순서 원칙

1. **스파이크가 관문** — 모델 채택이 확정되기 전에는 파이프라인 본 구현에 들어가지 않는다.
   (도메인 갭이 최대 리스크 — 결과에 따라 ②미학 가중치·모델 자체가 바뀔 수 있다)
2. **LLM은 마지막에 붙인다** — 기능 A는 템플릿 근거로 먼저 완성해 recall@K를 측정하고,
   그 위에 A-5 문장화를 얹는다. 결정적 부분과 LLM 부분의 검증을 분리하기 위함.
3. **wes 마이그레이션 협의는 각 단계 시작 시점에** — 스키마는 wes Flyway 소유라 리드타임이
   있다. 3·4단계 착수 전에 미리 계약 테이블을 합의한다.

## 단계별 계획

| # | 단계 | 이슈 | 핵심 산출물 | 착수 조건 |
|---|---|---|---|---|
| 1 | **모델 스파이크** | [#1](https://github.com/organic-agent/organic-agent-ai/issues/1) | 모델 비교 리포트(상관·속도·라이선스), 단계별 채택 확정 → tech-stack.md 갱신, Bedrock 리전 확정 | 즉시 |
| 2 | **스캐폴드** | [#2](https://github.com/organic-agent/organic-agent-ai/issues/2) | embedder 패턴 모듈 골격, ECR 이미지(<4GB), 로컬 CLI 3종 엔트리, pytest 기본 | #1과 병행 가능 (모델 무관 골격부터) |
| 3 | **점수 파이프라인 (배치 A)** | [#3](https://github.com/organic-agent/organic-agent-ai/issues/3) | `photo_analysis` 전수 적재(0-1~0-6), 골든셋 상관 리포트, 15분 예산 실측 | #1 채택 확정 + #2 + wes `photo_analysis` 마이그레이션 |
| 4 | **추천 채우기 — 결정적 조립 (배치 B)** | [#4](https://github.com/organic-agent/organic-agent-ai/issues/4) | AI 초안(템플릿 근거) 저장, recall@K 리포트, 가드레일 테스트 | #3 + wes `ai_selection_jobs` 마이그레이션 |
| 5 | **LLM 근거 문장화 (A-5)** | [#5](https://github.com/organic-agent/organic-agent-ai/issues/5) | `bedrock.py`(structured outputs·caching), photo_id 가드레일·템플릿 폴백, 검수 결과 | #4 |
| 6 | **평가 리그레션** | [#6](https://github.com/organic-agent/organic-agent-ai/issues/6) | 평가 하네스(recall@K·대표 일치율·커버리지), 목표 수치 확정 | #4 (측정 대상 존재) |
| 7 | **사진 진단 (동기 C)** | [#7](https://github.com/organic-agent/organic-agent-ai/issues/7) | 2~3초 응답 동기 API, wes `POST /diagnose` 계약, UI 배지 계약 | #3 (신호 존재) + #5 (bedrock.py 재사용) |
| 8 | **배포** | [#8](https://github.com/organic-agent/organic-agent-ai/issues/8) | Terraform(Lambda 3종·IAM·알람), wes 트리거 E2E | #3~#7 각 완성 시점마다 증분 배포 가능 |
| 9 | **실시간 재추천 (후속)** | [#9](https://github.com/organic-agent/organic-agent-ai/issues/9) | 재정렬 휴리스틱 실측 리포트 → 검증 후 API | #4·#5 검증 완료 후 |

의존 관계 요약:

```
#1 스파이크 ──┬──▶ #3 점수 파이프라인 ──▶ #4 추천 채우기 ──▶ #5 LLM 근거 ──▶ #9 (후속)
#2 스캐폴드 ──┘            │                    │                  │
                           │                    └──▶ #6 평가       └──▶ #7 진단
                           └────────────────────────────────────────────▶ #8 배포 (증분)
```

## 단계 완료의 정의

이슈의 태스크 체크리스트 완료 + **산출물 섹션의 문서 갱신까지** 끝나야 완료다.
특히 #1·#3의 실측치는 tech-stack.md의 예상치 표를 갱신해야 하고 (예상치 방치 금지),
#6의 목표 수치는 plan.md §6-5에 반영한다.
