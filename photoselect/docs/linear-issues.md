# Linear 이슈 규격 — photoselect (AI 1차 셀렉)

Linear에 옮겨 적기 위한 규격과, 그 규격으로 작성된 이슈 9건.
GitHub 이슈(#1~#9)·`roadmap.md`와 1:1 대응이다.

---

## 이슈 규격

| 항목 | 규칙 |
|---|---|
| **제목** | GitHub 이슈와 동일: `[photoselect] <타입>: <설명>` — 두 트래커 상호 추적용 |
| **라벨** | 모듈 라벨 `photoselect` + 타입 라벨 `feat` `chore` `test` `cicd` (GitHub 이모지 라벨과 동일 의미) |
| **Priority** | Urgent = 다른 작업의 관문 / High = 주 경로 / Medium = 병행 가능 / Low(No priority) = 후속·backlog |
| **Estimate** | 1·2·3·5·8 (상대 크기) |
| **관계** | 의존은 반드시 **Blocked by**로 걸기 (roadmap.md 의존 그래프 그대로) |
| **설명 본문** | 아래 4개 섹션 고정 |

설명 본문 구조 (GitHub 템플릿의 Linear 축약판):

```
**목적** — 왜 하는가 1~2문장

**태스크**
- [ ] …

**완료 조건 (산출물)** — 이슈가 끝나면 존재해야 하는 것
- …

**링크** — GitHub 이슈·PR·관련 문서
```

권장 구성: 프로젝트 **"photoselect — AI 1차 셀렉"** 하나에 전부 넣고 마일스톤 3개로 나눈다.

- **M1 기반** — 스파이크·스캐폴드 (아래 1·2)
- **M2 코어** — 점수 파이프라인·추천 채우기·LLM 근거·평가 (3·4·5·6)
- **M3 제공** — 사진 진단·배포 (7·8) / 실시간 재추천(9)은 backlog

상태 흐름: `Backlog → Todo → In Progress → In Review(PR 열림) → Done(PR 머지 + 산출물 문서 갱신)`

---

## 이슈 1 — M1 · Urgent · Estimate 5 · `chore`

**제목**: [photoselect] chore: 모델 스파이크 — 후보 모델 골든셋 평가와 채택 확정

**목적** — 모델 채택 확정 전에는 본 구현에 들어가지 않는다. 도메인 갭(웨딩 사진 vs 공개
데이터셋)이 최대 리스크라, 골든셋 상관으로 단계별 모델을 먼저 판정한다.

**태스크**
- [x] 평가 하네스 구축 (`scripts/spike/` — 매니페스트·러너 3종·지표·리포트)
- [x] Bedrock ap-northeast-2 가용성 확인 → `global.` 프로필 확정
- [x] CPU 1차 실측 — 합계 ~0.7s/장, 3,000장 직렬 ~35분 → 병렬화 필수 확정
- [ ] 골든셋 라벨 — 작가 최종 셀렉을 manifest `selected`에 채우기 (라벨 소스 확인 필요)
- [ ] dataset1·dataset2 전수 실행 → AUC·recall@K
- [ ] 상관 부족 축에만 후보 확장 (HyperIQA/LIQE/MUSIQ, Charm)
- [ ] 라이선스 원문 재확인 + torch.hub 커밋 SHA 고정

**완료 조건 (산출물)**
- 단계별 채택 모델 확정 → `docs/tech-stack.md` §2 표를 실측치로 갱신
- 모델 비교 리포트 (`docs/spike-report.md` 완성)

**링크** — GitHub [#1](https://github.com/organic-agent/organic-agent-ai/issues/1) · PR [#10](https://github.com/organic-agent/organic-agent-ai/pull/10)(하네스, 진행 중)

---

## 이슈 2 — M1 · High · Estimate 3 · `chore`

**제목**: [photoselect] chore: embedder 패턴 스캐폴드와 로컬 CLI end-to-end

**목적** — embedder와 같은 모듈 구조를 복제해, 로컬 CLI로 골든셋이 end-to-end 도는 골격을
만든다. 스파이크와 병행 가능 (모델 무관 골격부터).

**태스크**
- [ ] `handler.py`(score/draft/diagnose 분기)·`__main__.py`·`job.py`·`pipeline/`·`db.py`·`config.py`·`bedrock.py`
- [ ] `db.py` 접근 규칙 코드 강제 (status 미접근·AI 행만 쓰기·photo_ratings 쿼리 부재)
- [ ] Dockerfile 가중치 번들(고정 커밋·체크섬) + `MODEL_VERSIONS`
- [ ] requirements 핀 고정, 이미지 < 4GB
- [ ] pytest 기본 + db-tunnel 경유 로컬 실행 확인

**완료 조건 (산출물)**
- `python -m photoselect score --gallery-id 1 --job-id 1` 로컬 동작 (스텁 파이프라인 + 잡 라이프사이클)
- 빌드되는 ECR 이미지, 통과하는 기본 테스트

**링크** — GitHub [#2](https://github.com/organic-agent/organic-agent-ai/issues/2)

---

## 이슈 3 — M2 · High · Estimate 8 · `feat` · **Blocked by: 이슈 1, 이슈 2**

**제목**: [photoselect] feat: 0단계 점수 파이프라인(배치 A) 구현

**목적** — 갤러리당 1회, 사진마다 얼굴·기술·미학·장면·클러스터 신호를 `photo_analysis`에
적재한다. LLM 0회. 이후 모든 기능(추천·진단·레이아웃)의 공통 기반.

**태스크**
- [ ] wes `photo_analysis` Flyway 마이그레이션 협의 (착수 시점에 먼저)
- [ ] 0-1 faces / 0-2 quality / 0-3·0-4 aesthetic+장면 태그 / 0-5·0-6 cluster+순위
- [ ] 갤러리 내 백분위 통일, rank_reason_code 적재
- [ ] 멱등(처리 사진 스킵) + 프로세스 풀 병렬화, 15분 예산 실측 (스파이크 실측상 직렬 ~35분)

**완료 조건 (산출물)**
- Lambda EVENT로 도는 배치 A + `photo_analysis` 전 컬럼 적재 (model_version 포함)
- 골든셋 상관 리포트, 처리량 실측 → tech-stack.md §4 갱신

**링크** — GitHub [#3](https://github.com/organic-agent/organic-agent-ai/issues/3)

---

## 이슈 4 — M2 · High · Estimate 8 · `feat` · **Blocked by: 이슈 3**

**제목**: [photoselect] feat: 추천 채우기(기능 A) — 결정적 초안 조립과 가드레일

**목적** — 목표 장수까지 남은 슬롯을 채우는 초안 생성. 이 이슈는 LLM 없이 A-1~A-4 결정적
조립 + 템플릿 근거까지 — 선택 로직의 recall@K를 LLM과 분리해 검증한다.

**태스크**
- [ ] wes `ai_selection_jobs`·`photo_selection_items.source/ai_reason` 마이그레이션 협의
- [ ] A-1 입력 확정(shortfall) / A-2 취향 부분 센트로이드(콜드스타트 가중치 0) / A-3 커버리지 격자 / A-4 그리디(클러스터 대표 1장, 하드 필터 없음)
- [ ] 선정 근거 구조체 + 템플릿 근거 문장
- [ ] 가드레일: 무효 photo_id 거부·수동 불가침·개수 결정적·자동 재생성 금지, 단일 트랜잭션

**완료 조건 (산출물)**
- 배치 B Lambda (수십 초 완료), `source='AI'` 행 + 근거 저장
- recall@K 리포트와 가중치 초기값, 멱등·가드레일 테스트

**링크** — GitHub [#4](https://github.com/organic-agent/organic-agent-ai/issues/4)

---

## 이슈 5 — M2 · Medium · Estimate 5 · `feat` · **Blocked by: 이슈 4**

**제목**: [photoselect] feat: LLM 근거 문장화(A-5) — structured outputs와 템플릿 폴백

**목적** — 템플릿 근거를 Bedrock Claude(`global.anthropic.claude-opus-5`) 문장으로 승격.
LLM은 문장만 만들고 선택을 바꿀 수 없다.

**태스크**
- [ ] `bedrock.py` — structured outputs·prompt caching·멱등 재시도
- [ ] 근거 프롬프트 (점수 숫자 금지·제안 톤·신호 외 내용 금지)
- [ ] photo_id 검증 → 불일치 거부·누락 템플릿 폴백
- [ ] recorded fixture 테스트 + 내부 검수(신호-문장 불일치율)

**완료 조건 (산출물)**
- LLM 근거 붙은 초안, 폴백 경로 테스트, 검수 결과 기록
- `.claude/rules/bedrock.md` 성문화

**링크** — GitHub [#5](https://github.com/organic-agent/organic-agent-ai/issues/5)

---

## 이슈 6 — M2 · Medium · Estimate 3 · `test` · **Blocked by: 이슈 4**

**제목**: [photoselect] test: 골든셋 평가 리그레션 — recall@K·대표 일치율·커버리지

**목적** — 흩어진 측정을 반복 실행 가능한 평가 하네스로 통합하고 목표 수치를 확정한다
(CI 아님 — 로컬/스파이크 실행용).

**태스크**
- [ ] recall@K·클러스터 대표 일치율·장면 커버리지 구현 + 리포트 포맷 통일
- [ ] 가중치(w_t·w_a·w_p) 튜닝 절차 문서화
- [ ] 목표 수치 설정 → plan.md §6-5 갱신

**완료 조건 (산출물)**
- 평가 스크립트 + 리포트, 목표 수치 명시된 plan.md, 튜닝 이력

**링크** — GitHub [#6](https://github.com/organic-agent/organic-agent-ai/issues/6)

---

## 이슈 7 — M3 · Medium · Estimate 5 · `feat` · **Blocked by: 이슈 3, 이슈 5**

**제목**: [photoselect] feat: 사진 진단(기능 B) 동기 API

**목적** — "이 사진 어때?"에 2~3초 내 근거 있는 응답. ML 재추론 없이 DB 조회 + LLM 1회.

**태스크**
- [ ] B-1 신호 조회 (photo_analysis·클러스터 형제·백분위·세션 유사도)
- [ ] B-2 LLM — `global.anthropic.claude-haiku-4-5…`(품질 확인 후 확정), vision ≤1장, 톤 규칙(판정 금지·작가 의도 존중)
- [ ] structured output `{verdict_signals[], comparison, suggestion?}` — UI 배지 계약
- [ ] wes `POST /diagnose` 계약 + provisioned concurrency 산정
- [ ] `global.` 프로필 해외 라우팅의 데이터 위치 정책 확인 (미리보기 전송)

**완료 조건 (산출물)**
- 동기 C Lambda 2~3초 응답 실측, wes API 계약 문서, verdict_signals 코드 사전

**링크** — GitHub [#7](https://github.com/organic-agent/organic-agent-ai/issues/7)

---

## 이슈 8 — M3 · Medium · Estimate 5 · `cicd` · **Blocked by: 이슈 3** (이후 증분)

**제목**: [photoselect] cicd: Terraform 배포 — ECR·Lambda 3종·IAM·CloudWatch

**목적** — `organic-agent-infra`로 배포. 각 기능 완성 시점마다 증분 배포한다.

**태스크**
- [ ] ECR(태그=git SHA) + Lambda 3종 (배치 A 8~10GB/15분, B 1~2GB/5분, C provisioned concurrency)
- [ ] IAM 최소 권한·RDS 보안그룹·Parameter Store
- [ ] CloudWatch 구조화 로그 + 15분 예산 알람
- [ ] wes 트리거 E2E

**완료 조건 (산출물)**
- 배포된 Lambda 3종, 알람 동작 확인, 배포 절차 문서

**링크** — GitHub [#8](https://github.com/organic-agent/organic-agent-ai/issues/8)

---

## 이슈 9 — Backlog · Low · Estimate 8 · `feat` · **Blocked by: 이슈 4, 이슈 5**

**제목**: [photoselect] feat: 실시간 재추천(④ 대화형) — 후속

**목적** — 부부가 고르는 동안 다음 후보를 재정렬. 초안 기능 검증 후 착수 판단.

**태스크**
- [ ] CLIP 유사도 휴리스틱 vs 점수 순 베이스라인 실측 → 착수 여부 결정
- [ ] (착수 시) 저지연 동기 API + wes 계약, 콜드스타트 = 점수 순

**완료 조건 (산출물)**
- 재정렬 품질 실측 리포트 → 착수/보류 결정 기록

**링크** — GitHub [#9](https://github.com/organic-agent/organic-agent-ai/issues/9)
