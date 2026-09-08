# NEXT — 파이프라인 v2, 다음 할 일 (2026-09-08 15:20 KST 기준)

> 배경·결정·실측은 `docs/`(로컬): `pipeline-v2-dev-plan-2026-09-07.md`(전체) · `ai-pipeline-v2-plan-2026-09-07.md`(AI 몫) ·
> `progress-2026-09-08.md`(진행) · `gpu-benchmark-summary-2026-09-07.md`(벤치마크) · `embedder-photoselect-architecture.md` §0.1(v2 실행 모양).
> 이 파일은 "다음 손댈 것" 목록만.

## 0. 어디까지 왔나

- AI repo 몫(E1 · S1 · X1 · S2 · #81) 머지·검증 완료. GPU 워커 실측 장당 0.033s, 유휴 30s 자기 정지(SageMaker 로만 검증).
- **wes V15(#164) 가 2026-09-08 13:26 KST 운영 배포됨** — `photos.status` 두 값, `photo_analysis.error`, embedder 역할의 status UPDATE 권한 제거.
  같은 날 AI 쪽 대응 배포: **#84**(embedder status 안 씀) · **#86**(score 집기 `error IS NULL`, PREVIEW_MISSING/SCORE_FAILED, key=value 로그).
- **#94 categorize 운영 복구** — V15 가 EMBEDDED 를 없앴는데 `load_db` 가 그 값으로 대상을 골라 모든 갤러리 0장 → 잡 FAILED 였다. 13:26~15:0x 사이 분석은 전부 실패했을 것.
  배포 뒤 갤러리 하나로 확인 필요. (오늘 V15 대응에서 categorize 를 빠뜨렸다 — v2 계획이 "categorize 변경 없음"으로 분류한 탓)
- DB 비밀번호는 `--env-file` 하나로 결정(#91, `DB_PASSWORD_SSM_PARAM` 삭제). 인프라에 넘길 전달본 **#88** `score/deploy/gpu-worker/`(유닛·env·run 스크립트·권한 표·검증 절차). CPU 비트 동일 검사 **#90** `score/scripts/compare_local.py`.
- wes 는 지금 **PR-B(잡 층, V16)** 작업 중(브랜치 `feat/166-pipeline-v2-job-layer`). V16 은 photoselect 의 `ai_analysis_jobs` 권한을 `UPDATE (error, updated_at)` 만 남긴다.
- 아직 **실제 인스턴스에서 워커를 켜 본 적은 없다**. 인프라 PR-3b/3c 가 나오면 §1-B.

## 1. 순서

### A. 세 repo 싱크 — AI repo 몫 완료 (2026-09-08)
계약 수정 #84 · #86 · #94 · #95, 그리고 **죽은 코드 정리 #98(score) · #100(embedder) · #102(categorize 문서)** 까지 머지·배포 완료.
지운 것: score `jobs.py`·`chain.py`·`worker.py`·샤딩·조정자·잠금(약 250줄), embedder `quality.py`·관리자 품질 잡·샤딩·조정자·잠금·`EMBED_SET_STATUS`·`force`(약 200줄).
남긴 것: 갤러리 CLI 경로(wes `local-ai.sh` 가 쓴다) · `--local` 데이터셋 모드(`compare_local.py`) · 관리자 DERIVATIVE·EMBEDDING · 벤치마크 진입점.
테스트 embedder 47 · score 31 · categorize 27. CPU 경로 비트 동일 재확인.

- [ ] **인프라에 회신할 것(결정 K)**: embedder 롤의 `ReinvokeSelf`, score 롤의 `lambda:InvokeFunction`(자기 + categorize) **둘 다 제거 가능**.
  갤러리 fan-out·자기 재호출·체인이 코드에서 사라졌다. Lambda 인터페이스 VPC 엔드포인트는 categorize 의 Bedrock 때문에 남는다
- [ ] `docs/architecture/embedder-score-categorize.md` §2~§4 를 v2 기준으로 다시 쓰기(지금은 §0.1 만 v2)
- [ ] **wes 에 요청**: `scripts/gpu/score-worker.sh` 에서 `--once` 만 빼면 된다(#104 로 우리 쪽은 고쳤다).
  `--no-idle-stop` 이 이제 "EC2 정지만 건너뛰고 유휴가 되면 종료"라 큐를 다 비우고 끝난다 — 모델 1회 로드, 미리보기 낭비 없음

### A2. 다른 repo 상태 (2026-09-08 14:35 확인)

**wes 는 v2 네 단계를 하루에 다 올렸다** — V15 사진 층(#164, 13:26) · V16 잡 층(#166, 14:10) · PR-C GPU 제어(#168) · V17 정리(#170, 배포 중).
V17 은 `categorization_jobs`·`categorization_job_photos` DROP 인데 **AI repo 는 그 테이블을 쓰지 않는다 — 영향 없음**(grep 확인).

**지금 운영에서 도는 경로**(`/wes/prod/app.analysis.gpu.enabled` = **false**): wes 스위퍼 → embedder Lambda `{galleryId, photoIds}` 50장 →
**score Lambda 폴백** `{galleryId, photoIds}`(GPU 가 없으면 `GpuController.isFallbackDue` 가 유예 없이 바로 보낸다) → categorize Lambda `{galleryId, jobId}`.
셋 다 오늘 맞춘 계약이다(#84 · #86 · #94 · #95). GPU 워커는 인프라가 인스턴스를 만들 때까지 안 쓴다.

- [ ] **아직 아무 잡도 안 돌았다** — 세 Lambda 의 마지막 로그가 2026-09-07 02:00 이다. 오늘 고친 것들은 전부 **실측 미검증**.
  갤러리 하나로 업로드 → 분석을 돌려 embedder → score 폴백 → categorize → 폴더까지 확인하는 것이 다음 순서(로그 형식 `key=value` 도 같이 확인)
- ⚠️ **인프라 apply 가 main 에서 실패 중**(#41·#43 두 번). 원인은 `modules/score-gpu` 의 서비스 연결 역할 `description` 이 한글이라
  IAM 이 거부(`[\u0020-\u007E\u00A1-\u00FF]` 만 허용). SLR 둘 다 미생성 → **AMI 파이프라인 리소스가 안 만들어졌다**. 고칠 PR 도 아직 없다.
  다만 SSM 키(`app.analysis.embedder-function-name` · `gpu.enabled`)는 같은 apply 에서 **생성됐다**(독립 리소스라 통과) — 그래서 파이프라인은 흐를 수 있다.
- [ ] 인프라 PR-3c(GPU 인스턴스 2대·워커 롤·SG·유휴 알람·앱 롤 EC2 제어) 미착수. apply 가 빨간 상태라 그것부터 고쳐야 한다

### B. 워커를 실제 인스턴스에서 (인프라 I2 가 나온 날) — 절차는 `score/deploy/gpu-worker/README.md` §검증
- [ ] 콜드(Start → 첫 배치) ≤ 40s · 배치 로그 `score worker batch=32 …` · 유휴 30s 뒤 자기 정지 · env 파일 삭제 확인
- [ ] 두 대 동시에 같은 갤러리(중복 0, 합 = 대상 수)
- [ ] 자기 정지 실패(권한 제거) → wes W6 30분 / CloudWatch 알람이 잡는지. 점수 중·유휴 CPU 사용률 실측 → 알람 임계값(5%) 확정
- [ ] 이미지 갱신: main push → `gpu` 태그 이동 → 다음 기동 pull 로그 다이제스트

### C. 남은 AI 작업 (독립, 시간 나는 대로)
- [ ] **S4(선택)** `[score] feat: ARNIQA torch.compile/channels_last` — 점수 152s 중 ARNIQA 96s 대상, −40~70s 기대. **구현 전 계획 먼저**(SageMaker 실측 $0.5 안팎):
  ① 로컬 `compare_local.py check --tol 0`(CPU 불변) ② SageMaker 갤러리 8 fp16 스냅샷 ρ ≥ 0.99 ③ 컴파일 시간(콜드 +?s)과 유휴 30s 정책 비교. 안 되면 버림
- [ ] wes 에 물어볼 것: `photo_analysis.error` 값을 UI 에 그대로 보이는지(PREVIEW_MISSING · SCORE_FAILED 코드 유지 여부)

## 2. 결정 대기 (사용자)
- 갤러리 8 GPU fp16 점수 원복 여부(스냅샷 `docs/experiments/gpu-benchmark-2026-09-07-raw/cpu-g8.json`).
- 벤치마크 IAM 역할 2개 삭제 시점(인프라 PR-0a 가 지운다고 함).
- 다중 사용자 운영 시 유휴 정지 600s 로 올릴 시점(인스턴스 `WORKER_IDLE_STOP_SECONDS` 한 줄, AMI 재빌드 불필요).

## 3. 자주 쓰는 명령

```bash
# 테스트
cd embedder && .venv/bin/python -m pytest tests -q        # 81
cd score && .venv/bin/python -m pytest -q                  # 45
cd categorize && ../score/.venv/bin/python -m pytest -q    # 23
# CPU 경로 비트 동일 검사 (두 커밋, 로컬 데이터셋)
cd score && .venv/bin/python scripts/compare_local.py check "dataset1/데이터셋1" --rev main --limit 16
# GPU 워커 검증 (SageMaker, 인프라 없이) — 먼저 galleries 의 일부 clip_embedding 을 NULL 로
cd score && .venv/bin/python scripts/sagemaker_benchmark.py run --gallery-id 8 --instance ml.g6.xlarge \
  --label worker --container-args "worker --gpu" --env WORKER_POLL_SECONDS=2
# DB 터널 (wes)
../organic-agent-server/wes/scripts/db-tunnel.sh 15432
# 점수 스냅샷·비교
DB_HOST=localhost DB_PORT=15432 DB_NAME=wes_db DB_USER=photoselect DB_PASSWORD=… DB_SSLMODE=require \
  .venv/bin/python scripts/snapshot_scores.py dump 8 now.json && .venv/bin/python scripts/snapshot_scores.py compare ../docs/experiments/gpu-benchmark-2026-09-07-raw/cpu-g8.json now.json
# wes 배포 상태 (V16 이 올라가는지 감시)
gh run list -R organic-agent/organic-agent-server -L 3 --workflow "[PROD] Build and Deploy"
```
