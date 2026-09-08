# NEXT — 파이프라인 v2, 다음 할 일 (2026-09-08 오후 기준)

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

### A. ⚠️ wes V16(PR-B) 배포 **전에** 끝내야 하는 것 — 잡 테이블 계약
V16 이 적용되면 score Lambda 갤러리 경로(`jobs.start/record/record_shard/shard_done`)와 categorize(`jobs.py` 의 status DONE·result 쓰기)가
permission denied 로 깨진다. wes PR-B 머지 시점을 확인하고 **같은 날** 배포한다(오늘 V15 와 같은 방식).
- [ ] **C1** `[categorize] fix`: 잡 4상태 계약 — `jobs.start/finish/fail` 의 `status`·`started_at`·`finished_at`·`result` 쓰기 삭제(V16 이 컬럼을 지우고 CHECK 를
  ANALYZING·CATEGORIZING·DONE·FAILED 로 바꾸며 photoselect 에 `UPDATE (error, updated_at)` 만 남긴다), 실패 시 `error` 만. `job.run` 의 "RUNNING 이어야 한다" 검사 삭제
  (wes 가 CATEGORIZING 상태로 넘긴다). 잡을 닫는 것은 wes. **지금 먼저 넣으면 안 된다** — 현재 wes 는 categorize 가 DONE 을 찍어야 잡을 닫는다.
  wes PR-B 머지 확인 → 같은 날 배포. 테스트 24 유지
- [ ] `[score] chore`: 갤러리 샤딩·조정자·자기 재호출·categorize 체인·`jobs.*` 삭제. Lambda 는 `photoIds` 폴백만. `worker.py`(잡 폴링 로컬 워커) 삭제.
  `handler.py` 가 옛 갤러리 페이로드를 받으면 명확한 에러. README 샤딩 절 삭제
- [ ] `[embedder] chore`: 갤러리 페이로드 경로·조정자·fan-out·advisory lock·`EMBED_SET_STATUS` 손잡이·`quality.py`·관리자 품질 잡·`technical_quality_*` 삭제.
  **추가 계약**(wes §7 E1): 결정적 실패(디코드 불가) → `photo_analysis.error` UPSERT, 일시 실패 → `UPDATE photos SET dispatched_at = NULL`(V15 가 embedder 에
  `dispatched_at` UPDATE 권한을 줬다). 완료 로그 `embedder gallery=G photos=N ok=K failed=F seconds=S`
- [ ] 인프라에 회신: embedder 갤러리 경로가 사라지면 `ReinvokeSelf`·score `InvokeFunction`(자기·categorize) 권한 제거 가능(결정 K). wes 가 categorize 를 직접 부른다
- [ ] `docs/embedder-photoselect-architecture.md` §2~§4 를 v2 기준으로 다시 쓰기(지금은 §0.1 만 v2)

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
- 갤러리 8 GPU fp16 점수 원복 여부(스냅샷 `docs/gpu-benchmark-2026-09-07/cpu-g8.json`).
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
  .venv/bin/python scripts/snapshot_scores.py dump 8 now.json && .venv/bin/python scripts/snapshot_scores.py compare ../docs/gpu-benchmark-2026-09-07/cpu-g8.json now.json
# wes 배포 상태 (V16 이 올라가는지 감시)
gh run list -R organic-agent/organic-agent-server -L 3 --workflow "[PROD] Build and Deploy"
```
