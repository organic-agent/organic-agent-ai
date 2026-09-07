# NEXT — 파이프라인 v2, 내일부터 할 일 (2026-09-08 기준)

> 배경·결정·실측은 `docs/`(로컬): `pipeline-v2-dev-plan-2026-09-07.md`(전체) · `ai-pipeline-v2-plan-2026-09-07.md`(AI 몫) ·
> `progress-2026-09-08.md`(어제 진행) · `gpu-benchmark-summary-2026-09-07.md`(벤치마크). 이 파일은 "다음 손댈 것" 목록만.

## 0. 어디까지 왔나

- AI repo 몫(E1 embedder photoIds · S1 GPU 워커 루프 · X1 이미지 CI · S2 프리페치 · #81 수정)은 **머지·검증 완료**. GPU 워커 실측 장당 0.033s, 유휴 30s 자기 정지.
- 운영 wes 는 아직 옛 계약(갤러리 페이로드·EMBEDDED·score Lambda 샤딩). 새 코드는 옛·새 둘 다 호환 — wes 전환 전까지 운영 영향 없음.
- 아직 **실제 인스턴스에서 워커를 켜 본 적은 없다**(SageMaker 로만 검증). 인프라 I1·I2 가 나오면 합류 3.

## 1. 내일 순서

### A. 합류 준비 — 인프라·wes 에 넘길 것 (오전, 30분)
- [ ] 인프라(I1 AMI 컴포넌트)에 넘길 **워커 systemd 유닛·부팅 스크립트** 초안: `docker pull <repo>:gpu` → `docker run --gpus all --env-file … <repo>:gpu`(CMD 가 `worker --gpu`), env 는 `DB_HOST/PORT/NAME/USER`, `DB_PASSWORD_SSM_PARAM=/wes/prod/photoselect.db.password`, `DB_SSLMODE=verify-full`, `DB_SSLROOTCERT=/opt/rds-ca/global-bundle.pem`, `S3_BUCKET`, `WORKER_IDLE_STOP_SECONDS=30`. 참고: `score/scripts/ec2_benchmark.py` 의 `RUN_SCRIPT`.
- [ ] 인스턴스 역할 권한 목록 전달: S3 GetObject(미리보기), ECR pull, `ssm:GetParameter`(+KMS decrypt) 그 파라미터, 자기 `ec2:StopInstances`(태그 조건). `lambda:GetFunctionConfiguration` 불필요.
- [ ] wes 에 전달: embedder 페이로드 `{"galleryId","jobId","photoIds":[…]}`(#74), score 폴백 `{"galleryId","photoIds":[…]}`(#76), `EMBED_SET_STATUS` 를 V15 와 같이 빈 값으로. `photo_analysis (photo_id) WHERE embedding IS NOT NULL AND clip_embedding IS NULL` 부분 인덱스 제안.

### B. 워커를 실제 인스턴스에서 (인프라 I2 가 나온 날)
- [ ] 정지 인스턴스 → `StartInstances` → 부팅 → 워커가 집기 → 유휴 30s 뒤 스스로 정지되는지. 콜드(Start → 첫 배치) 목표 ≤ 40s(부팅 16 + pull 확인 2 + 로드 9 + 워밍).
- [ ] 두 대 동시에 켜서 같은 갤러리를 나눠 먹는지(중복 0, 합 = 대상 수).
- [ ] 자기 정지가 실패하는 상황(권한 제거)에서 wes 30분 감시가 잡는지 — wes W6 가 있을 때.

### C. 남은 AI 작업 (독립, 시간 나는 대로)
- [ ] **S4(선택)** `[score] feat: ARNIQA torch.compile/channels_last` — 점수 152s 중 ARNIQA 96s 대상, −40~70s 기대. 조건: fp16 스냅샷 비교 ρ ≥ 0.99, 컴파일 시간이 콜드에 더해지므로 유휴 30s 정책과 같이 판단. 안 되면 버림.
- [ ] `docs/embedder-photoselect-architecture.md` 에 v2 실행 모양(스트리밍·GPU 워커·폴백) 반영.
- [ ] `scratchpad/compare_local.py` 를 `score/scripts/compare_local.py` 로 승격(CPU 경로 비트 동일 검사 재사용).

### D. wes V15~V17 배포 뒤 (정리 이슈, 그날)
- [ ] `[embedder] chore`: 갤러리 페이로드 경로·조정자·fan-out·advisory lock·`EMBEDDED` 쓰기 삭제, `technical_quality_*`·`quality.py`·관리자 품질 잡 삭제(wes 가 그 잡을 지운 뒤).
- [ ] `[score] chore`: 갤러리 샤딩·조정자·자기 재호출·categorize 체인·`jobs.record/record_shard/shard_done` 삭제. Lambda 는 `photoIds` 폴백만. `worker.py`(잡 폴링 로컬 워커) 삭제.
- [ ] `[categorize] chore`: 잡 4상태 계약(ANALYZING→CATEGORIZING 은 wes 가, Lambda 는 `error` 만 쓰기). `jobs.py` 축소.
- [ ] 문서: `score/README.md`·`embedder/README.md` 의 샤딩·EMBEDDED 절 삭제, `docs/embedder-photoselect-architecture.md` 갱신.

## 2. 결정 대기 (사용자)
- 갤러리 8 GPU fp16 점수 원복 여부(스냅샷 `docs/gpu-benchmark-2026-09-07/cpu-g8.json`).
- 벤치마크 IAM 역할 2개 삭제 시점(인프라가 지운다고 함).
- 다중 사용자 운영 시 유휴 정지 600s 로 올릴 시점.

## 3. 자주 쓰는 명령

```bash
# 테스트
cd embedder && .venv/bin/python -m pytest tests -q        # 80
cd score && .venv/bin/python -m pytest -q                  # 40
# GPU 워커 검증 (SageMaker, 인프라 없이) — 먼저 galleries 의 일부 clip_embedding 을 NULL 로
cd score && .venv/bin/python scripts/sagemaker_benchmark.py run --gallery-id 8 --instance ml.g6.xlarge \
  --label worker --container-args "worker --gpu" --env WORKER_POLL_SECONDS=2
# DB 터널 (wes)
../organic-agent-server/wes/scripts/db-tunnel.sh 15432
# 점수 스냅샷·비교
DB_HOST=localhost DB_PORT=15432 DB_NAME=wes_db DB_USER=photoselect DB_PASSWORD=… DB_SSLMODE=require \
  .venv/bin/python scripts/snapshot_scores.py dump 8 now.json && .venv/bin/python scripts/snapshot_scores.py compare ../docs/gpu-benchmark-2026-09-07/cpu-g8.json now.json
```
