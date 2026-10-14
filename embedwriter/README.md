# embedwriter

임베더 결과 큐(SQS)를 읽어 `photo_analysis.embedding` · `photos.preview_key`·EXIF 를 **한 트랜잭션에 수백 장씩** 적재한다(wes #286).
torch 없음, 의존성은 `psycopg[binary]` 하나.

```
wes ──EVENT {galleryId, photoIds, storageKeys}──▶ embedder (VPC 밖, DB 없음, RESULT_QUEUE_URL)
        └─ 모델 배치(8장 ≈ 35KB) = 메시지 하나 ──▶ 결과 큐 ──▶ embedwriter (VPC 안, 동시 실행 ≤ 2) ──▶ RDS
                                                    └─ 5번 실패 → DLQ
```

**왜:** 부하 실측에서 동시 임베더 72개가 8장마다 각자 `INSERT photo_analysis` + `UPDATE photos` 를 하며 DB 시간의 57%
(INSERT 평균 86ms)를 먹어 사용자 요청을 굶겼다. 쓰기를 동시 ≤ 2 세션으로 모으고, 행마다 왕복 대신 `unnest` 한 문장으로 쓴다.

## 구조

```
embedwriter/
├── handler.py     Lambda 진입(SQS, ReportBatchItemFailures) — 묶음 한 트랜잭션, 실패 시 메시지별 재시도
├── __main__.py    로컬 CLI — python -m embedwriter --event event.json | --message msg.json
├── message.py     계약 v1 읽기·검증(짝: embedder/repository/results.py)
├── writer.py      SQL 한 문장(unnest → 가드 → photo_analysis UPSERT + photos UPDATE)
├── connection.py  psycopg 접속(웜 호출 사이 재사용)
└── settings.py    환경변수 → Settings
```

모듈이 작아 embedder 처럼 controller/service/repository 층으로 나누지 않았다.

## 메시지 계약 v1

```json
{"v": 1, "galleryId": 12, "embeddingModel": "facebook/dinov3-vitb16-pretrain-lvd1689m",
 "photos": [{"photoId": 101, "storageKey": "galleries/12/a.jpg", "previewKey": "previews/galleries/12/a.jpg",
             "vector": "<base64, little-endian float32 × EMBED_DIM>",
             "metadata": {"takenAt": "2026-10-01T03:04:05", "cameraMake": "…", "cameraModel": "…", "exposureTime": "1/200",
                          "fNumber": 2.8, "iso": 400, "width": 8192, "height": 5464, "byteSize": 123} | null}]}
```

- `takenAt` 은 ISO-8601. 컬럼이 `timestamp without time zone`(EXIF 벽시계)이라 오프셋이 붙어 와도 벽시계 값만 쓴다.
- 모양이 틀린 메시지(JSON 아님 · `v ≠ 1` · 타입·길이 틀림 · 벡터 차원 ≠ `EMBED_DIM` · NaN)는 SQL 전에 걸러 그 메시지만 실패.

## 쓰기 규칙 — embedder `store_embeddings` 와 같다

- `photos.id = photoId AND storage_key = storageKey AND deleted_at IS NULL` 이고 갤러리가 살아 있는 사진만 쓴다. 대상 배정 뒤
  원본이 교체된 늦은 결과·휴지통·지워진 갤러리는 0행. photo_analysis 도 같은 조건으로 거른다.
- `photo_analysis` UPSERT — `embedding` · `embedding_model` 만 갈아 끼우고 version + 1. 점수·그룹 컬럼은 그대로.
- `photos` — `preview_key` 덮어쓰기, EXIF 는 `COALESCE`(메타 없음 = 기존 값 유지), version + 1, updated_at.
- 같은 사진이 한 묶음에 두 번 오면(SQS 중복 전달) 뒤의 것만 쓰고, photo_id 순으로 정렬해 writer 둘의 잠금 순서를 맞춘다.
- UPSERT 라 같은 메시지를 다시 써도 결과가 같다.

## 실패

| 무엇 | 어떻게 |
|---|---|
| 메시지 모양이 틀림 | 그 메시지만 `batchItemFailures` → 5번 뒤 DLQ |
| 묶음 문장이 실패(한 메시지의 값 · 교착 등) | 메시지마다 각자 트랜잭션으로 다시 쓰고, 그래도 실패한 메시지만 `batchItemFailures` |
| DB 접속 불가 | 전 메시지가 `batchItemFailures` — 다시 온다 |

로그는 호출마다 한 줄: `embedwriter records=… photos=… written=… failed=… bulk=True|False seconds=…`.
`written < photos` 는 CAS 에 빗나간 늦은 결과(정상)다.

## DB 권한

embedder 와 같은 `embedder` 역할을 쓴다. 필요한 권한은 wes V15 `EMBEDDER_GRANT_CONTRACT` 에 이미 다 있다 —
`photo_analysis` SELECT·INSERT·UPDATE, `photos` SELECT(id, gallery_id, storage_key, deleted_at, EXIF, version)·
UPDATE(preview_key, EXIF, version, updated_at), `galleries` SELECT(id, deleted_at). `tests/test_writer_pg.py` 가 이 권한
그대로 만든 역할로 문장을 실제로 돌린다.

## 로컬 실행 · 테스트

```bash
cd embedwriter && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q          # 단위 테스트(DB 없음). 실제 DB 검증 2건은 건너뜀

# 실제 PostgreSQL + pgvector 로 SQL·권한 검증(선택)
docker run -d --rm --name ew-pg -e POSTGRES_PASSWORD=pw -p 127.0.0.1:55439:5432 pgvector/pgvector:pg16
EMBEDWRITER_PG_DSN=postgresql://postgres:pw@127.0.0.1:55439/postgres .venv/bin/python -m pytest tests -q
docker rm -f ew-pg

# 메시지 하나를 공유 DB 에 (터널: wes scripts/db-tunnel.sh)
DB_HOST=localhost DB_PORT=15432 DB_NAME=… DB_USER=embedder DB_PASSWORD=… DB_SSLMODE=require \
  .venv/bin/python -m embedwriter --message msg.json
```

## 빌드와 배포

```bash
./deploy.sh                       # wes-embedwriter
NAME_PREFIX=wes-dev ./deploy.sh   # wes-dev-embedwriter
```

ECR 저장소·함수·SQS 이벤트 소스 매핑(`function_response_types = ["ReportBatchItemFailures"]`, `scaling_config.maximum_concurrency = 2`)
·DLQ(maxReceiveCount 5)는 인프라(`../organic-agent-infra`)가 만든다. 주의 셋은 embedder 와 같다 — `--platform linux/amd64`,
`--provenance=false --sbom=false`, terraform apply 로는 코드가 안 나간다.

## 환경변수

| 변수 | 기본값 | 비고 |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | — | 필수(포트 기본 5432). embedder 역할 |
| `DB_SSLMODE` | `verify-full` | 터널로 로컬 실행할 때는 `require` |
| `DB_SSLROOTCERT` | `/opt/rds-ca/global-bundle.pem` | 이미지에 구워져 있다 |
| `EMBED_DIM` | `768` | `vector(n)` · embedder `EMBED_DIM` 과 같아야 한다 |
