# embedder 배포 환경 테스트 계획 — GET 프리페치 · JPEG 축소 디코드 (#39)

작성 2026-09-05. 로컬에서는 회선(약 5MB/s)이 병목이라 #39 의 효과를 검증할 수 없었다(`local-e2e-2026-09-05.md` §5).
운영 Lambda 는 같은 리전 S3 라 회선이 병목이 아니고 vCPU 가 느려 CPU 몫이 커지므로, 효과는 거기서 재야 한다.
이 문서는 **무엇을, 어떻게, 어떤 기준으로** 재는지와 이상 시 되돌리는 절차를 적는다. 실측 결과는 §6 에 채운다.

## 1. 대상과 기준선

| 항목 | 값 |
|---|---|
| 함수 | `wes-embedder` (ap-northeast-2), 메모리 3008MB(≈2 vCPU), 타임아웃 900s, `EMBED_BATCH_SIZE=8` |
| 배포 경로 | PR 머지 → `deploy-lambda.yml` 이 `embedder/**` 변경을 보고 ECR 푸시 + `update-function-code` (#37). 수동은 `embedder/deploy.sh` |
| 로그 | CloudWatch `/aws/lambda/wes-embedder`. 진행 라인 `진행 N/M (장당 X.XXs)`, 결과 `완료: {...elapsedSeconds, stopped, remaining, failed...}`, `REPORT ... Max Memory Used` |
| 새 설정 | `EMBED_DOWNLOAD_WORKERS`(기본 4, 코드 기본값 — Terraform 변경 없이 동작) |

**배포 전 기준선(현재 코드, CloudWatch 2026-08-22·08-30 실행):**

| 실행 | 장당 | 비고 |
|---|---|---|
| 40장 · 8장 · 24장 | 0.46~0.53s | 콜드 스타트 Init 5.9~7.0s 별도 |
| 16장 · 80장 | 0.70~0.71s | |
| 176장 (08-30) | 0.48s | |
| Max Memory Used | 1,136~1,268MB / 3,008MB | 모델 포함 |

즉 운영 Lambda 의 현재 장당 시간은 **0.5~0.7s** 이고, 여기서 CPU(디코드·축소·인코드)와 GET 대기의 비율은 아직 모른다.
로컬 M 시리즈 맥에서 CPU 경로가 46~104ms 였으므로 Lambda vCPU 에서는 그 3~5배(0.15~0.5s)로 추정한다 — 이 추정이 맞으면
장당 시간의 절반 이상이 CPU 이고 축소 디코드가 그중 30~40% 를 깎는다.

## 2. 가설과 합격 기준

| 가설 | 측정 | 합격 |
|---|---|---|
| H1 장당 시간이 준다 | 같은 갤러리·같은 사진 수의 `장당` 평균 (웜 컨테이너 배치만, 첫 배치 제외) | 기준선 대비 **25% 이상 감소** (0.5s → 0.38s 이하). 감소 없으면 §5 의 진단으로 |
| H2 메모리가 안전하다 | `REPORT Max Memory Used` | **2,400MB 이하** (3,008 의 80%). 프리페치 창은 원본 최대 24장 ≈ 300MB 이므로 1.6GB 안팎 예상 |
| H3 결과가 같다 | `완료:` 의 `processed`·`failed`·`metadataFailed`, DB `photos.status`, `photo_analysis.embedding` 차원·개수 | 기준선과 동일. `failed` 0 |
| H4 미리보기 화질 | 배포 후 만들어진 `previews/…jpg` 를 배포 전 파일과 비교 | 크기(가로·세로) 동일, 파일 크기 ±10%, 눈으로 봐서 차이 없음 (로컬 실측 평균 0.4/255) |
| H5 데드라인·재호출 계약 유지 | 15분 안에 못 끝나는 큰 갤러리에서 `stopped: true` → `reinvoked: true` → 다음 호출이 나머지를 이어 감 | 재호출 체인 끝에 `remaining: 0`, 중복 처리 없음(`processed` 합 = 대상 수) |
| H6 GET 실패 격리 | (선택) 존재하지 않는 `storage_key` 를 가진 사진 1장을 테스트 갤러리에 섞음 | 그 사진만 `failed`, 나머지는 정상. 스레드 풀 예외가 잡 전체를 죽이지 않음 |

## 3. 절차

1. **기준선 고정 (배포 전).** 위 표 외에, 테스트에 쓸 갤러리 G 로 배포 전 코드를 한 번 더 돌려 같은 조건의 숫자를 남긴다.
   ```bash
   aws lambda invoke --region ap-northeast-2 --function-name wes-embedder --invocation-type Event \
     --payload '{"galleryId": G, "force": true}' --cli-binary-format raw-in-base64-out /dev/null
   aws logs tail /aws/lambda/wes-embedder --region ap-northeast-2 --since 30m --format short | grep -E "장당|완료:|REPORT"
   ```
   G 는 **100~300장, 원본 5~13MB, JPEG 위주** 인 스테이징/테스트 갤러리. `force: true` 는 이미 EMBEDDED 인 사진을 다시 처리하므로
   운영 고객 갤러리에는 쓰지 않는다 (미리보기가 같은 내용으로 덮이지만 벡터가 다시 적히는 동안 wes 의 `isAnalyzed` 창이 열린다).
2. **머지 → 자동 배포.** Actions 의 `deploy-lambda` 가 초록인지, `aws lambda get-function-configuration --query CodeSha256` 가 푸시된
   다이제스트와 같은지 확인한다 (`deploy.sh` 와 같은 검증).
3. **같은 갤러리 G 로 3회 실행.** 1회차는 콜드 스타트(Init 포함)라 참고만, 2·3회차 `장당` 평균을 기준선과 비교한다.
   실행 사이에 `force: true` 를 유지한다.
4. **H3·H4 확인.** 실행 뒤 DB 와 S3:
   ```sql
   select status, count(*) from photos where gallery_id = G and deleted_at is null group by 1;
   select count(*), min(vector_dims(embedding)) from photo_analysis a join photos p on p.id = a.photo_id where p.gallery_id = G;
   ```
   미리보기 2~3장을 `aws s3 cp` 로 받아 배포 전 사본과 크기·용량을 비교한다.
5. **H5 (큰 갤러리).** 사진 800장 이상 갤러리로 1회. `stopped`/`reinvoked` 라인과 재호출된 RequestId 들의 `processed` 합을 본다.
   이 갤러리는 로컬 E2E 의 갤러리 1(822장)을 스테이징 S3 에 올린 것이 적당하다.
6. **결과 기록.** §6 표를 채우고 `embedder-photoselect-architecture.md` §2.3 의 "실측 아직 없음" 을 갱신한다.

## 4. 조정 손잡이

- `EMBED_DOWNLOAD_WORKERS` — H1 이 미달이고 로그에서 배치 시간이 GET 대기(첫 사진의 `download.result()`)에 몰려 있으면 8 로 올린다.
  Lambda 함수 환경 변수는 Terraform(`modules/analysis/main.tf` embedder env)에 넣어야 apply 에서 안 지워진다.
- `embedder_memory_mb` — 3008 → 4096 이상으로 올리면 vCPU 도 비례해 늘어 CPU 몫이 준다. 프리페치와 별개로 가장 확실한 손잡이지만
  비용이 비례한다. H1 미달 시 두 번째 후보.
- `EMBED_BATCH_SIZE` — 8 유지. 프리페치 창(두 배치)이 이 값에 비례해 메모리를 쓴다.

## 5. H1 미달 시 진단

`장당` 이 안 줄면 원인은 셋 중 하나다. 순서대로 본다.

1. **GET 이 이미 빨랐다** — Lambda 에서 6.6MB 원본이 0.1s 안에 오면 프리페치의 이득은 0.1s 뿐이다. 기준선 0.5s 의 나머지가 CPU 라는 뜻이므로 축소 디코드가 효과를 냈어야 한다 → 2 로.
2. **축소 디코드가 안 걸렸다** — 원본이 HEIC(아이폰)이거나, 긴 변이 1536×2 = 3072px 미만이면 draft 가 1 을 고른다. 테스트 갤러리의 원본 포맷·크기 분포를 `photos.byte_size`·`width` 로 확인한다. HEIC 비중이 크면 이 변경은 그 갤러리에 효과가 없는 게 정상이다.
3. **CPU 가 PUT·JPEG 인코드에 있다** — `to_jpeg` 의 `optimize=True, progressive=True` 는 인코드 시간을 2~3배 늘린다. 로컬 실측에서 CPU 경로 46~81ms 중 인코드 몫을 따로 재지 않았다. 다음 이슈 후보.

## 6. 실측 결과 (배포 후 채움)

| 날짜 | 갤러리(장수) | 코드 | 장당 (2·3회차 평균) | Max Memory | failed | 비고 |
|---|---|---|---|---|---|---|
| | | 배포 전 | | | | |
| | | #39 | | | | |

## 7. 되돌리기

- 이전 이미지로: `aws lambda update-function-code --function-name wes-embedder --image-uri <ECR>@<이전 digest>` — 이전 digest 는
  `aws ecr describe-images --repository-name wes-embedder --query 'sort_by(imageDetails,&imagePushedAt)[-2].imageDigest'`.
- 코드 되돌림 없이 동작만 이전과 같게: `EMBED_DOWNLOAD_WORKERS=1` 로 프리페치를 사실상 끈다 (축소 디코드는 남는다).
- 잡이 중간에 죽어도 배치 단위 commit 이라 재호출이 이어 간다 — 데이터 복구 절차는 없다.
