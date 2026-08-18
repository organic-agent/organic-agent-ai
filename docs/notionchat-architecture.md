# notionchat 아키텍처 문서

> 작성일: 2026-08-18 · 대상 코드: `notionchat/` 패키지 전체
> 목적: 개선·배포 전 현재 구성 파악용. 계획 문서는 `docs/notion-chatbot-plan.md`, 규칙은 `.claude/rules/bedrock.md` 참고.

팀 Notion(인터뷰·기획·아키텍처·비용 자료)을 자연어로 조회하는 내부용 챗봇.
**RAG 없이 에이전트 검색 방식** — 사전 적재·임베딩 없이, LLM이 Notion API를 도구로 직접
검색·조회한다. 동시에 Bedrock 첫 실전 프로젝트로, tool use 루프와 Bedrock 클라이언트는
이후 P2(보정 요청 구조화)가 물려받을 공용 자산이다.

## 0. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| E2E 동작 | ✅ 실제 팀 Notion 30+ 페이지 대상, 검색→읽기→출처 포함 답변 검증 완료 |
| 사용 모델 | Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-...`, us-east-1) — 프롬프트 캐싱 + usage 로깅 |
| 모델 가용성 | 계정 차단은 최신 세대만(Opus 4.7+/Sonnet 5/GPT-5.6). Opus 4.6·Sonnet 4.6 이하 사용 가능 (§6) |
| 인터페이스 | 웹 UI(FastAPI + vanilla JS) + 터미널 CLI (로직 완전 공유) |
| 테스트 | pytest 8건 (블록→마크다운 변환기, 순수 함수만) |
| 배포 | ✅ https://chat.easyselect.kr (Vercel, 공유 계정 로그인) |
| 미구현 | 골든 질문 평가(계획서 §2-5), 답변 토큰 스트리밍, 대화 히스토리 길이 관리 |

## 1. 전체 아키텍처

```
사용자
 │
 ├── 웹 UI (web/ ← api/main.py가 서빙) ──────────┐
 └── 터미널 CLI (api/__main__.py) ────────────────────┤
                                 ▼
                    run_turn / ChatSession (api/core/provider.py)   ← 히스토리 관리, 오류 정규화
                                 │
                    tool use 루프 (api/core/agent.py)
                    boto3 bedrock-runtime.converse()             ← 상한 8회, 로그 콜백
                                 │
                    도구 4종 (api/notion/tools.py)
        search_notion / read_page / query_database / list_children
                                 │
                    NotionClient (api/notion/client.py)           ← httpx, 429 backoff
                                 │
                    api/notion/blocks.py                          ← 블록 트리 → 마크다운 변환
                                 │
                    Notion REST API (api.notion.com)
```

핵심 설계 결정:

- **적재 파이프라인 없음.** 질의 시점에 Notion API를 호출해 항상 최신 내용을 본다.
  Notion의 알려진 문제(중첩 블록, 표 문장화, 폐기 문서 오염, 증분 동기화)가 전부
  적재의 문제라서, 적재를 안 하면 문제 자체가 사라진다. 검색 품질이 부족하다고
  **평가로 확인되면** 그때 pgvector 시맨틱 검색을 도구 하나로 추가한다(Phase 2).
- **인터페이스와 대화 루프의 분리.** 인터페이스(웹/CLI)는 `run_turn(질문, 히스토리)` 하나만 안다.
  나중에 Slack 봇을 붙여도 `api/core/provider.py` 아래는 그대로 재사용된다.
- **단일 Converse 경로.** Converse는 Bedrock의 제조사 공통 인터페이스라 gpt-oss든
  Claude든 같은 루프로 호출된다. 모델 교체 = `NOTIONCHAT_MODEL_ID` 변경뿐이며,
  히스토리 포맷도 동일하게 유지된다. (Anthropic 전용 기능이 필요해지면 그때
  Mantle 클라이언트 도입 검토 — bedrock.md)

## 2. 한 질문의 처리 흐름

"인터뷰 결과 요약해줘"가 입력되면:

1. 웹 UI는 `run_turn(question, history, on_log)`을 직접, CLI는 `ChatSession.ask()` 경유로 호출
2. run_turn이 히스토리 체크포인트를 잡고(`len(history)`), 질문을 Converse 포맷으로 append
3. tool use 루프 시작 (api/core/agent.py):
   - `converse(modelId, system, messages, toolConfig, inferenceConfig)` 호출
   - 응답 `stopReason`이 `tool_use`면: 각 toolUse 블록에 대해
     `run_tool(notion, name, input)` 실행 → 결과를 `toolResult` 블록으로 append → 루프 반복
   - 도구 호출마다 `on_log("search_notion({...})")` 콜백 → UI 상태 박스/stderr에 표시
   - `stopReason`이 `end_turn`이면: content의 text 블록을 이어붙여 답변으로 반환
4. 오류 발생 시: `del messages[checkpoint:]`로 히스토리를 질문 이전으로 복구 후
   `ChatError`(한국어 메시지)로 변환해 던짐 → UI는 이것만 잡아 표시
5. UI가 답변 + 도구 호출 로그(expander)를 렌더링. 히스토리는 세션에 유지되어 후속 질문에 맥락 제공

안전장치:

- **도구 호출 상한**: 질의당 `MAX_TOOL_CALLS`(8)회. 초과 시 도구 실행 대신
  "지금까지 얻은 정보로 답하라"는 에러 결과를 반환해 폭주를 막는다
- **블록 재귀 깊이 제한**: `MAX_BLOCK_DEPTH`(3). 초과 부분은
  `… (깊이 제한 — list_children("id")로 확장 가능)` 주석으로 대체 — 모델이 필요하면 스스로 확장
- **Notion 도구 오류는 루프를 죽이지 않음**: `NotionAPIError`는 `is_error` 도구 결과로
  모델에게 전달되어 모델이 대응(재검색 등)

## 3. 모듈별 상세

패키지 구조 — 배포 단위 기준 2분할 (api = 백엔드 전부, web = 프론트 단독):

```
notionchat/
├── api/                   # 백엔드 전부 — 단독 배포 단위
│   ├── main.py            #   FastAPI 라우팅 + SSE + CORS (무상태 규격)
│   ├── __main__.py        #   터미널 CLI (python -m notionchat.api)
│   ├── config.py          #   설정 단일 지점 (.env 자동 로드)
│   ├── core/              #   대화 엔진
│   │   ├── provider.py    #     무상태 코어 run_turn + ChatSession
│   │   ├── agent.py       #     tool use 루프 (Bedrock Converse)
│   │   └── prompts.py     #     시스템 프롬프트
│   └── notion/            #   Notion 연동
│       ├── client.py      #     REST API 클라이언트 (429 backoff)
│       ├── blocks.py      #     블록 트리 → 마크다운 (순수 함수)
│       └── tools.py       #     도구 4종 정의 + 실행 + 출처 링크 생성
└── web/                   # 프론트엔드 — 단독 배포 단위 (정적 파일만)
    ├── index.html
    ├── style.css
    └── app.js             #   API_BASE로 API 주소 지정 가능 (기본 동일 출처)
```

### `api/config.py` — 설정 단일 지점
- `.env` 자동 로드: repo 루트의 `.env`를 읽어 **미설정 환경변수만** 채움
  (이미 export된 값이 우선 → 일시적 override 가능). 의존성 없이 자체 구현
- `MODEL_ID` 기본 `openai.gpt-oss-120b-1:0` — 메인 모델. Bedrock 모델 교체는 이 값만 변경
- `AWS_REGION` 기본 `us-east-1` — gpt-oss가 여기만 있음
- `MAX_TOKENS`(8192), `MAX_TOOL_CALLS`(8), `MAX_BLOCK_DEPTH`(3)
- `NOTION_WORKSPACE`: 출처 링크용 워크스페이스 슬러그 (현재 `yasic`)
- 규칙: **모델 ID는 이 파일에만 존재해야 한다** (bedrock.md)

### `api/core/provider.py` — 대화 진입점
- `run_turn(question, history, on_log) -> (answer, history)`: 무상태 코어.
  실패 시 히스토리를 질문 이전으로 롤백 + `ChatError`(한국어)로 변환
  - 잡는 예외: `ModelResponseError`(자체), botocore `ClientError`/`BotoCoreError`
- `ChatSession`: 히스토리를 들고 있는 로컬 CLI용 얇은 래퍼. `reset()`으로 초기화
- boto3/Notion 클라이언트는 프로세스 수명 동안 재사용 (서버리스 웜 스타트 대응)

### `api/core/agent.py` — tool use 루프 (P2 상속 자산)
- boto3 `converse()` 사용. 메시지 포맷: `{"role", "content": [{"text"}|{"toolUse"}|{"toolResult"}]}`
- `tool_config()`: 도구 정의(`TOOL_DEFINITIONS`)를 Converse `toolSpec`으로 변환
  — 도구 정의는 한 곳(api/notion/tools.py)에만 존재
- `stopReason` 가드: `max_tokens`는 `ModelResponseError`로 실패 처리
- gpt-oss의 `reasoningContent` 블록은 그대로 히스토리에 보존(에코백), 답변 추출 시 text만 취함
- Converse는 제조사 공통 인터페이스라 Claude 개방 시에도 이 루프 그대로 사용
  (Anthropic 전용 기능이 필요해지면 그때 Mantle 클라이언트 도입 검토 — bedrock.md)

### `api/notion/client.py` — Notion API 클라이언트
- httpx 동기 클라이언트, base `https://api.notion.com/v1`, `Notion-Version: 2022-06-28`, timeout 30s
- 429 처리: `Retry-After` 헤더 + 0.1s 여유로 최대 3회 재시도 (그 외 재시도 없음 — 적재가 없어 저위험)
- 메서드: `search`(POST /search), `page_meta`(GET /pages/{id}),
  `block_children`/`all_block_children`(GET /blocks/{id}/children, 페이지네이션 추종),
  `query_database`(POST /databases/{id}/query)
- 토큰: `config.notion_token()` — 없으면 안내 메시지와 함께 SystemExit

### `api/notion/blocks.py` — 블록 트리 → 마크다운 변환 (순수 함수, 테스트 대상)
- 지원 블록: paragraph, heading 1–3, bulleted/numbered/to_do 리스트(번호 리셋 처리),
  toggle(▸), code(언어 표기), quote, callout, divider, table(마크다운 표),
  child_page/child_database(ID 포함 표시), image/video/file/pdf/embed/bookmark(플레이스홀더+URL)
- rich text: `plain_text` 연결, href 있으면 `[text](url)`
- 재귀: `has_children`이면 `fetch_children` 콜백으로 하위 조회, `max_depth` 초과 시 확장 안내 주석
- `properties_to_text()`: DB 행 속성 문장화 — title/rich_text/select/multi_select/status/
  date/people/number/checkbox/url/email/phone_number/relation 지원, 빈 값 생략

### `api/notion/tools.py` — 도구 정의 + 실행 (§4 참고)

### `api/core/prompts.py` — 시스템 프롬프트
정책 (전문은 파일 참고):
- 내부 사실은 반드시 도구로 확인 후 답변 — 기억으로 지어내기 금지
- 검색 결과가 비면 동의어/짧은 키워드로 1~2회 재검색
- **모든 답변에 출처 링크** — 도구가 반환한 url을 변형 없이 그대로 사용
- 같은 주제 문서 여러 개면 `last_edited_time` 최신 우선, 충돌 시 답변에 명시
- 못 찾은 문서는 "없다" 단정 금지 — "Integration 미연결일 수 있음" 안내 (권한 경계)
- 캐시 보호: 프롬프트에 날짜·세션 ID 등 가변 값 금지

### `api/__main__.py` — 터미널 CLI
- `python -m notionchat.api`. readline으로 입력 히스토리/편집 지원. `exit`/Ctrl-D 종료
- ChatSession만 사용 — 웹 UI와 같은 run_turn 로직 공유

### `tests/test_blocks.py`
- 변환기 순수 함수 8건: rich text 링크, 기본 블록, 번호 리스트 리셋, 이미지 플레이스홀더,
  깊이 제한 주석, 재귀, 표 렌더링, 속성 문장화. **네트워크 없음**

## 4. 도구 4종 스펙

| 도구 | 입력 | 반환 | 용도 |
|---|---|---|---|
| `search_notion` | query | 페이지/DB 목록: 제목, id, url, last_edited | 항상 첫 진입점. 키워드 검색 |
| `read_page` | page_id | 제목+url+last_edited 헤더 + 본문 마크다운 | 찾은 페이지 내용 확인 |
| `query_database` | database_id, filter(JSON 문자열, 선택) | 행별 속성 문장 + id/url/last_edited | 칸반·회의록 등 DB 조회 |
| `list_children` | block_id | 하위 블록/페이지 목록 | 구조 탐색, 깊이 제한 확장 |

- 도구 설명(description)에 "언제 쓰는지"를 명시 — 모델의 도구 선택 정확도에 직결
- 모든 스키마 `additionalProperties: false` + `required`
- `search`는 10건, `query_database`는 20건 페이지 제한 (전체 페이지네이션은 read_page의 블록만)

### 출처 링크 생성 (`_page_url`) — 주의 필요한 부분
Notion API의 `url` 필드는 워크스페이스 슬러그가 빠진 `app.notion.com/p/<id>` 형식이라
**브라우저에서 접근이 안 된다** (실측). 그래서 API url을 버리고 페이지 ID로 직접 생성:
- `NOTION_WORKSPACE` 설정 시: `https://app.notion.com/p/<slug>/<id제거하이픈>`
- 미설정 시 폴백: `https://www.notion.so/<id>`

## 5. 설정 · 실행

### 환경변수 (.env — repo 루트, gitignore 대상, 시작 시 자동 로드)

| 변수 | 필수 | 현재값/기본값 | 설명 |
|---|---|---|---|
| `NOTION_TOKEN` | ✅ | (설정됨) | 내부 통합 시크릿. 읽기 전용 권한. 페이지에 통합 연결 필수 |
| `NOTION_WORKSPACE` | 권장 | `yasic` | 출처 링크용 워크스페이스 슬러그 |
| `NOTIONCHAT_AUTH_ID`/`PASSWORD` | 배포 시 필수 | — | 공유 로그인 계정. 미설정 = 무인증(로컬용) |
| `AWS_REGION` | — | `us-east-1` | Bedrock 리전 |
| `NOTIONCHAT_MODEL_ID` | — | `openai.gpt-oss-120b-1:0` | 모델 교체 시 이것만 변경 |

AWS 자격증명은 `~/.aws/credentials` default 프로필 사용 (별도 API 키 없음).
팀원 온보딩: `cp .env.example .env` 후 토큰 채우기.

### 실행

```bash
source .venv/bin/activate
uvicorn notionchat.api.main:app --reload   # 웹 UI (localhost:8000)
python -m notionchat.api                     # CLI
pytest                             # 테스트
```

## 6. 모델 상황과 복귀 경로 (중요)

현 AWS 계정(Organization 멤버, SCP 다수)의 실측 결과:

| 대상 | 상태 |
|---|---|
| Claude 전 모델 / GPT-5.6 (Luna 포함) | ❌ 계정 차단 — "not available for this account", 사용 사례 폼 단계 아님 |
| gpt-oss, Nova | ✅ 첫 호출 자동 활성화로 사용 가능 |
| 서울 리전 Mantle 엔드포인트 | ❌ 미배포 (NXDOMAIN) |
| 도쿄 리전 Mantle | ❌ 조직 SCP 명시 거부 |
| us-east-1 | ✅ SCP 통과 (현재 사용) |

**당분간 gpt-oss가 메인이다.** Claude 전환은 계정 차단이 풀린 뒤의 개선 항목으로 미뤄둔다
(① 조직 관리자에게 개방 요청 ② 별도 AWS 계정 등). 차단이 풀리면 `.env` 한 줄로 전환 — 코드 수정 없음:

```
NOTIONCHAT_MODEL_ID=anthropic.claude-opus-5
```

전환 시 반드시 골든 질문 평가로 gpt-oss와 비교 후 결정한다 (bedrock.md 규칙).
Converse 포맷은 모델과 무관하게 동일하므로 전환 시 코드·히스토리 변경이 없다.

## 7. 알려진 한계 · 개선 후보

배포 전 검토 권장 순서대로:

1. **골든 질문 평가 미구축** (계획서 §2-5) — 인터뷰·기획 문서에서 답이 명확한 질문
   10~15개로 정확도·출처 검증. 모델 비교(gpt-oss vs Claude vs Haiku)의 전제 조건
2. ~~인증 없음~~ → **해결됨**: 공유 계정 로그인 구현 (api/auth.py, §8).
   배포 시 `NOTIONCHAT_AUTH_ID`/`PASSWORD` 환경변수 설정 필수 — 미설정이면 무인증으로 열림
3. **대화 히스토리 무제한 증가** — 긴 세션에서 토큰 한도 초과 가능.
   오래된 턴 절삭 또는 도구 결과 축약(tool result가 가장 큼) 필요
4. **응답 스트리밍 없음** — 답변이 완성될 때까지 대기. 도구 로그가 진행감을 주지만
   최종 답변 토큰 스트리밍을 붙이면 체감 개선 (`converse_stream` API)
5. **동시성** — NotionClient가 동기 httpx. 사용자 수가 늘면 세션별 클라이언트 생성 비용과
   Notion rate limit(통합당 평균 3 req/s) 고려 필요
6. **검색 품질** — Notion 검색은 키워드 중심. 평가에서 "못 찾는" 실패가 유의미하면
   Phase 2(pgvector 시맨틱 검색 도구 추가)로 — 재작성이 아니라 도구 하나 추가
7. **query_database filter를 모델이 JSON 문자열로 작성** — 복잡한 필터에서 실수 가능.
   자주 쓰는 필터 패턴을 enum화하는 것도 방법
8. **이미지·draw.io 다이어그램은 안 보임** — 플레이스홀더 처리. 다이어그램 옆에 설명
   텍스트를 적는 문서 습관이 유일한 대응 (계획서 §3)
9. **Claude 미검증** — 계정 개방 후 모델 전환 시 스모크 + 골든 질문 평가 필수

## 8. FastAPI 서버 규격 (배포용, `api/main.py` + `web/`)

Vercel 서버리스 배포를 위해 **무상태(stateless)** 로 설계. 핵심: 서버가 대화 히스토리를
들지 않고, 클라이언트가 `history`를 왕복시킨다.

```
GET  /            채팅 웹 UI (web/index.html)
GET  /web/*       정적 자원 (style.css, app.js)
GET  /api/health  {"status", "model", "region", "auth": "off"|"required"|"ok"}
POST /api/login   {"id", "password"} → 성공 시 서명 세션 쿠키(nc_session, 30일) 발급
POST /api/logout  세션 쿠키 삭제
POST /api/chat    본문: {"question": str, "history": list}  → SSE 스트림. 인증 활성 시 쿠키 필수(무효면 401)
```

SSE 이벤트 순서: `tool_log`(도구 호출마다) → `answer`{answer, history} 또는 `error`{message}.
클라이언트 규약: **answer의 history를 저장했다가 다음 요청에 그대로 회신** (불투명 값 —
수정 금지, 초기화는 `[]`). Converse 포맷은 순수 dict라 JSON 직렬화가 그대로 보장된다.

내부 구조: `api/core/provider.py의 run_turn(question, history, on_log) -> (answer, history)`가 무상태 코어.
CLI의 `ChatSession`은 이 함수의 얇은 래퍼로, 웹 UI와 CLI가 같은 로직을 공유한다. SSE 스트리밍은 워커 스레드 + 큐로 도구 로그를 실시간 전달.

```bash
uvicorn notionchat.api.main:app --reload    # 로컬 실행 → http://localhost:8000
```

인증(api/auth.py): 공유 계정 1개 방식. `NOTIONCHAT_AUTH_ID`/`NOTIONCHAT_AUTH_PASSWORD`가
둘 다 설정된 경우에만 활성화(미설정 = 로컬 무인증 모드). 세션은 만료시각을 HMAC 서명한
HttpOnly 쿠키로, 서버 저장소가 없어 서버리스에서 동작. 서명 키는 계정 정보에서 유도되어
비밀번호 변경 = 전체 세션 무효화. 웹은 health의 auth 상태와 401 응답에 반응해 로그인
화면을 표시한다.

분리 배포 대응: api는 CORS 미들웨어 포함(`NOTIONCHAT_CORS_ORIGINS`로 출처 제한),
web은 순수 정적 파일이라 아무 정적 호스팅에나 올릴 수 있고 `window.NOTIONCHAT_API_BASE`로
API 주소를 지정한다. 로컬 개발 시엔 api가 web을 동봉 서빙(`/`, `/web/*`).

검증됨: health/index/SSE 이벤트 흐름, 실제 팀 Notion 대상 멀티턴(대명사 참조 해석) E2E.
미구현: 답변 토큰 스트리밍(현재는 도구 로그만 실시간), 인증, Vercel 설정 파일.

## 9. 배포 (Vercel 프로덕션 운영 중)

- URL: **https://chat.easyselect.kr** (보조: https://notionchat-five.vercel.app)
- 도메인: easyselect.kr의 Route 53 존(같은 AWS 계정)에 CNAME `chat` → cname.vercel-dns.com,
  소유 검증 TXT `_vercel` 레코드. SSL은 Vercel이 자동 발급·갱신 (Vercel 프로젝트 `notionchat`, 계정 sukangpunch)
- 구성: repo 루트의 `vercel.json`(legacy builds/routes — 전 경로를 `api/index.py` 함수로) +
  `api/index.py`(notionchat.api.main:app 재수출) + `.vercelignore`
- 환경변수(프로덕션): NOTION_TOKEN, NOTION_WORKSPACE, NOTIONCHAT_AWS_REGION,
  NOTIONCHAT_AWS_ACCESS_KEY_ID/SECRET(Vercel이 표준 AWS_* 이름을 예약하므로 전용 이름 사용),
  NOTIONCHAT_AUTH_ID/PASSWORD
- 재배포: `vercel deploy --prod --yes` (repo 루트에서)
- 개선 권장: 현재 AWS 키가 개인 IAM 유저(hyungjun) — Bedrock 호출 전용 최소권한 IAM 유저로 교체

### 비용 관측·절감
- **usage 로깅**: 턴마다 토큰 합계를 SSE answer 이벤트·웹 UI(도구 로그 패널)·CLI·서버 로그(`[usage]` 라인,
  Vercel 함수 로그)에 남긴다 — 비용 실측과 골든 질문 평가의 기반
- **프롬프트 캐싱**: Claude 모델일 때(cachePoint) 시스템 프롬프트와 대화 프리픽스를 캐시.
  히스토리에는 cachePoint를 저장하지 않고 요청 시점에만 붙인다(core/agent.py).
  Haiku 최소 캐시 단위(4096토큰) 미만 프리픽스는 조용히 무시됨 — 페이지를 읽은 뒤 후속 턴에서
  효과가 큼 (실측: 후속 턴 입력 48,666토큰 중 비캐시 1토큰)

### 다른 배포 옵션 (검토된 것)

| 옵션 | 비고 |
|---|---|
| Vercel (서버리스) | FastAPI가 무상태 규격이라 가능. vercel.json + 진입점 설정 필요. 함수 실행 시간 제한 유의 |
| AWS ECS/EC2 | `../organic-agent-infra` Terraform과 일관. IAM 역할로 자격증명 깔끔. 작업량 중간 |
| Slack 봇 | 배포라기보다 인터페이스 전환. 팀 사용성 최종형. run_turn 재사용 |

배포 시 공통: `NOTION_TOKEN`은 시크릿 매니저로, AWS는 IAM 역할로, 개인 토큰 대신 팀 공용 통합 권장.
