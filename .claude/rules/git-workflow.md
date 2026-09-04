# Git 작업 규칙 — 이슈·브랜치·커밋·PR

이 repo는 여러 모듈(파트)이 함께 사는 모노레포다. 규칙의 뼈대는
`organic-agent-server`의 방식을 따르고, 그 앞에 **모듈 prefix**를 얹는다.

## 모듈 prefix

최상위 디렉토리는 **실행 단위**다 — 디렉토리 하나가 따로 배포되는 것 하나다. 현재:

| 모듈 | 실행 단위 |
|---|---|
| `embedder` · `score` · `categorize` | Lambda 함수 하나씩 (#35) |
| `notionchat` | 서비스 (Vercel) |

실행 단위가 아닌 것(계획·아키텍처·설계 이력·학습 노트)은 전부 `docs/` 아래다. 최상위에 새 디렉토리를
만드는 것은 곧 새 배포 단위를 만드는 것이다.
새 모듈이 생기면 그 디렉토리 이름을 그대로 쓴다.

- 한 이슈/브랜치/PR은 **한 모듈만** 다룬다. 두 모듈을 건드리면 이슈를 나눈다.
- repo 전체에 걸친 작업(루트 CLAUDE.md, 공통 규칙, CI 등)은 prefix를 생략하고
  타입만 쓴다 (예: `docs: 작업 규칙 추가`).

## 타입 (organic-agent-server 라벨 세트와 동일)

`feat`(🌼 feat) · `fix`(🔨fix) · `docs`(📚 docs) · `chore`(⚡️chore) ·
`refactor`(🧹refactor) · `test`(🧪 test) · `build`(🧪 build) · `cicd`(🚦cicd)

## 이슈

- 제목: `[<모듈>] <타입>: <한글 설명>` — 예: `[score] feat: 러너 캐시로 웜 컨테이너 재사용`
- 라벨: 타입에 해당하는 이모지 라벨 1개
- 본문: `.github/ISSUE_TEMPLATE/issue-template.md` — Description / Task 체크리스트 /
  **산출물** / Related Module 네 섹션. 산출물 섹션에 "이 이슈가 끝나면 무엇이 존재해야
  하는가"(코드·리포트·문서 갱신)를 명시한다.
- DB 스키마가 필요한 작업이라도 마이그레이션 이슈는 여기 만들지 않는다 — wes
  (`organic-agent-server`) 쪽에서 협의·진행하고, 이 repo 이슈에는 태스크로만 적는다.

## 브랜치

- 형식: `<모듈>/<타입>/<이슈번호>-<kebab-slug>` — 예: `score/feat/40-runner-cache`
- repo 전체 작업: `<타입>/<이슈번호>-<slug>` (server 방식 그대로)
- 항상 `main`에서 분기한다.

## 커밋

- 작업 브랜치 안의 커밋 메시지는 자유. `main`에 남는 메시지는 PR squash 머지 제목이다.
- `main`에 남는 형식: `[<모듈>] <타입>: <한글 설명>(#이슈번호)` —
  예: `[score] feat: 러너 캐시로 웜 컨테이너 재사용(#40)`
- 문서만 바꾸는 소규모 변경은 이슈·PR 없이 `main` 직접 커밋을 허용한다
  (형식: `[<모듈>] docs: ...` 또는 `docs: ...`). 코드 변경은 반드시 PR을 거친다.

## PR

- 제목 = squash 머지 커밋 제목과 동일: `[<모듈>] <타입>: <설명>(#이슈번호)`
- 본문: `.github/PULL_REQUEST_TEMPLATE.md` — 연관 이슈(`close #N`) / 구현 사항 /
  리뷰 요구사항
- 머지 방식: **squash merge**, 머지 후 브랜치 삭제
- 하나의 PR은 하나의 이슈를 닫는 것을 기본으로 한다. 여러 이슈를 한 PR로 합칠 때는
  server 방식대로 제목에 나열한다 (예: `(#84, #85, #86)`).
