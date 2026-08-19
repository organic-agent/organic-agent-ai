<!-- 워크스페이스 구조 지도 — 시스템 프롬프트에 그대로 삽입된다 (docs/retrieval-improvement-plan.md 0-a).
     규칙: 자주 묻는 영역 위주, "무엇을 물으면 여기를 봐야 하는지"가 드러나게.
     실제 Notion 루트 페이지의 섹션(Pages/References/...) 구조를 따른다.
     Notion 구조가 바뀌면 scripts/list_notion_ids.py로 다시 뽑아 갱신할 것. 이 주석은 삽입 시 제거된다. -->

루트: 오가닉 에이전트 홈 (page_id: 35bac24a664c8077b3fde58e3ca3f0bc) — 길을 잃으면 여기서 list_children으로 탐색.

Pages (핵심 작업 문서):
- 회의록 (database_id: 35cac24a664c8040b0c7cf7776a456b0): 회의 기록 DB. 결정 사항·액션 아이템 질문은 여기.
- ERD (page_id: 392ac24a664c80ac9f0fe67632b0aea0): DB 설계. 사용자/갤러리·초대/사진·선택/협업 셀렉 도메인별 테이블 정의.
- Architecture (page_id: 392ac24a664c802e975af97c7238bbb8): 인포메이션·애플리케이션·시스템·인프라 아키텍처. 하위에 Infra (page_id: 3a1ac24a664c80028d65d15e399f64ee).
- PoC (page_id: 399ac24a664c8022aad4e549ff1efc6f): 기술 검증. 업로드 방식 비교, 이미지 카테고리화 POC.
- 기획 (page_id: 3a4ac24a664c803ebc57d7e954f31933): 기획 개요·사용자 시나리오·painpoint 리스트 하위 포함.
  - 오가닉 기획 칸반 (database_id: 3bbac24a664c80aa8125dee2f3365d4a): 기능별 기획 작업 보드. 기능 진행 상태는 여기.
- 스튜디오 인터뷰 (page_id: 3afac24a664c80f5a579cba5c3e909e7): 인터뷰 질문 프레임·템플릿 (실제 기록 아님).
- 인터뷰 결과 정리 (page_id: 3b5ac24a664c80e68ad1e5667bf12ed5): **실제 진행한 인터뷰의 녹취·정리**.
  "실제 인터뷰 내용/결과가 어디 있냐"는 질문은 여기. 고객 페인포인트 근거도 여기.

References (운영·행정):
- 팀 운영 (page_id: 392ac24a664c80ef9e28e43e617f7643): Team R&R·깃허브 컨벤션·팀 규칙.
- 비용 처리 (page_id: 391ac24a664c80e0bb10fb4166db9dc1): 팀이 실제 쓴 결제 회차·활동비 정리. 내역 DB는 비용 처리 (database_id: 391ac24a664c80fb84c2efe8c6d2719d).
- 공식 활동비 지원 규정(지원 항목·금액·절차·기한)은 Notion이 아니라 read_expense_guide 도구로 조회한다.
- 계정(AWS·어드민·서비스·구글 등) (page_id: 391ac24a664c809b9d8df26b6f169856): 공용 계정 안내.
- 기획 심의 결과 (page_id: 391ac24a664c8075a629d30de6faa626): 심의 피드백(기획 완성도·아이디어·기술 가능성·기대효과).
- 증빙서류 (page_id: 3a4ac24a664c80249c65e5a4c3858f85): 구매·서비스 이용 증빙.
- 창고 (page_id: 37cac24a664c80bebca8c7fa4288877b): 멘토님 피드백, 기획 심의 예상 질문, 웨딩 불편함 조사 등 보관. 당장 다른 곳에 보관하기 애매한 자료들.

Schedule (일정):
- ASM 부산 17 일정 (database_id: 35bac24a664c805cbd74db676e16dfbc): 팀 일정 DB. 우리 팀의 멘토링·특강 약속 날짜는 여기.
- 소마 공식 일정(발대식·수료식·중간점검·TOPCIT·최종점검·교육·창업 프로그램)은 read_soma_schedule 도구로 조회한다
  (원본이 이미지라 Notion에서 직접 못 읽음).
- 활동비·학습비 신청 기간(차수 일정)은 read_expense_guide에 있다.