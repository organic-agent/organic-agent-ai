"""FastAPI 챗봇 서버 — 무상태 규격 (Vercel 서버리스 배포 대응).

실행(로컬): notionchat/ 디렉토리에서 uvicorn api.main:app --reload
프론트엔드는 web/ 에 있고, 이 서버가 정적 파일로 서빙한다.

API 규격
--------
GET  /            채팅 웹 UI (web/index.html)
GET  /web/*       정적 자원 (style.css, app.js)
GET  /api/health  상태 확인: {"status", "model", "region"}
POST /api/chat    질문 1턴 수행. SSE(Server-Sent Events) 스트림 응답.

  요청(JSON):
    {
      "question": "질문 텍스트",            # 필수
      "history": [ ... ]                    # 이전 턴에서 받은 history를 그대로 회신 (첫 턴은 [])
    }

  응답(SSE, text/event-stream) — 이벤트 순서:
    event: tool_log   data: {"message": "search_notion({...})"}    # 도구 호출마다 0회 이상
    event: answer     data: {"answer": "...", "history": [...]}    # 성공 종료 (1회)
    event: error      data: {"message": "..."}                     # 실패 종료 (1회)

  클라이언트 규약: answer 이벤트의 history를 저장했다가 다음 요청에 그대로 보낸다.
  history는 서버가 해석하는 불투명(opaque) 값이며 클라이언트가 내용을 수정하면 안 된다.
  대화 초기화 = history를 []로.
"""

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import auth, config
from api.core.provider import ChatError, run_turn

app = FastAPI(title="notionchat", version="0.1.0")

# web을 별도 도메인으로 배포하는 경우를 위한 CORS. 배포 시 NOTIONCHAT_CORS_ORIGINS에
# 프론트 주소를 지정한다 (쉼표 구분). 미설정이면 동일 출처 서빙만 가정하고 전체 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_WEB = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list = Field(default_factory=list)


class LoginRequest(BaseModel):
    id: str
    password: str


def require_auth(request: Request) -> None:
    """인증 활성 시 유효한 세션 쿠키를 요구한다. 무효면 401 → 웹이 로그인 화면 표시."""
    if not auth.enabled():
        return
    if not auth.verify(request.cookies.get(auth.COOKIE_NAME, "")):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict:
    if not auth.enabled():
        auth_state = "off"
    elif auth.verify(request.cookies.get(auth.COOKIE_NAME, "")):
        auth_state = "ok"
    else:
        auth_state = "required"
    return {"status": "ok", "model": config.MODEL_ID, "region": config.AWS_REGION, "auth": auth_state}


@app.post("/api/login")
def login(req: LoginRequest, response: Response) -> dict:
    if not auth.enabled():
        return {"status": "ok"}
    if not auth.check_credentials(req.id, req.password):
        raise HTTPException(status_code=401, detail="ID 또는 비밀번호가 올바르지 않습니다")
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.make_token(),
        max_age=auth.SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=bool(os.environ.get("VERCEL")),  # 로컬 http 개발 허용, Vercel(https)에선 secure
    )
    return {"status": "ok"}


@app.post("/api/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(auth.COOKIE_NAME)
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest, _auth: None = Depends(require_auth)) -> StreamingResponse:
    def stream():
        events: queue.Queue = queue.Queue()
        terminal: dict = {}

        def on_log(message: str) -> None:
            events.put(("tool_log", {"message": message}))

        def worker() -> None:
            try:
                answer, history, usage = run_turn(req.question, req.history, on_log=on_log)
                terminal["event"] = ("answer", {"answer": answer, "history": history, "usage": usage})
            except ChatError as e:
                terminal["event"] = ("error", {"message": str(e)})
            except Exception as e:  # 마지막 방어선 — 스트림을 무한 대기로 두지 않는다
                terminal["event"] = ("error", {"message": f"서버 오류: {type(e).__name__}: {e}"})
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while (item := events.get()) is not None:
            yield _sse(*item)
        thread.join()
        yield _sse(*terminal["event"])

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 로컬 개발 편의용 동봉 서빙 — web을 별도 배포하면 이 마운트 없이 API만 동작한다
if _WEB.exists():
    app.mount("/web", StaticFiles(directory=_WEB), name="web")
