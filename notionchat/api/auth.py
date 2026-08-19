"""공유 계정 인증 — 서버 저장소 없는 서명 쿠키 세션 (서버리스 대응).

NOTIONCHAT_AUTH_ID / NOTIONCHAT_AUTH_PASSWORD 가 둘 다 설정된 경우에만 활성화된다
(미설정 = 로컬 개발용 무인증 모드). 세션은 만료시각을 HMAC 서명한 쿠키라 서버가
아무것도 저장하지 않으며, 서명 키를 계정 정보에서 유도하므로 비밀번호를 바꾸면
기존 세션이 전부 무효화된다. (키 직접 지정: NOTIONCHAT_AUTH_SECRET)
"""

import hashlib
import hmac
import time

from api import config

COOKIE_NAME = "nc_session"
SESSION_TTL = 60 * 60 * 24 * 30  # 30일


def _auth_id() -> str:
    return config.env("NOTIONCHAT_AUTH_ID", "")


def _auth_password() -> str:
    return config.env("NOTIONCHAT_AUTH_PASSWORD", "")


def _secret() -> bytes:
    override = config.env("NOTIONCHAT_AUTH_SECRET")
    if override:
        return override.encode()
    return hashlib.sha256(f"notionchat:{_auth_id()}:{_auth_password()}".encode()).digest()


def enabled() -> bool:
    return bool(_auth_id() and _auth_password())


def check_credentials(login_id: str, password: str) -> bool:
    id_ok = hmac.compare_digest(login_id.encode(), _auth_id().encode())
    pw_ok = hmac.compare_digest(password.encode(), _auth_password().encode())
    return id_ok and pw_ok


def make_token() -> str:
    expires = int(time.time()) + SESSION_TTL
    signature = hmac.new(_secret(), str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def verify(token: str) -> bool:
    expires_str, _, signature = token.partition(".")
    if not expires_str.isdigit() or not signature:
        return False
    if int(expires_str) < time.time():
        return False
    expected = hmac.new(_secret(), expires_str.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
