#!/usr/bin/env bash
#
# VLM 스파이크 로컬 환경 — Ollama 기동·모델 준비를 한 번에.
#
# 이 스크립트가 명령의 **정본**이다. README.md·spike-report.md·vlm_tag.py 독스트링에
# 흩어져 있던 설치/기동 명령을 여기로 모았다. 바꿀 일이 있으면 여기만 고친다.
#
#   ./setup_ollama.sh              기동 + 기본 모델(gemma3:12b) 확인. 매일 아침 이것만
#   ./setup_ollama.sh --compare    비교군(qwen2.5vl:7b)까지 받는다
#   ./setup_ollama.sh --status     지금 상태만 보고 아무것도 안 한다
#   ./setup_ollama.sh --stop       이 스크립트가 띄운 서버를 내린다
#   ./setup_ollama.sh --install    ollama 자체가 없을 때 brew로 설치까지
#
# 몇 번을 돌려도 안전하다 — 이미 떠 있으면 안 띄우고, 이미 받은 모델은 안 받는다.
# 데이터 원칙: 전부 로컬 추론이다. 이미지가 외부로 나가지 않는다 (CLAUDE.md).

set -euo pipefail

HOST="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
BASE_MODEL="gemma3:12b"          # vlm_tag.py --model 기본값과 같아야 한다
COMPARE_MODEL="qwen2.5vl:7b"     # tech-stack.md §2의 12B/7B 비교용
READY_TIMEOUT=60                 # 기동 후 응답을 기다리는 초

cd "$(dirname "$0")"
mkdir -p out
LOG="out/ollama.log"
PIDFILE="out/ollama.pid"

say()  { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die()  { printf '\n[중단] %s\n' "$*" >&2; exit 1; }

is_up() { curl -sf --max-time 3 "$HOST/api/version" >/dev/null 2>&1; }

version() { curl -s --max-time 3 "$HOST/api/version" | sed 's/[{}"]//g'; }

# ── 인자 ─────────────────────────────────────────────────────────────────────
WANT_COMPARE=0 DO_INSTALL=0 MODE=up
for arg in "$@"; do
  case "$arg" in
    --compare) WANT_COMPARE=1 ;;
    --install) DO_INSTALL=1 ;;
    --status)  MODE=status ;;
    --stop)    MODE=stop ;;
    -h|--help) sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "모르는 인자: $arg  (--help 참고)" ;;
  esac
done

# ── --stop ───────────────────────────────────────────────────────────────────
if [[ "$MODE" == stop ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE"
    say "이 스크립트가 띄운 ollama serve를 내렸다."
  else
    rm -f "$PIDFILE"
    say "이 스크립트가 띄운 서버가 없다."
    is_up && say "다만 $HOST 는 응답한다 — brew services나 다른 터미널이 띄운 것이다."
  fi
  exit 0
fi

# ── ollama 설치 확인 ─────────────────────────────────────────────────────────
head_ "1. ollama"
if ! command -v ollama >/dev/null 2>&1; then
  if [[ "$DO_INSTALL" == 1 ]]; then
    command -v brew >/dev/null 2>&1 || die "brew가 없다. https://brew.sh 먼저."
    say "brew install ollama ..."
    brew install ollama
  else
    die "ollama가 없다. 설치하려면:  ./setup_ollama.sh --install   (또는 brew install ollama)"
  fi
fi
say "$(command -v ollama)"

# ── 서버 기동 ────────────────────────────────────────────────────────────────
head_ "2. 서버"
if is_up; then
  say "이미 떠 있다 — $HOST  $(version)"
elif [[ "$MODE" == status ]]; then
  say "응답 없음 ($HOST). 인자 없이 실행하면 띄운다."
else
  # OLLAMA_FLASH_ATTENTION=1 — 08-25 실측을 이 설정으로 했다. 바꾸면 속도 수치가 달라진다.
  say "기동 중 ... (로그: $LOG)"
  OLLAMA_FLASH_ATTENTION=1 nohup ollama serve >>"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq "$READY_TIMEOUT"); do
    is_up && break
    sleep 1
  done
  is_up || die "$READY_TIMEOUT초 안에 응답이 없다. $LOG 를 볼 것."
  say "기동 완료 — $HOST  $(version)  (pid $(cat "$PIDFILE"))"
fi

# ── 모델 ─────────────────────────────────────────────────────────────────────
head_ "3. 모델"

# 데몬이 떠 있으면 `ollama list`가 정답이다. 꺼져 있으면 그게 실패하므로
# 매니페스트 파일을 직접 본다 — 안 그러면 "받아 놨는데 없음"이라고 거짓말을 한다.
MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"
have() {
  if is_up; then
    ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$1"
  else
    [[ -f "$MODELS_DIR/manifests/registry.ollama.ai/library/${1%%:*}/${1##*:}" ]]
  fi
}

ensure() {
  if have "$1"; then
    say "있음  $1"
  elif [[ "$MODE" == status ]]; then
    say "없음  $1  (인자 없이 실행하면 받는다)"
  else
    say "받는 중  $1  ... 수 GB, 처음 한 번만"
    ollama pull "$1"
  fi
}

ensure "$BASE_MODEL"
[[ "$WANT_COMPARE" == 1 ]] && ensure "$COMPARE_MODEL"
if [[ "$WANT_COMPARE" == 0 ]] && ! have "$COMPARE_MODEL"; then
  say "(비교군 $COMPARE_MODEL 은 없다. 필요하면 --compare)"
fi

# ── 요약 ─────────────────────────────────────────────────────────────────────
head_ "준비됨"
if is_up; then
  ollama list 2>/dev/null | sed 's/^/  /' || true
else
  say "(서버가 꺼져 있어 목록 생략)"
fi
say ""
say "모델 저장 위치: ${OLLAMA_MODELS:-$HOME/.ollama/models}  ($(du -sh "${OLLAMA_MODELS:-$HOME/.ollama}" 2>/dev/null | cut -f1)) — repo 밖이다"
say ""
say "다음:"
say "  .venv/bin/python vlm_tag.py --manifest out/manifest.csv --n 50"
say "  .venv/bin/python vlm_review.py --tags out/vlm_tags.csv --out out/vlm_review.html"
