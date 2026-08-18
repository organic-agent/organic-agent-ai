// notionchat 웹 UI — /api/chat SSE 스트림을 소비하는 채팅 클라이언트.
// 서버는 무상태: answer 이벤트로 받은 history를 저장했다가 다음 요청에 그대로 보낸다.

// web을 API와 다른 도메인에 배포할 때: index.html의 <script>에서
// window.NOTIONCHAT_API_BASE = "https://api.example.com" 지정. 미지정 시 동일 출처.
const API_BASE = window.NOTIONCHAT_API_BASE || "";

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const loginOverlay = document.getElementById("login");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const logoutBtn = document.getElementById("logout");

let history = []; // 서버가 준 불투명 값 — 수정 금지, 초기화는 []

function showLogin() {
  loginOverlay.classList.remove("hidden");
  document.getElementById("login-id").focus();
}

// 서버 상태 확인 + 인증 상태에 따라 로그인 화면/로그아웃 버튼 토글
async function refreshHealth() {
  try {
    const h = await (await fetch(`${API_BASE}/api/health`)).json();
    document.getElementById("meta").textContent = `${h.model} · ${h.region}`;
    if (h.auth === "required") showLogin();
    else loginOverlay.classList.add("hidden");
    logoutBtn.classList.toggle("hidden", h.auth === "off");
  } catch {
    /* 서버 미기동 등 — 채팅 시도 시 오류로 표면화됨 */
  }
}
refreshHealth();

loginForm.onsubmit = async (e) => {
  e.preventDefault();
  loginError.textContent = "";
  const res = await fetch(`${API_BASE}/api/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: document.getElementById("login-id").value,
      password: document.getElementById("login-pw").value,
    }),
  }).catch(() => null);
  if (res && res.ok) {
    document.getElementById("login-pw").value = "";
    loginOverlay.classList.add("hidden");
    refreshHealth();
    input.focus();
  } else {
    loginError.textContent = res ? (await res.json()).detail : "서버에 연결할 수 없습니다";
  }
};

logoutBtn.onclick = async () => {
  await fetch(`${API_BASE}/api/logout`, { method: "POST" }).catch(() => {});
  history = [];
  chat.innerHTML = "";
  showLogin();
};

document.getElementById("reset").onclick = () => {
  history = [];
  chat.innerHTML = "";
  input.focus();
};

function el(tag, cls, parent) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (parent) parent.appendChild(node);
  return node;
}

function addMessage(role, who) {
  const msg = el("div", `msg ${role}`, chat);
  el("div", "who", msg).textContent = who;
  return msg;
}

function renderMarkdown(target, text) {
  target.innerHTML = DOMPurify.sanitize(marked.parse(text));
  target.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight });
}

// SSE 스트림을 읽어 이벤트별 콜백을 호출한다
async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message", data = "";
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

async function askServer(question, msg) {
  const logsBox = el("details", "logs", msg);
  logsBox.open = true;
  const summary = el("summary", null, logsBox);
  summary.textContent = "⚙ 도구 호출";
  const typing = el("div", "typing", msg);
  typing.textContent = "Notion을 검색하는 중";
  let logCount = 0;

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  if (res.status === 401) {
    showLogin();
    throw new Error("로그인이 필요합니다");
  }
  if (!res.ok) throw new Error(`서버 응답 ${res.status}`);

  await readSSE(res, (event, data) => {
    if (event === "tool_log") {
      logCount += 1;
      summary.textContent = `⚙ 도구 호출 ${logCount}회`;
      el("div", null, logsBox).textContent = data.message;
      scrollDown();
    } else if (event === "answer") {
      history = data.history;
      typing.remove();
      logsBox.open = false;
      if (data.usage) {
        const u = data.usage;
        el("div", null, logsBox).textContent =
          `📊 토큰 — 입력 ${u.inputTokens} · 출력 ${u.outputTokens} · 캐시 읽기 ${u.cacheReadInputTokens}`;
        summary.textContent = `⚙ 도구 호출 ${logCount}회 · 토큰 ${u.inputTokens + u.outputTokens}`;
      }
      if (logCount === 0 && !data.usage) logsBox.remove();
      renderMarkdown(el("div", "bubble", msg), data.answer);
    } else if (event === "error") {
      typing.remove();
      msg.classList.add("error");
      el("div", "bubble", msg).textContent = data.message;
    }
  });
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  send.disabled = true;

  const userMsg = addMessage("user", "나");
  el("div", "bubble", userMsg).textContent = question;
  const botMsg = addMessage("assistant", "notionchat");
  scrollDown();

  try {
    await askServer(question, botMsg);
  } catch (err) {
    botMsg.classList.add("error");
    el("div", "bubble", botMsg).textContent = `요청 실패: ${err.message}`;
  } finally {
    send.disabled = false;
    input.focus();
    scrollDown();
  }
};
