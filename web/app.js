const apiInput = document.querySelector("#apiBase");
const saveApiButton = document.querySelector("#saveApi");
const statusEl = document.querySelector("#status");
const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const queryInput = document.querySelector("#query");
const sendButton = document.querySelector("#send");
const useCacheInput = document.querySelector("#useCache");
const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#documentFiles");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const documentList = document.querySelector("#documentList");
const refreshDocsButton = document.querySelector("#refreshDocs");
const syncAfterUploadInput = document.querySelector("#syncAfterUpload");
const agentStateText = document.querySelector("#agentStateText");

const defaultApiBase =
  window.location.protocol === "file:" || window.location.hostname.includes("github.io")
    ? "http://127.0.0.1:8000"
    : window.location.origin;

apiInput.value = localStorage.getItem("omnirouter.apiBase") || defaultApiBase;

saveApiButton.addEventListener("click", () => {
  localStorage.setItem("omnirouter.apiBase", apiInput.value.trim());
  setAgentMode("checking");
  checkHealth();
  loadDocuments();
});

refreshDocsButton.addEventListener("click", () => {
  pulseAgentField("documents");
  loadDocuments();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    uploadStatus.textContent = "Choose at least one document.";
    return;
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading and indexing...";
  setAgentMode("syncing");
  pulseAgentField("upload");
  try {
    const result = await uploadDocuments(files);
    uploadStatus.textContent = `Uploaded ${result.uploaded.length} file(s). Corpus ${result.version || "not synced"}: ${result.source_count} source(s), ${result.chunk_count} chunk(s).`;
    fileInput.value = "";
    await loadDocuments();
    setAgentMode("ready");
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
    setAgentMode("error");
  } finally {
    uploadButton.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";
  sendButton.disabled = true;
  setAgentMode("thinking");
  pulseAgentField("query");

  const agentMessage = appendMessage("agent", "");
  try {
    await streamAgentResponse(query, agentMessage);
  } catch (error) {
    agentMessage.textContent = `Request failed: ${error.message}`;
    setAgentMode("error");
  } finally {
    sendButton.disabled = false;
    queryInput.focus();
    if (!agentMessage.textContent.startsWith("Request failed:")) {
      setAgentMode("ready");
    }
  }
});

function apiBase() {
  return apiInput.value.trim().replace(/\/$/, "");
}

async function checkHealth() {
  statusEl.textContent = "Checking API...";
  setAgentMode("checking");
  try {
    const response = await fetch(`${apiBase()}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    statusEl.textContent = data.status === "ok" ? "API online" : "API unknown";
    setAgentMode(data.status === "ok" ? "ready" : "checking");
  } catch {
    statusEl.textContent = "API offline";
    setAgentMode("offline");
  }
}

async function uploadDocuments(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("sync", syncAfterUploadInput.checked ? "true" : "false");

  const response = await fetch(`${apiBase()}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }
  return response.json();
}

async function loadDocuments() {
  try {
    const response = await fetch(`${apiBase()}/documents`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderDocuments(data.documents || []);
  } catch {
    documentList.innerHTML = "<li>Document list unavailable.</li>";
  }
}

function renderDocuments(documents) {
  if (documents.length === 0) {
    documentList.innerHTML = "<li>No uploaded documents yet.</li>";
    return;
  }
  documentList.innerHTML = "";
  for (const doc of documents) {
    const item = window.document.createElement("li");
    item.textContent = `${doc.path} (${formatBytes(doc.size_bytes)})`;
    documentList.appendChild(item);
  }
}

async function streamAgentResponse(query, targetEl) {
  const response = await fetch(`${apiBase()}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      use_cache: useCacheInput.checked,
      thread_id: getThreadId(),
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`API returned HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  setAgentMode("thinking");

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const eventText of events) {
      handleSseEvent(eventText, targetEl);
    }
  }
}

function handleSseEvent(eventText, targetEl) {
  const lines = eventText.split("\n");
  const eventName = (lines.find((line) => line.startsWith("event:")) || "event: token")
    .replace("event:", "")
    .trim();
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!dataLine) return;

  const payload = JSON.parse(dataLine.replace("data:", "").trim());
  if (eventName === "meta") {
    appendMessage("meta", payload.source === "cache" ? "Loaded from cache" : `Thread ${payload.thread_id}`);
    pulseAgentField(payload.source === "cache" ? "cache" : "thread");
  }
  if (eventName === "token") {
    targetEl.textContent += payload.token || "";
    chatLog.scrollTop = chatLog.scrollHeight;
    pulseAgentField("token");
  }
}

function appendMessage(kind, text) {
  const element = document.createElement("div");
  element.className = `message ${kind}`;
  element.textContent = text;
  chatLog.appendChild(element);
  chatLog.scrollTop = chatLog.scrollHeight;
  return element;
}

function getThreadId() {
  const key = "omnirouter.threadId";
  let threadId = localStorage.getItem(key);
  if (!threadId) {
    threadId = `web-${crypto.randomUUID()}`;
    localStorage.setItem(key, threadId);
  }
  return threadId;
}

function formatBytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function createAgentField() {
  const canvas = document.querySelector("#agentCanvas");
  if (!canvas) return { setMode() {}, pulse() {} };

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const context = canvas.getContext("2d", { alpha: true });
  const palette = ["#3157d5", "#08a7a1", "#d68a1f", "#db5860"];
  const labels = ["Planner", "Retriever", "Critic", "Coder", "Synthesizer", "Verifier", "Memory", "Router"];
  const modes = {
    checking: { label: "Agents connecting", speed: 0.58, packets: 0.018, glow: 0.7 },
    ready: { label: "Agents in sync", speed: 0.72, packets: 0.032, glow: 0.9 },
    thinking: { label: "Agents collaborating", speed: 1.42, packets: 0.092, glow: 1.35 },
    syncing: { label: "Agents indexing", speed: 1.05, packets: 0.07, glow: 1.15 },
    offline: { label: "Agents waiting", speed: 0.28, packets: 0.006, glow: 0.45 },
    error: { label: "Agents need attention", speed: 0.38, packets: 0.012, glow: 0.65 },
  };

  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let modeName = "checking";
  let mode = modes.checking;
  let agents = [];
  let packets = [];
  let ripples = [];
  let frame = 0;
  let lastPulse = 0;

  function resize() {
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    agents = buildAgents();
  }

  function buildAgents() {
    const count = width < 720 ? 9 : 16;
    const centerX = width * 0.5;
    const centerY = height * 0.48;
    const radiusX = Math.max(190, width * 0.38);
    const radiusY = Math.max(150, height * 0.29);

    return Array.from({ length: count }, (_, index) => {
      const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
      return {
        label: labels[index % labels.length],
        color: palette[index % palette.length],
        baseX: centerX + Math.cos(angle) * radiusX,
        baseY: centerY + Math.sin(angle) * radiusY,
        x: 0,
        y: 0,
        angle,
        orbit: 22 + (index % 4) * 8,
        drift: 0.5 + (index % 5) * 0.12,
        phase: index * 0.72,
        size: index % 3 === 0 ? 5.5 : 4.5,
      };
    });
  }

  function setMode(nextMode) {
    modeName = modes[nextMode] ? nextMode : "ready";
    mode = modes[modeName];
    if (agentStateText) {
      agentStateText.textContent = mode.label;
    }
  }

  function pulse(kind = "token") {
    const now = performance.now();
    if (kind === "token" && now - lastPulse < 120) return;
    lastPulse = now;

    const source = agents[Math.floor(Math.random() * agents.length)];
    const target = agents[Math.floor(Math.random() * agents.length)];
    if (!source || !target || source === target) return;

    packets.push({
      source,
      target,
      progress: 0,
      speed: kind === "token" ? 0.026 : 0.018,
      color: kind === "cache" ? "#d68a1f" : source.color,
      radius: kind === "query" || kind === "upload" ? 3.6 : 2.5,
    });

    ripples.push({
      x: source.x || source.baseX,
      y: source.y || source.baseY,
      radius: 8,
      alpha: 0.34,
      color: source.color,
    });
  }

  function draw(timestamp) {
    frame += reducedMotion.matches ? 0.2 : mode.speed;
    context.clearRect(0, 0, width, height);

    updateAgents(timestamp);
    drawMesh();
    drawPackets();
    drawRipples();
    drawAgents();

    if (!reducedMotion.matches && Math.random() < mode.packets) {
      pulse("ambient");
    }

    requestAnimationFrame(draw);
  }

  function updateAgents(timestamp) {
    const time = timestamp * 0.00024;
    for (const agent of agents) {
      agent.x = agent.baseX + Math.cos(time * agent.drift + agent.phase) * agent.orbit;
      agent.y = agent.baseY + Math.sin(time * agent.drift * 1.4 + agent.phase) * agent.orbit;
    }
  }

  function drawMesh() {
    for (let i = 0; i < agents.length; i += 1) {
      for (let j = i + 1; j < agents.length; j += 1) {
        const a = agents[i];
        const b = agents[j];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        const limit = width < 720 ? 250 : 330;
        if (distance > limit) continue;

        const alpha = (1 - distance / limit) * 0.34 * mode.glow;
        const gradient = context.createLinearGradient(a.x, a.y, b.x, b.y);
        gradient.addColorStop(0, hexToRgba(a.color, alpha));
        gradient.addColorStop(1, hexToRgba(b.color, alpha));
        context.strokeStyle = gradient;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
    }
  }

  function drawPackets() {
    packets = packets.filter((packet) => packet.progress < 1);
    for (const packet of packets) {
      packet.progress += packet.speed * mode.speed;
      const eased = easeInOut(packet.progress);
      const x = lerp(packet.source.x, packet.target.x, eased);
      const y = lerp(packet.source.y, packet.target.y, eased);

      context.shadowBlur = 18;
      context.shadowColor = packet.color;
      context.fillStyle = packet.color;
      context.beginPath();
      context.arc(x, y, packet.radius, 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;
    }
  }

  function drawRipples() {
    ripples = ripples.filter((ripple) => ripple.alpha > 0.01);
    for (const ripple of ripples) {
      ripple.radius += 0.72 * mode.speed;
      ripple.alpha *= 0.95;
      context.strokeStyle = hexToRgba(ripple.color, ripple.alpha);
      context.lineWidth = 1;
      context.beginPath();
      context.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
      context.stroke();
    }
  }

  function drawAgents() {
    const showLabels = width > 760;
    for (const agent of agents) {
      const halo = 22 + Math.sin(frame * 0.03 + agent.phase) * 5;
      const gradient = context.createRadialGradient(agent.x, agent.y, 0, agent.x, agent.y, halo);
      gradient.addColorStop(0, hexToRgba(agent.color, 0.7 * mode.glow));
      gradient.addColorStop(1, hexToRgba(agent.color, 0));
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(agent.x, agent.y, halo, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = agent.color;
      context.beginPath();
      context.arc(agent.x, agent.y, agent.size, 0, Math.PI * 2);
      context.fill();

      context.strokeStyle = "rgba(255, 255, 255, 0.82)";
      context.lineWidth = 1.5;
      context.stroke();

      if (showLabels) {
        context.font = "600 11px Inter, ui-sans-serif, system-ui";
        context.fillStyle = "rgba(31, 41, 55, 0.72)";
        context.fillText(agent.label, agent.x + 10, agent.y - 8);
      }
    }
  }

  function hexToRgba(hex, alpha) {
    const value = hex.replace("#", "");
    const red = parseInt(value.slice(0, 2), 16);
    const green = parseInt(value.slice(2, 4), 16);
    const blue = parseInt(value.slice(4, 6), 16);
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }

  function lerp(start, end, amount) {
    return start + (end - start) * amount;
  }

  function easeInOut(value) {
    return value < 0.5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2;
  }

  window.addEventListener("resize", resize);
  resize();
  setMode("checking");
  requestAnimationFrame(draw);

  return { setMode, pulse };
}

const agentField = createAgentField();

function setAgentMode(mode) {
  agentField.setMode(mode);
}

function pulseAgentField(kind) {
  agentField.pulse(kind);
}

checkHealth();
loadDocuments();
