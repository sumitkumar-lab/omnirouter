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

const defaultApiBase =
  window.location.protocol === "file:" || window.location.hostname.includes("github.io")
    ? "http://127.0.0.1:8000"
    : window.location.origin;

apiInput.value = localStorage.getItem("omnirouter.apiBase") || defaultApiBase;

saveApiButton.addEventListener("click", () => {
  localStorage.setItem("omnirouter.apiBase", apiInput.value.trim());
  checkHealth();
  loadDocuments();
});

refreshDocsButton.addEventListener("click", loadDocuments);

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    uploadStatus.textContent = "Choose at least one document.";
    return;
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading and indexing...";
  try {
    const result = await uploadDocuments(files);
    uploadStatus.textContent = `Uploaded ${result.uploaded.length} file(s). Corpus ${result.version || "not synced"}: ${result.source_count} source(s), ${result.chunk_count} chunk(s).`;
    fileInput.value = "";
    await loadDocuments();
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
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

  const agentMessage = appendMessage("agent", "");
  try {
    await streamAgentResponse(query, agentMessage);
  } catch (error) {
    agentMessage.textContent = `Request failed: ${error.message}`;
  } finally {
    sendButton.disabled = false;
    queryInput.focus();
  }
});

function apiBase() {
  return apiInput.value.trim().replace(/\/$/, "");
}

async function checkHealth() {
  statusEl.textContent = "Checking API...";
  try {
    const response = await fetch(`${apiBase()}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    statusEl.textContent = data.status === "ok" ? "API online" : "API unknown";
  } catch {
    statusEl.textContent = "API offline";
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
  }
  if (eventName === "token") {
    targetEl.textContent += payload.token || "";
    chatLog.scrollTop = chatLog.scrollHeight;
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

checkHealth();
loadDocuments();

function formatBytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
