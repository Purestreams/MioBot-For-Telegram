const I18N = {
  zh: {
    "login.title": "后台管理",
    "login.tokenLabel": "登录令牌",
    "login.button": "登录",
    "login.hint": "在 Telegram 私聊里使用 /webadmin_token 生成临时令牌。",
    "login.empty": "请输入登录令牌。",
    "login.failed": "登录失败，令牌可能已过期或已使用。",
    "app.title": "后台管理",
    "app.logout": "退出",
    "nav.dashboard": "概览",
    "nav.chats": "聊天记录",
    "nav.memory": "个人记忆",
    "nav.global": "全局记忆",
    "actions.refresh": "刷新",
    "actions.search": "搜索",
    "actions.save": "保存",
    "actions.delete": "删除",
    "actions.add": "添加",
    "actions.accept": "接受",
    "actions.reject": "拒绝",
    "dashboard.title": "概览",
    "dashboard.messages": "消息数",
    "dashboard.chats": "聊天数",
    "dashboard.userMemories": "用户记忆",
    "dashboard.userFacts": "个人事实",
    "dashboard.pendingCandidates": "待审候选",
    "dashboard.globalFacts": "全局事实",
    "dashboard.schema": "DB 版本",
    "chats.title": "聊天记录",
    "chats.chatList": "Chat ID",
    "chats.messages": "消息",
    "chats.search": "搜索消息",
    "chats.pickChat": "选择一个 chat_id 查看消息。",
    "memory.title": "个人记忆",
    "memory.users": "用户",
    "memory.detail": "详情",
    "memory.pickUser": "选择一个用户查看记忆。",
    "memory.summary": "摘要",
    "memory.facts": "事实",
    "memory.candidates": "候选",
    "memory.noFacts": "没有事实。",
    "memory.noCandidates": "没有待审候选。",
    "global.title": "全局记忆",
    "global.chats": "群聊",
    "global.detail": "记忆事实",
    "global.pickChat": "选择一个 chat_id 查看全局记忆。",
    "global.noFacts": "没有全局记忆。",
    "fields.id": "ID",
    "fields.chatId": "chat_id",
    "fields.messages": "消息",
    "fields.globalFacts": "全局事实",
    "fields.latest": "最新",
    "fields.user": "用户",
    "fields.facts": "事实",
    "fields.summary": "摘要",
    "fields.refreshed": "刷新日期",
    "fields.type": "类型",
    "fields.text": "内容",
    "fields.confidence": "置信度",
    "fields.priority": "优先级",
    "fields.status": "状态",
    "fields.timestamp": "时间",
    "fields.username": "用户名",
    "toast.saved": "已保存。",
    "toast.deleted": "已删除。",
    "toast.added": "已添加。",
    "toast.accepted": "已接受。",
    "toast.rejected": "已拒绝。",
    "errors.auth": "需要重新登录。",
    "errors.request": "请求失败。"
  },
  en: {
    "login.title": "Web Admin",
    "login.tokenLabel": "Login token",
    "login.button": "Sign in",
    "login.hint": "Create a temporary token in a private Telegram chat with /webadmin_token.",
    "login.empty": "Enter a login token.",
    "login.failed": "Sign-in failed. The token may be expired or already used.",
    "app.title": "Web Admin",
    "app.logout": "Sign out",
    "nav.dashboard": "Dashboard",
    "nav.chats": "Chat History",
    "nav.memory": "User Memory",
    "nav.global": "Global Memory",
    "actions.refresh": "Refresh",
    "actions.search": "Search",
    "actions.save": "Save",
    "actions.delete": "Delete",
    "actions.add": "Add",
    "actions.accept": "Accept",
    "actions.reject": "Reject",
    "dashboard.title": "Dashboard",
    "dashboard.messages": "Messages",
    "dashboard.chats": "Chats",
    "dashboard.userMemories": "User memories",
    "dashboard.userFacts": "User facts",
    "dashboard.pendingCandidates": "Pending candidates",
    "dashboard.globalFacts": "Global facts",
    "dashboard.schema": "DB schema",
    "chats.title": "Chat History",
    "chats.chatList": "Chat IDs",
    "chats.messages": "Messages",
    "chats.search": "Search messages",
    "chats.pickChat": "Select a chat_id to view messages.",
    "memory.title": "User Memory",
    "memory.users": "Users",
    "memory.detail": "Detail",
    "memory.pickUser": "Select a user to view memory.",
    "memory.summary": "Summary",
    "memory.facts": "Facts",
    "memory.candidates": "Candidates",
    "memory.noFacts": "No facts.",
    "memory.noCandidates": "No pending candidates.",
    "global.title": "Global Memory",
    "global.chats": "Chats",
    "global.detail": "Memory facts",
    "global.pickChat": "Select a chat_id to view global memory.",
    "global.noFacts": "No global memory.",
    "fields.id": "ID",
    "fields.chatId": "chat_id",
    "fields.messages": "Messages",
    "fields.globalFacts": "Global facts",
    "fields.latest": "Latest",
    "fields.user": "User",
    "fields.facts": "Facts",
    "fields.summary": "Summary",
    "fields.refreshed": "Refreshed",
    "fields.type": "Type",
    "fields.text": "Text",
    "fields.confidence": "Confidence",
    "fields.priority": "Priority",
    "fields.status": "Status",
    "fields.timestamp": "Time",
    "fields.username": "Username",
    "toast.saved": "Saved.",
    "toast.deleted": "Deleted.",
    "toast.added": "Added.",
    "toast.accepted": "Accepted.",
    "toast.rejected": "Rejected.",
    "errors.auth": "Sign in again.",
    "errors.request": "Request failed."
  }
};

const state = {
  lang: localStorage.getItem("miobot.webadmin.lang") || "zh",
  view: "dashboard",
  selectedChatId: null,
  selectedUserKey: null,
  selectedGlobalChatId: null,
  admin: null
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function text(key) {
  return I18N[state.lang]?.[key] || I18N.en[key] || key;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function applyI18n() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  $$('[data-i18n]').forEach((node) => {
    node.textContent = text(node.dataset.i18n);
  });
  $$('[data-i18n-placeholder]').forEach((node) => {
    node.placeholder = text(node.dataset.i18nPlaceholder);
  });
  $$('.lang-button').forEach((button) => {
    button.classList.toggle('active', button.dataset.lang === state.lang);
  });
}

function setLanguage(lang) {
  state.lang = lang === "en" ? "en" : "zh";
  localStorage.setItem("miobot.webadmin.lang", state.lang);
  applyI18n();
  refreshCurrentView(false);
}

function showToast(message, isError = false) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 2600);
}

function setLoginStatus(message, kind = "") {
  const status = $('#loginStatus');
  status.textContent = message || "";
  status.className = `status-text ${kind}`.trim();
}

async function api(path, options = {}) {
  const init = { ...options };
  init.headers = { Accept: "application/json", ...(options.headers || {}) };
  if (init.body && typeof init.body !== "string") {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
  const responseText = await response.text();
  let data = null;
  if (responseText) {
    try {
      data = JSON.parse(responseText);
    } catch (_error) {
      data = { detail: responseText };
    }
  }
  if (response.status === 401) {
    showLogin();
    throw new Error(text("errors.auth"));
  }
  if (!response.ok) {
    throw new Error(data?.detail || text("errors.request"));
  }
  return data || {};
}

function showLogin() {
  $('#loginView').hidden = false;
  $('#appView').hidden = true;
}

function showApp(admin) {
  state.admin = admin || state.admin || {};
  $('#loginView').hidden = true;
  $('#appView').hidden = false;
  $('#adminIdentity').textContent = state.admin.admin_username || state.admin.admin_user_id || "admin";
  refreshCurrentView(true);
}

async function login() {
  const token = $('#tokenInput').value.trim();
  if (!token) {
    setLoginStatus(text("login.empty"), "error");
    return;
  }
  try {
    const result = await api('/api/auth/login', { method: 'POST', body: { token } });
    setLoginStatus("");
    $('#tokenInput').value = "";
    showApp(result.admin);
  } catch (_error) {
    setLoginStatus(text("login.failed"), "error");
  }
}

async function logout() {
  try {
    await api('/api/auth/logout', { method: 'POST' });
  } catch (_error) {
    // Clearing the local view is enough after a failed logout request.
  }
  state.admin = null;
  showLogin();
}

function showView(view) {
  state.view = view;
  $$('.nav-item').forEach((button) => button.classList.toggle('active', button.dataset.view === view));
  $$('.view').forEach((section) => section.classList.toggle('active', section.id === `${view}View`));
  refreshCurrentView(true);
}

function refreshCurrentView(showErrors) {
  if ($('#appView').hidden) return;
  const task = {
    dashboard: loadDashboard,
    chats: loadChats,
    memory: loadUsers,
    global: loadGlobalChats
  }[state.view];
  if (task) {
    task().catch((error) => {
      if (showErrors) showToast(error.message || text("errors.request"), true);
    });
  }
}

function renderTable(container, columns, rows, options = {}) {
  clear(container);
  if (!rows.length) {
    container.appendChild(element('div', 'empty-state', options.emptyText || ''));
    return;
  }

  const table = element('table', 'data-table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  columns.forEach((column) => {
    const th = document.createElement('th');
    th.textContent = column.label;
    if (column.width) th.style.width = column.width;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    if (options.onSelect) tr.classList.add('selectable');
    if (options.selectedId !== undefined && options.getId && options.getId(row) === options.selectedId) {
      tr.classList.add('selected');
    }
    columns.forEach((column) => {
      const td = document.createElement('td');
      const value = column.render ? column.render(row) : row[column.key];
      td.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
      tr.appendChild(td);
    });
    if (options.onSelect) {
      tr.addEventListener('click', () => options.onSelect(row));
    }
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

async function loadDashboard() {
  const stats = await api('/api/dashboard');
  const metrics = [
    [text("dashboard.messages"), stats.message_count],
    [text("dashboard.chats"), stats.chat_count],
    [text("dashboard.userMemories"), stats.user_memory_count],
    [text("dashboard.userFacts"), stats.user_memory_fact_count],
    [text("dashboard.pendingCandidates"), stats.pending_candidate_count],
    [text("dashboard.globalFacts"), stats.global_memory_fact_count],
    [text("dashboard.schema"), stats.db_schema_version]
  ];
  const grid = $('#metricGrid');
  clear(grid);
  metrics.forEach(([label, value]) => {
    const box = element('section', 'metric');
    box.appendChild(element('div', 'meta', label));
    box.appendChild(element('div', 'metric-value', value ?? 0));
    grid.appendChild(box);
  });
}

async function loadChats() {
  const result = await api('/api/chats?limit=120');
  renderTable($('#chatList'), [
    { label: text("fields.chatId"), render: (row) => row.chat_id, width: '130px' },
    { label: text("fields.messages"), render: (row) => row.message_count, width: '86px' },
    { label: text("fields.globalFacts"), render: (row) => row.global_fact_count, width: '96px' },
    { label: text("fields.latest"), render: (row) => row.latest_message_at || '-' }
  ], result.chats || [], {
    selectedId: state.selectedChatId,
    getId: (row) => row.chat_id,
    emptyText: text("chats.pickChat"),
    onSelect: (row) => {
      state.selectedChatId = row.chat_id;
      $('#chatDetailTitle').textContent = `${text("chats.messages")} ${row.chat_id}`;
      loadMessages().catch((error) => showToast(error.message, true));
      loadChats().catch(() => {});
    }
  });
}

async function loadMessages() {
  const container = $('#messageList');
  if (!state.selectedChatId) {
    container.className = 'message-list empty-state';
    container.textContent = text("chats.pickChat");
    return;
  }
  const query = new URLSearchParams({ limit: '160' });
  const search = $('#messageSearch').value.trim();
  if (search) query.set('q', search);
  const result = await api(`/api/chats/${state.selectedChatId}/messages?${query.toString()}`);
  clear(container);
  container.className = 'message-list';
  const rows = result.messages || [];
  if (!rows.length) {
    container.appendChild(element('div', 'empty-state', text("chats.messages")));
    return;
  }
  rows.forEach((message) => {
    const item = element('article', 'message-row');
    const meta = element('div', 'message-meta');
    [
      `#${message.id}`,
      message.timestamp,
      message.username,
      message.telegram_user_key || '',
      message.reply_to_username ? `reply: ${message.reply_to_username}` : ''
    ].filter(Boolean).forEach((value) => meta.appendChild(element('span', '', value)));
    item.appendChild(meta);
    item.appendChild(element('div', 'message-content', message.content));
    container.appendChild(item);
  });
}

async function loadUsers() {
  const result = await api('/api/memory/users?limit=160');
  renderTable($('#userList'), [
    { label: text("fields.user"), render: (row) => row.latest_display_name || row.telegram_user_key },
    { label: text("fields.facts"), render: (row) => row.fact_count, width: '70px' },
    { label: text("fields.refreshed"), render: (row) => row.last_refreshed_date || '-', width: '110px' }
  ], result.users || [], {
    selectedId: state.selectedUserKey,
    getId: (row) => row.telegram_user_key,
    emptyText: text("memory.pickUser"),
    onSelect: (row) => {
      state.selectedUserKey = row.telegram_user_key;
      $('#memoryDetailTitle').textContent = row.latest_display_name || row.telegram_user_key;
      loadUserDetail().catch((error) => showToast(error.message, true));
      loadUsers().catch(() => {});
    }
  });
}

async function loadUserDetail() {
  const container = $('#memoryDetail');
  if (!state.selectedUserKey) {
    container.className = 'empty-state';
    container.textContent = text("memory.pickUser");
    return;
  }
  const result = await api(`/api/memory/users/${encodeURIComponent(state.selectedUserKey)}`);
  clear(container);
  container.className = '';

  const summaryBlock = element('section', 'memory-summary');
  summaryBlock.appendChild(element('h3', '', text("memory.summary")));
  const textarea = element('textarea', 'textarea');
  textarea.value = result.memory?.memory_text || '';
  summaryBlock.appendChild(textarea);
  const saveSummary = element('button', 'primary-button', text("actions.save"));
  saveSummary.type = 'button';
  saveSummary.addEventListener('click', async () => {
    await api(`/api/memory/users/${encodeURIComponent(state.selectedUserKey)}/summary`, {
      method: 'PUT',
      body: { memory_text: textarea.value }
    });
    showToast(text("toast.saved"));
    await loadUsers();
  });
  summaryBlock.appendChild(saveSummary);
  container.appendChild(summaryBlock);

  container.appendChild(element('h3', '', text("memory.facts")));
  const factList = element('div', 'fact-list');
  const facts = result.facts || [];
  if (!facts.length) factList.appendChild(element('div', 'empty-state', text("memory.noFacts")));
  facts.forEach((fact) => factList.appendChild(renderUserFactRow(fact)));
  container.appendChild(factList);

  container.appendChild(element('h3', '', text("memory.candidates")));
  const candidateList = element('div', 'candidate-list');
  const candidates = result.candidates || [];
  if (!candidates.length) candidateList.appendChild(element('div', 'empty-state', text("memory.noCandidates")));
  candidates.forEach((candidate) => candidateList.appendChild(renderCandidateRow(candidate)));
  container.appendChild(candidateList);
}

function renderUserFactRow(fact) {
  const row = element('div', 'fact-row');
  const grid = element('div', 'fact-grid');
  const typeInput = element('input', 'text-input');
  typeInput.value = fact.fact_type || 'note';
  const textInput = element('input', 'text-input');
  textInput.value = fact.fact_text || '';
  const confidenceInput = element('input', 'text-input');
  confidenceInput.value = fact.confidence ?? 0.5;
  const actions = element('div', 'row-actions');
  const saveButton = element('button', 'primary-button', text("actions.save"));
  const deleteButton = element('button', 'danger-button', text("actions.delete"));
  saveButton.type = deleteButton.type = 'button';
  saveButton.addEventListener('click', async () => {
    await api(`/api/memory/facts/${fact.id}`, {
      method: 'PATCH',
      body: {
        fact_type: typeInput.value,
        fact_text: textInput.value,
        confidence: Number(confidenceInput.value)
      }
    });
    showToast(text("toast.saved"));
    await loadUserDetail();
  });
  deleteButton.addEventListener('click', async () => {
    await api(`/api/memory/facts/${fact.id}`, { method: 'DELETE' });
    showToast(text("toast.deleted"));
    await loadUserDetail();
  });
  actions.append(saveButton, deleteButton);
  grid.append(typeInput, textInput, confidenceInput, actions);
  row.appendChild(element('div', 'meta', `#${fact.id}`));
  row.appendChild(grid);
  return row;
}

function renderCandidateRow(candidate) {
  const row = element('div', 'candidate-row');
  row.appendChild(element('div', 'meta', `#${candidate.id} | ${candidate.priority}/${candidate.fact_type} | ${candidate.confidence}`));
  row.appendChild(element('div', 'message-content', candidate.fact_text));
  const actions = element('div', 'row-actions');
  const accept = element('button', 'primary-button', text("actions.accept"));
  const reject = element('button', 'danger-button', text("actions.reject"));
  accept.type = reject.type = 'button';
  accept.addEventListener('click', async () => {
    await api(`/api/memory/candidates/${candidate.id}/accept`, { method: 'POST' });
    showToast(text("toast.accepted"));
    await loadUserDetail();
    await loadUsers();
  });
  reject.addEventListener('click', async () => {
    await api(`/api/memory/candidates/${candidate.id}/reject`, { method: 'POST' });
    showToast(text("toast.rejected"));
    await loadUserDetail();
  });
  actions.append(accept, reject);
  row.appendChild(actions);
  return row;
}

async function loadGlobalChats() {
  const result = await api('/api/global-memory/chats?limit=160');
  renderTable($('#globalChatList'), [
    { label: text("fields.chatId"), render: (row) => row.chat_id, width: '130px' },
    { label: text("fields.globalFacts"), render: (row) => row.global_fact_count, width: '96px' },
    { label: text("fields.latest"), render: (row) => row.latest_message_at || '-' }
  ], result.chats || [], {
    selectedId: state.selectedGlobalChatId,
    getId: (row) => row.chat_id,
    emptyText: text("global.pickChat"),
    onSelect: (row) => {
      state.selectedGlobalChatId = row.chat_id;
      $('#globalDetailTitle').textContent = `${text("global.detail")} ${row.chat_id}`;
      loadGlobalDetail().catch((error) => showToast(error.message, true));
      loadGlobalChats().catch(() => {});
    }
  });
}

async function loadGlobalDetail() {
  const container = $('#globalFactList');
  if (!state.selectedGlobalChatId) {
    container.className = 'empty-state';
    container.textContent = text("global.pickChat");
    return;
  }
  const result = await api(`/api/global-memory/chats/${state.selectedGlobalChatId}`);
  clear(container);
  container.className = 'fact-list';
  const facts = result.facts || [];
  if (!facts.length) {
    container.appendChild(element('div', 'empty-state', text("global.noFacts")));
    return;
  }
  facts.forEach((fact) => container.appendChild(renderGlobalFactRow(fact)));
}

function renderGlobalFactRow(fact) {
  const row = element('div', 'fact-row');
  row.appendChild(element('div', 'meta', `#${fact.id}`));
  const grid = element('div', 'fact-grid');
  const typeInput = element('input', 'text-input');
  typeInput.value = fact.fact_type || 'note';
  const textInput = element('input', 'text-input');
  textInput.value = fact.fact_text || '';
  const confidenceInput = element('input', 'text-input');
  confidenceInput.value = fact.confidence ?? 0.5;
  const actions = element('div', 'row-actions');
  const saveButton = element('button', 'primary-button', text("actions.save"));
  const deleteButton = element('button', 'danger-button', text("actions.delete"));
  saveButton.type = deleteButton.type = 'button';
  saveButton.addEventListener('click', async () => {
    await api(`/api/global-memory/chats/${state.selectedGlobalChatId}/facts/${fact.id}`, {
      method: 'PATCH',
      body: {
        fact_type: typeInput.value,
        fact_text: textInput.value,
        confidence: Number(confidenceInput.value)
      }
    });
    showToast(text("toast.saved"));
    await loadGlobalDetail();
  });
  deleteButton.addEventListener('click', async () => {
    await api(`/api/global-memory/chats/${state.selectedGlobalChatId}/facts/${fact.id}`, { method: 'DELETE' });
    showToast(text("toast.deleted"));
    await loadGlobalDetail();
    await loadGlobalChats();
  });
  actions.append(saveButton, deleteButton);
  grid.append(typeInput, textInput, confidenceInput, actions);
  row.appendChild(grid);
  return row;
}

async function addGlobalFact() {
  if (!state.selectedGlobalChatId) {
    showToast(text("global.pickChat"), true);
    return;
  }
  const factText = $('#globalFactText').value.trim();
  if (!factText) return;
  await api(`/api/global-memory/chats/${state.selectedGlobalChatId}/facts`, {
    method: 'POST',
    body: {
      fact_type: $('#globalFactType').value.trim() || 'note',
      fact_text: factText,
      confidence: Number($('#globalFactConfidence').value || 0.9)
    }
  });
  $('#globalFactText').value = '';
  showToast(text("toast.added"));
  await loadGlobalDetail();
  await loadGlobalChats();
}

function bindEvents() {
  $$('.lang-button').forEach((button) => button.addEventListener('click', () => setLanguage(button.dataset.lang)));
  $('#loginButton').addEventListener('click', login);
  $('#tokenInput').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') login();
  });
  $('#logoutButton').addEventListener('click', logout);
  $$('.nav-item').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
  $('#refreshDashboard').addEventListener('click', () => loadDashboard().catch((error) => showToast(error.message, true)));
  $('#refreshChats').addEventListener('click', () => loadChats().catch((error) => showToast(error.message, true)));
  $('#messageSearchButton').addEventListener('click', () => loadMessages().catch((error) => showToast(error.message, true)));
  $('#messageSearch').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') loadMessages().catch((error) => showToast(error.message, true));
  });
  $('#refreshUsers').addEventListener('click', () => loadUsers().catch((error) => showToast(error.message, true)));
  $('#refreshGlobal').addEventListener('click', () => loadGlobalChats().catch((error) => showToast(error.message, true)));
  $('#addGlobalFact').addEventListener('click', () => addGlobalFact().catch((error) => showToast(error.message, true)));
}

async function boot() {
  applyI18n();
  bindEvents();
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  if (token) {
    $('#tokenInput').value = token;
    window.history.replaceState({}, document.title, window.location.pathname);
    await login();
    return;
  }
  try {
    const result = await api('/api/me');
    if (result.authenticated) {
      showApp(result.admin);
    } else {
      showLogin();
    }
  } catch (_error) {
    showLogin();
  }
}

boot();
