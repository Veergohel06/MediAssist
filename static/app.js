/**
 * MediAssist – app.js
 * Handles all chat UI interactions: sending messages, rendering bubbles,
 * showing a loading indicator, auto-growing the textarea, and basic markdown.
 */

/* ── DOM references ─────────────────────────────────────────────── */
const messagesArea = document.getElementById('messagesArea');
const messageInput  = document.getElementById('messageInput');
const sendBtn       = document.getElementById('sendBtn');

/* ── State ───────────────────────────────────────────────────────── */
let isBotTyping = false; // Prevent double-sends while bot is responding

/* ═══════════════════════════════════════════════════════════════════
   UTILITY HELPERS
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Returns the current local time formatted as "HH:MM AM/PM".
 */
function getTimestamp() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Very lightweight markdown renderer for bot messages.
 * Supports: **bold**, _italic_, and line-breaks.
 * IMPORTANT: User input is never passed through this – only server responses.
 */
function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

/**
 * Safely encode user-typed text so it cannot inject HTML.
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

/**
 * Scroll the messages area smoothly to the bottom.
 */
function scrollToBottom() {
  messagesArea.scrollTo({ top: messagesArea.scrollHeight, behavior: 'smooth' });
}

/* ═══════════════════════════════════════════════════════════════════
   MESSAGE RENDERING
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Append a message bubble to the chat.
 *
 * @param {string} text    - The message text (may contain markdown for bot).
 * @param {'user'|'bot'} sender
 * @param {boolean} isError - Whether to render the bubble as an error state.
 * @returns {HTMLElement}  - The bubble element (useful for updates).
 */
function appendMessage(text, sender, isError = false) {
  const isBot = sender === 'bot';

  const messageEl = document.createElement('div');
  messageEl.classList.add('message', isBot ? 'message--bot' : 'message--user');
  messageEl.setAttribute('role', 'article');
  messageEl.setAttribute('aria-label', `${isBot ? 'MediAssist' : 'You'}: ${text}`);

  const avatarEl = document.createElement('div');
  avatarEl.className = 'message__avatar';
  avatarEl.setAttribute('aria-hidden', 'true');
  avatarEl.textContent = isBot ? '🩺' : '👤';

  const bodyEl = document.createElement('div');
  bodyEl.className = 'message__body';

  const nameEl = document.createElement('span');
  nameEl.className = 'message__name';
  nameEl.textContent = isBot ? 'MediAssist' : 'You';

  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'message__bubble';
  if (isError) bubbleEl.classList.add('message__bubble--error');

  // Bot messages may contain basic markdown; user messages are always escaped
  bubbleEl.innerHTML = isBot ? renderMarkdown(text) : escapeHtml(text);

  const timeEl = document.createElement('span');
  timeEl.className = 'message__time';
  timeEl.setAttribute('aria-label', `Sent at ${getTimestamp()}`);
  timeEl.textContent = getTimestamp();

  bodyEl.appendChild(nameEl);
  bodyEl.appendChild(bubbleEl);
  bodyEl.appendChild(timeEl);

  messageEl.appendChild(avatarEl);
  messageEl.appendChild(bodyEl);

  messagesArea.appendChild(messageEl);
  scrollToBottom();

  return bubbleEl; // Return bubble so caller can update it if needed
}

/* ── Loading indicator ─────────────────────────────────────────── */

let loadingEl = null;

/**
 * Show the three-dot typing indicator in the chat.
 */
function showLoadingIndicator() {
  loadingEl = document.createElement('div');
  loadingEl.className = 'loading-indicator';
  loadingEl.setAttribute('aria-live', 'polite');
  loadingEl.setAttribute('aria-label', 'MediAssist is typing');

  loadingEl.innerHTML = `
    <div class="loading-indicator__avatar" aria-hidden="true">🩺</div>
    <div class="loading-indicator__dots" role="status">
      <span></span><span></span><span></span>
    </div>`;

  messagesArea.appendChild(loadingEl);
  scrollToBottom();
}

/**
 * Remove the typing indicator from the chat.
 */
function removeLoadingIndicator() {
  if (loadingEl) {
    loadingEl.remove();
    loadingEl = null;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   INPUT HELPERS
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Auto-grow the textarea as the user types, up to the CSS max-height.
 */
function autoGrowTextarea() {
  messageInput.style.height = 'auto';
  messageInput.style.height = `${messageInput.scrollHeight}px`;
}

/**
 * Lock / unlock the input and send button during a bot response.
 */
function setInputLocked(locked) {
  messageInput.disabled = locked;
  sendBtn.disabled = locked;
  isBotTyping = locked;
  if (!locked) messageInput.focus();
}

/* ═══════════════════════════════════════════════════════════════════
   CORE SEND LOGIC
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Send the user's message to the Flask backend and display the response.
 */
async function sendMessage() {
  const rawText = messageInput.value.trim();

  // ── Guard: empty message ───────────────────────────────────────
  if (!rawText) {
    messageInput.focus();
    return;
  }

  // ── Guard: bot still typing ────────────────────────────────────
  if (isBotTyping) return;

  // ── 1. Display user message ────────────────────────────────────
  appendMessage(rawText, 'user');

  // ── 2. Clear & reset input ─────────────────────────────────────
  messageInput.value = '';
  messageInput.style.height = 'auto';

  // ── 3. Lock UI & show loading ──────────────────────────────────
  setInputLocked(true);
  showLoadingIndicator();

  try {
    // ── 4. POST to /chat with CSRF Token ─────────────────────────
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    const headers = { 'Content-Type': 'application/json' };
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }

    const response = await fetch('/chat', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ message: rawText }),
    });

    removeLoadingIndicator();

    // ── 5. Parse & display bot reply ─────────────────────────────
    if (response.ok) {
      const data = await response.json();
      appendMessage(data.response, 'bot');
    } else if (response.status === 401) {
      // Unauthenticated session expired
      window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
      return;
    } else {
      // Server returned a 4xx / 5xx with a JSON error body
      let errMsg = 'Something went wrong. Please try again.';
      try {
        const errData = await response.json();
        if (errData.error) errMsg = errData.error;
      } catch (_) { /* ignore JSON parse failure */ }
      appendMessage(`⚠️ ${errMsg}`, 'bot', true);
    }
  } catch (networkError) {
    // Network failure (server down, no internet, etc.)
    removeLoadingIndicator();
    appendMessage(
      '⚠️ Unable to reach MediAssist. Please check your connection and try again.',
      'bot',
      true
    );
    console.error('Network error:', networkError);
  } finally {
    setInputLocked(false);
  }
}

/* ═══════════════════════════════════════════════════════════════════
   EVENT LISTENERS
   ═══════════════════════════════════════════════════════════════════ */

// Send on button click
sendBtn.addEventListener('click', sendMessage);

// Send on Enter (Shift+Enter inserts a new line)
messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-grow textarea as user types
messageInput.addEventListener('input', autoGrowTextarea);

// New Chat / Reset conversation handler
const newChatBtn = document.getElementById('newChatBtn');
if (newChatBtn) {
  newChatBtn.addEventListener('click', async () => {
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    const headers = { 'Content-Type': 'application/json' };
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }
    try {
      await fetch('/reset', { method: 'POST', headers });
    } catch (_) { /* ignore */ }
    messagesArea.innerHTML = '';
    init();
  });
}

/* ═══════════════════════════════════════════════════════════════════
   INITIALISATION – Welcome message
   ═══════════════════════════════════════════════════════════════════ */

function init() {
  const welcomeText = [
    '👋 Hello! I\'m **MediAssist**, your virtual health assistant.',
    '',
    'I can provide general health information on topics like:',
    '• Headaches, fever, cough, or stomach issues',
    '• Sleep, stress, and mental well-being',
    '• Nutrition, hydration, and exercise',
    '',
    'Just type your question below and I\'ll do my best to help!',
    '',
    '_Remember: I\'m not a substitute for professional medical advice. Always consult a doctor for personal health concerns._',
  ].join('\n');

  // Small delay so the page render is complete before the message appears
  setTimeout(() => appendMessage(welcomeText, 'bot'), 300);

  messageInput.focus();
}

init();
