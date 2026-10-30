/**
 * MediAssist API Client Module
 * Communicates with the Flask backend. Keeps all credentials and AI calls server-side.
 */

/**
 * Send a user message to the Flask chat endpoint.
 * @param {string} message - Sanitized text message from the user
 * @returns {Promise<string>} The assistant's text response
 */
export async function sendChatMessage(message, conversationId = null) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = {
    'Content-Type': 'application/json',
  };
  if (csrfToken) {
    headers['X-CSRFToken'] = csrfToken;
  }

  const payload = { message };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }

  const response = await fetch('/chat', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(payload),
  });

  if (response.status === 401) {
    window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
    throw new Error('Authentication required.');
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const errorMsg = data.error || `Server responded with status ${response.status}`;
    const error = new Error(errorMsg);
    error.status = response.status;
    throw error;
  }

  if (!data.response) {
    throw new Error('Received an empty response from MediAssist.');
  }

  return {
    text: data.response,
    sources: Array.isArray(data.sources) ? data.sources : [],
    triage: data.triage || {},
    evidence: data.evidence || data.triage?.evidence || {},
    citations: data.citations || data.triage?.citation_validation || {},
    conversation_id: data.conversation_id,
    created_at: data.created_at || data.timestamp,
  };
}

/**
 * Stream a consultation response incrementally from the Flask SSE endpoint.
 * @param {string} message - User query
 * @param {string|null} conversationId - Active conversation UUID
 * @param {object} extraPayload - Optional MCQ selection or question ID
 * @returns {AsyncGenerator<{event: string, data: object}>} Structured SSE events
 */
export async function* streamChatMessage(message, conversationId = null, extraPayload = {}) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
  };
  if (csrfToken) {
    headers['X-CSRFToken'] = csrfToken;
  }

  const payload = { message, stream: true };
  if (conversationId) payload.conversation_id = conversationId;
  if (extraPayload.selected_option) payload.selected_option = extraPayload.selected_option;
  if (extraPayload.question_id) payload.question_id = extraPayload.question_id;

  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(payload),
  });

  if (response.status === 401) {
    window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
    throw new Error('Authentication required.');
  }

  if (!response.ok || !response.body) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Server responded with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let currentEvent = 'chunk';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) {
        currentEvent = 'chunk';
        continue;
      }
      if (line.startsWith('event:')) {
        currentEvent = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        try {
          const data = JSON.parse(line.slice(5).trim());
          yield { event: currentEvent, data };
        } catch (_) {}
      }
    }
  }
}


/**
 * Reset conversation history on the Flask backend.
 * @returns {Promise<boolean>}
 */
export async function resetConversation() {
  try {
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    const headers = {
      'Content-Type': 'application/json',
    };
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }

    const response = await fetch('/reset', {
      method: 'POST',
      headers: headers,
    });
    return response.ok;
  } catch (err) {
    console.warn('Unable to reset conversation on server:', err);
    return false;
  }
}

/**
 * Fetch consultation history for current user.
 */
export async function fetchConversations() {
  const response = await fetch('/api/conversations');
  if (response.status === 401) return [];
  const data = await response.json().catch(() => ({}));
  return Array.isArray(data.conversations) ? data.conversations : [];
}

/**
 * Fetch full messages for a specific consultation thread.
 */
export async function fetchConversationDetails(convId) {
  const response = await fetch(`/api/conversations/${convId}`);
  if (!response.ok) return null;
  const data = await response.json().catch(() => ({}));
  return data.conversation || null;
}

/**
 * Delete a consultation thread.
 */
export async function deleteConversationApi(convId) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = {};
  if (csrfToken) headers['X-CSRFToken'] = csrfToken;
  const response = await fetch(`/api/conversations/${convId}`, {
    method: 'DELETE',
    headers,
  });
  return response.ok;
}

/**
 * Rename a consultation thread.
 */
export async function updateConversationTitle(convId, newTitle) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = { 'Content-Type': 'application/json' };
  if (csrfToken) headers['X-CSRFToken'] = csrfToken;
  const response = await fetch(`/api/conversations/${convId}`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify({ title: newTitle }),
  });
  return response.ok;
}

/**
 * Fetch attached documents for a specific conversation.
 * @param {string} convId
 * @returns {Promise<Array>}
 */
export async function fetchDocuments(convId) {
  if (!convId) return [];
  const response = await fetch(`/api/conversations/${convId}/documents`);
  if (!response.ok) return [];
  const data = await response.json().catch(() => ({}));
  return Array.isArray(data.documents) ? data.documents : [];
}

/**
 * Upload a document attachment to a conversation.
 * @param {string} convId
 * @param {File} file
 * @param {boolean} isAsync
 * @returns {Promise<object>}
 */
export async function uploadDocument(convId, file, isAsync = true) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const formData = new FormData();
  formData.append('file', file);

  const headers = {};
  if (csrfToken) headers['X-CSRFToken'] = csrfToken;

  const url = `/api/conversations/${convId}/upload${isAsync ? '?async=1' : ''}`;
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || 'Failed to upload document.');
  }
  return data;
}

/**
 * Fetch async ingestion status for an uploaded document.
 * @param {string} convId
 * @param {string} docId
 * @returns {Promise<object>}
 */
export async function fetchDocumentStatus(convId, docId) {
  const response = await fetch(`/api/conversations/${convId}/documents/${docId}/status`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || 'Failed to fetch document status.');
  }
  return data;
}

/**
 * Delete an attached document from a conversation.
 * @param {string} convId
 * @param {string} docId
 * @returns {Promise<boolean>}
 */
export async function deleteDocumentApi(convId, docId) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = {};
  if (csrfToken) headers['X-CSRFToken'] = csrfToken;

  const response = await fetch(`/api/conversations/${convId}/documents/${docId}`, {
    method: 'DELETE',
    headers,
  });
  return response.ok;
}

/**
 * Update user explanation profile perspective.
 * @param {string} academicYear
 * @param {string} targetExam
 * @returns {Promise<boolean>}
 */
export async function updateUserProfile(academicYear, targetExam) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  const headers = { 'Content-Type': 'application/json' };
  if (csrfToken) headers['X-CSRFToken'] = csrfToken;

  const response = await fetch('/api/profile', {
    method: 'PATCH',
    headers,
    body: JSON.stringify({
      academic_year: academicYear,
      target_exam: targetExam,
    }),
  });
  return response.ok;
}

