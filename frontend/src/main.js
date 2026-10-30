/**
 * MediAssist – Modern Consultation Chat Controller
 * ============================================================================
 * Powers the modern, clinical-grade consultation interface with real-time RAG
 * evidence citations, emergency red-flag triage cards, suggestion prompts,
 * character counting, and CSRF-protected communication with the Flask backend.
 */

(function () {
  'use strict';

  // ── DOM Element References ──────────────────────────────────────────────────
  const welcomeScreen = document.getElementById('welcomeScreen');
  const messagesContainer = document.getElementById('messagesContainer');
  const chatViewport = document.getElementById('chatViewport');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');
  const charCounter = document.getElementById('charCounter');
  const newChatBtn = document.getElementById('newChatBtn');
  const conversationsList = document.getElementById('conversationsList');
  const historyCountBadge = document.getElementById('historyCountBadge');

  // Document attachment and upload elements
  const composerCard = document.getElementById('composerCard');
  const attachmentBar = document.getElementById('attachmentBar');
  const attachmentChipsList = document.getElementById('attachmentChipsList');
  const uploadProgressBanner = document.getElementById('uploadProgressBanner');
  const uploadProgressText = document.getElementById('uploadProgressText');
  const fileAttachmentInput = document.getElementById('fileAttachmentInput');
  const attachBtn = document.getElementById('attachBtn');

  // Theme toggle button
  const themeToggleBtn = document.getElementById('themeToggleBtn');

  // Academic profile switcher elements
  const profileModalToggleBtn = document.getElementById('profileModalToggleBtn');
  const profilePopover = document.getElementById('profilePopover');
  const profileForm = document.getElementById('profileForm');
  const selectAcademicYear = document.getElementById('selectAcademicYear');
  const selectTargetExam = document.getElementById('selectTargetExam');
  const headerProfileLabel = document.getElementById('headerProfileLabel');
  const saveProfileBtn = document.getElementById('saveProfileBtn');
  const profileSaveStatus = document.getElementById('profileSaveStatus');

  // Mobile sidebar drawer elements
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');
  const openSidebarBtn = document.getElementById('openSidebarBtn');
  const closeSidebarBtn = document.getElementById('closeSidebarBtn');

  // Suggestion prompt cards
  const suggestionCards = document.querySelectorAll('.prompt-card');

  // Evidence Drawer elements
  const evidenceDrawer = document.getElementById('evidenceDrawer');
  const evidenceDrawerBackdrop = document.getElementById('evidenceDrawerBackdrop');
  const closeEvidenceDrawerBtn = document.getElementById('closeEvidenceDrawerBtn');
  const evidenceDrawerBody = document.getElementById('evidenceDrawerBody');
  const evidenceDrawerTitle = document.getElementById('evidenceDrawerTitle');
  const evidenceDrawerKicker = document.getElementById('evidenceDrawerKicker');
  let lastFocusedCitationTrigger = null;
  let activeEvidenceSources = [];

  // State
  let isGenerating = false;
  let lastUserMessage = '';
  let currentConversationId = null;
  const MAX_LENGTH = 2000;


  // ── Utility: CSRF Token ─────────────────────────────────────────────────────
  function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
  }

  // ── Utility: UTC Date Parser ────────────────────────────────────────────────
  function parseUtcDate(isoString) {
    if (!isoString) return null;
    let s = String(isoString).trim();
    if (!s) return null;
    if (!s.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(s)) {
      s += 'Z';
    }
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }

  // ── Utility: Localized Formatted Time ───────────────────────────────────────
  function getFormattedTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function formatMessageTime(isoString) {
    if (!isoString) return getFormattedTime();
    const d = parseUtcDate(isoString);
    if (!d) return getFormattedTime();
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function formatRelativeTime(isoString) {
    if (!isoString) return '';
    const d = parseUtcDate(isoString);
    if (!d) return '';
    const now = new Date();
    const diffSec = Math.floor((now.getTime() - d.getTime()) / 1000);
    if (diffSec < 60) return 'Just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHours = Math.floor(diffMin / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  // ── Utility: Session Expiry Banner ──────────────────────────────────────────
  function handleSessionExpired(customMsg) {
    let banner = document.getElementById('sessionExpiryBanner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'sessionExpiryBanner';
      banner.className = 'session-expiry-banner';
      banner.setAttribute('role', 'alert');
      document.body.appendChild(banner);
    }
    const msg = customMsg || 'Your session has expired. Please sign in again to continue.';
    banner.innerHTML = `
      <span>⚠️ ${escapeHtml(msg)}</span>
      <a href="/login?next=${encodeURIComponent(window.location.pathname)}" class="session-signin-link">Sign In</a>
    `;
  }

  // ── Utility: HTML Escaping ──────────────────────────────────────────────────
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ── Functional Theme Switcher (Light / Dark) ────────────────────────────────
  function initTheme() {
    function getPreferredTheme() {
      try {
        const saved = localStorage.getItem('mediassist_theme');
        if (saved === 'dark' || saved === 'light') return saved;
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
          return 'dark';
        }
      } catch (_) { }
      return 'light';
    }

    function applyTheme(theme, persist = false) {
      document.documentElement.setAttribute('data-theme', theme);
      if (persist) {
        try {
          localStorage.setItem('mediassist_theme', theme);
        } catch (_) { }
      }
      if (themeToggleBtn) {
        const isDark = theme === 'dark';
        themeToggleBtn.setAttribute('aria-pressed', isDark ? 'true' : 'false');
        themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
        themeToggleBtn.setAttribute('title', isDark ? 'Switch to light theme' : 'Switch to dark theme');
      }
    }

    // Apply active theme immediately
    const currentTheme = getPreferredTheme();
    applyTheme(currentTheme, false);

    // Toggle button click listener
    themeToggleBtn?.addEventListener('click', () => {
      const activeTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const newTheme = activeTheme === 'dark' ? 'light' : 'dark';
      applyTheme(newTheme, true);
    });

    // Listen for OS theme changes when no manual selection has been stored
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        let hasSaved = null;
        try {
          hasSaved = localStorage.getItem('mediassist_theme');
        } catch (_) { }
        if (!hasSaved) {
          applyTheme(e.matches ? 'dark' : 'light', false);
        }
      });
    } catch (_) { }
  }

  // ── Interactive Evidence Drawer Controller ─────────────────────────────────
  function openEvidenceDrawer(source, index = 0) {
    if (!evidenceDrawer || !source) return;
    lastFocusedCitationTrigger = document.activeElement;

    let displayTitle = (source.title || 'Medical Reference').trim();
    if (displayTitle.endsWith('.md') || displayTitle.endsWith('.txt')) {
      displayTitle = displayTitle.replace(/\.(md|txt)$/i, '').replace(/_/g, ' ');
    }
    if (displayTitle.indexOf(' ') === -1 && displayTitle.indexOf('_') !== -1) {
      displayTitle = displayTitle.replace(/_/g, ' ');
    }

    if (evidenceDrawerKicker) {
      evidenceDrawerKicker.textContent = `Medical Evidence · Citation [${index + 1}]`;
    }
    if (evidenceDrawerTitle) {
      evidenceDrawerTitle.textContent = displayTitle;
    }

    const safeUrl = (source.url && (source.url.startsWith('https://') || source.url.startsWith('http://'))) ? source.url : '';
    const cleanSection = (source.section || '').replace(/^#{1,6}\s*/, '').trim();
    const pageNum = source.page_number || source.page || '';
    const publisher = (source.source || source.author_organization || '').trim();
    const retrievalDate = (source.retrieval_date || '').trim();
    const confidenceScore = source.score != null ? (typeof source.score === 'number' ? (source.score > 1 ? source.score.toFixed(2) : (source.score * 100).toFixed(1) + '%') : String(source.score)) : '';
    const pmid = (source.pmid || source.pubmed_id || '').trim();
    const excerptText = (source.excerpt || source.text || source.snippet || '').trim();

    if (evidenceDrawerBody) {
      evidenceDrawerBody.innerHTML = `
        <div class="evidence-card" role="region" aria-label="Evidence metadata and excerpt">
          <h4 class="evidence-source-title">${escapeHtml(displayTitle)}</h4>

          <div class="evidence-metadata-grid">
            <div class="evidence-meta-pill">
              <span class="meta-label">Curriculum Section</span>
              <span class="meta-value">${cleanSection ? escapeHtml(cleanSection) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
            <div class="evidence-meta-pill">
              <span class="meta-label">Publisher / Authority</span>
              <span class="meta-value">${publisher ? escapeHtml(publisher) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
            <div class="evidence-meta-pill">
              <span class="meta-label">Page Number</span>
              <span class="meta-value">${pageNum ? 'Page ' + escapeHtml(String(pageNum)) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
            <div class="evidence-meta-pill">
              <span class="meta-label">Verification Date</span>
              <span class="meta-value">${retrievalDate ? escapeHtml(retrievalDate) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
            <div class="evidence-meta-pill">
              <span class="meta-label">Retrieval Score</span>
              <span class="meta-value">${confidenceScore ? escapeHtml(confidenceScore) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
            <div class="evidence-meta-pill">
              <span class="meta-label">PubMed ID</span>
              <span class="meta-value">${pmid ? escapeHtml(pmid) : '<em class="unavailable-label">Unavailable</em>'}</span>
            </div>
          </div>

          <div class="evidence-excerpt-section">
            <h5 class="evidence-section-label">Retrieved Verbatim Excerpt</h5>
            <div class="evidence-excerpt-box">
              ${excerptText ? formatSafeMarkdown(excerptText) : '<p class="unavailable-text">Verbatim excerpt unavailable for this reference.</p>'}
            </div>
          </div>

          <div class="evidence-link-section">
            ${safeUrl 
              ? `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer" class="evidence-external-link">
                   <span>Open Verified Primary Source</span>
                   <span aria-hidden="true">↗</span>
                 </a>`
              : `<span class="evidence-link-disabled">External Link: <em class="unavailable-label">Unavailable (Curated Internal Learning Source)</em></span>`
            }
          </div>
        </div>
      `;
    }

    evidenceDrawer.setAttribute('aria-hidden', 'false');
    evidenceDrawer.classList.add('active');
    evidenceDrawerBackdrop?.classList.add('active');
    evidenceDrawerBackdrop?.setAttribute('aria-hidden', 'false');
    closeEvidenceDrawerBtn?.focus();
  }

  function closeEvidenceDrawer() {
    if (!evidenceDrawer) return;
    evidenceDrawer.setAttribute('aria-hidden', 'true');
    evidenceDrawer.classList.remove('active');
    evidenceDrawerBackdrop?.classList.remove('active');
    evidenceDrawerBackdrop?.setAttribute('aria-hidden', 'true');
    if (lastFocusedCitationTrigger && typeof lastFocusedCitationTrigger.focus === 'function') {
      try {
        lastFocusedCitationTrigger.focus();
      } catch (_) {}
    }
  }

  closeEvidenceDrawerBtn?.addEventListener('click', closeEvidenceDrawer);
  evidenceDrawerBackdrop?.addEventListener('click', closeEvidenceDrawer);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && evidenceDrawer && evidenceDrawer.classList.contains('active')) {
      closeEvidenceDrawer();
    }
  });

  // Global click delegator for citation badges and inline source links
  document.addEventListener('click', (e) => {
    const badge = e.target.closest('.citation-badge');
    if (badge) {
      e.preventDefault();
      const numStr = badge.dataset.citationNum;
      const idxStr = badge.dataset.citationIdx;
      let targetIdx = 0;
      if (numStr) {
        targetIdx = parseInt(numStr, 10) - 1;
      } else if (idxStr) {
        targetIdx = parseInt(idxStr, 10);
      }
      if (activeEvidenceSources && activeEvidenceSources[targetIdx]) {
        openEvidenceDrawer(activeEvidenceSources[targetIdx], targetIdx);
      } else if (activeEvidenceSources && activeEvidenceSources.length > 0) {
        openEvidenceDrawer(activeEvidenceSources[0], 0);
      }
      return;
    }

    // Also allow inline source links without an external URL or matching verified sources to open drawer
    const inlineLink = e.target.closest('.inline-source-link');
    if (inlineLink && activeEvidenceSources && activeEvidenceSources.length > 0) {
      const linkText = inlineLink.textContent.replace('↗', '').trim().toLowerCase();
      const matched = activeEvidenceSources.find(s => {
        const title = (s.title || '').toLowerCase();
        return title.includes(linkText) || linkText.includes(title);
      });
      if (matched) {
        // If it's an internal source or user clicked the citation text
        if (!matched.url || !matched.url.startsWith('http')) {
          e.preventDefault();
          openEvidenceDrawer(matched, activeEvidenceSources.indexOf(matched));
        }
      }
    }
  });


  // ── Utility: Unwrap AI Raw Responses and JSON Artifacts ────────────────────
  function unwrapMedicalAnswer(rawText) {
    if (!rawText) return '';
    let text = String(rawText).trim();

    // 0. If response contains literal unescaped \n and no real newlines, convert them
    if (!text.includes('\n') && text.includes('\\n')) {
      text = text.replace(/\\n/g, '\n');
    }

    // 1. Detect and unwrap JSON answer wrapper (complete, partial, or streaming code block)
    const hasJsonFence = /^```(?:json)?\s*\{/i.test(text);
    const hasRawJson = /^\{\s*"(?:answer|response)"/i.test(text);

    if (hasJsonFence || hasRawJson) {
      const cleanBlock = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
      try {
        const parsed = JSON.parse(cleanBlock);
        if (parsed.answer) text = parsed.answer;
        else if (parsed.response) text = parsed.response;
      } catch (_) {
        // Handle unclosed or streaming JSON answers
        const answerStart = text.match(/"(?:answer|response)"\s*:\s*"/i);
        if (answerStart) {
          const startIdx = answerStart.index + answerStart[0].length;
          let extracted = text.slice(startIdx);
          extracted = extracted.replace(/(?:\\")?"\s*\}?\s*(?:```)?$/i, '');
          extracted = extracted.replace(/```$/i, '');
          extracted = extracted
            .replace(/\\n/g, '\n')
            .replace(/\\"/g, '"')
            .replace(/\\\\/g, '\\')
            .replace(/\\t/g, '\t')
            .replace(/\\r/g, '');
          text = extracted.trim();
        }
      }
    }

    // 2. Fallback: match inline answer attribute in JSON structure
    const inlineMatch = text.match(/"(?:answer|response)"\s*:\s*"((?:\\.|[^"\\])*)"/);
    if (inlineMatch && text.startsWith('{')) {
      text = inlineMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    }

    // 3. Detect if wrapped in a single enclosing markdown code fence
    const outerFenceMatch = text.match(/^```(?:markdown|md|text)?\s*([\s\S]+?)\s*```$/i);
    if (outerFenceMatch && !outerFenceMatch[1].includes('```')) {
      text = outerFenceMatch[1];
    }

    // 4. Sanitation: remove bracketed internal citation identifiers or chunk codes
    text = text.replace(/\[\s*(?:[a-zA-Z0-9_-]+:p\d+|[a-zA-Z0-9_-]+_\d+)\s*\]/gi, '');
    text = text.replace(/\b(?:chunk_\d+|doc_\d+|source_[a-zA-Z0-9_-]+_\d+)\b/gi, '');
    text = text.replace(/\s+([.,;:!?])/g, '$1');

    return text;
  }

  // ── Inline Markdown Renderer (Bold, Italic, Math, Symbols, Links, Codes) ─────
  function formatInlineMarkdown(text) {
    if (!text) return '';

    // First escape HTML entities for safety
    let formatted = escapeHtml(text);

    // 1. Math expressions and equations ($$display$$ and $inline$)
    formatted = formatted.replace(/\$\$([\s\S]+?)\$\$/g, '<span class="math-display" role="math">$1</span>');
    formatted = formatted.replace(/\$([^\$\n]+?)\$/g, '<span class="math-inline" role="math">$1</span>');

    // 2. Bold and Italic combinations
    formatted = formatted.replace(/\*\*\*([^*\n]+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    formatted = formatted.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/(?<!\w)\*([^*\n]+?)\*(?!\w)/g, '<em>$1</em>');
    formatted = formatted.replace(/(?<!\w)_([^_\n]+?)_(?!\w)/g, '<em>$1</em>');
    formatted = formatted.replace(/~~([^~\n]+?)~~/g, '<del>$1</del>');

    // 3. Inline Code: `code`
    formatted = formatted.replace(/`([^`\n]+?)`/g, '<code class="inline-code">$1</code>');

    // 4. Medical / Scientific notations, Greek letters & mathematical arrows
    formatted = formatted.replace(/\\(?:alpha|Alpha)\b/g, '<span class="sym">α</span>');
    formatted = formatted.replace(/\\(?:beta|Beta)\b/g, '<span class="sym">β</span>');
    formatted = formatted.replace(/\\(?:gamma|Gamma)\b/g, '<span class="sym">γ</span>');
    formatted = formatted.replace(/\\(?:delta)\b/g, '<span class="sym">δ</span>');
    formatted = formatted.replace(/\\(?:Delta)\b/g, '<span class="sym">Δ</span>');
    formatted = formatted.replace(/\\(?:mu)\b/g, '<span class="sym">μ</span>');
    formatted = formatted.replace(/\\(?:lambda|Lambda)\b/g, '<span class="sym">λ</span>');
    formatted = formatted.replace(/\\(?:theta|Theta)\b/g, '<span class="sym">θ</span>');

    // Comparison & arithmetic symbols
    formatted = formatted.replace(/\\(?:pm)\b/g, '<span class="sym">±</span>');
    formatted = formatted.replace(/\\(?:le)\b/g, '<span class="sym">≤</span>');
    formatted = formatted.replace(/\\(?:ge)\b/g, '<span class="sym">≥</span>');
    formatted = formatted.replace(/\\(?:neq)\b/g, '<span class="sym">≠</span>');
    formatted = formatted.replace(/\\(?:approx)\b/g, '<span class="sym">≈</span>');
    formatted = formatted.replace(/\\(?:times)\b/g, '<span class="sym">×</span>');
    formatted = formatted.replace(/\\(?:cdot)\b/g, '<span class="sym">·</span>');

    // Arrows
    formatted = formatted.replace(/(?:\\to|-&gt;)/g, '<span class="sym-arrow">→</span>');
    formatted = formatted.replace(/(?:\\leftarrow|&lt;-)/g, '<span class="sym-arrow">←</span>');
    formatted = formatted.replace(/\\uparrow/g, '<span class="sym-arrow">↑</span>');
    formatted = formatted.replace(/\\downarrow/g, '<span class="sym-arrow">↓</span>');
    formatted = formatted.replace(/&lt;-&gt;/g, '<span class="sym-arrow">↔</span>');

    // Common medical subscripts for arterial blood gases and ECG leads
    formatted = formatted.replace(/\bO_?2\b/g, 'O<sub>2</sub>');
    formatted = formatted.replace(/\bCO_?2\b/g, 'CO<sub>2</sub>');
    formatted = formatted.replace(/\bSpO_?2\b/g, 'SpO<sub>2</sub>');
    formatted = formatted.replace(/\bPaO_?2\b/g, 'PaO<sub>2</sub>');
    formatted = formatted.replace(/\bPaCO_?2\b/g, 'PaCO<sub>2</sub>');
    formatted = formatted.replace(/\bV_?([1-6])(R?)\b/g, 'V<sub>$1</sub>$2');

    // Superscripts and subscripts: ^(text) or ^number
    formatted = formatted.replace(/\^([0-9\+\-]+)/g, '<sup>$1</sup>');
    formatted = formatted.replace(/&lt;([0-9]+%?)/g, '&lt;$1');

    // 5. Responsive medical figures & diagrams: ![alt](url)
    formatted = formatted.replace(
      /!\[([^\]]*)\]\(((?:https?:\/\/|\/|\.\/|data:image\/)[^\s\)\"\']+)\)/g,
      '<figure class="answer-figure"><img src="$2" alt="$1" class="answer-img" loading="lazy" onerror="this.onerror=null;this.parentElement.innerHTML=\'<span class=\\\'figure-fallback-note\\\'>📊 [Clinical Illustration: $1]</span>\';" /><figcaption class="answer-figcaption">$1</figcaption></figure>'
    );

    // 6. Verified markdown links: [Title](https://...)
    formatted = formatted.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s\)\"\']+)\)/g,
      '<a class="inline-source-link" href="$2" target="_blank" rel="noopener noreferrer">$1 <span class="chip-ext-icon">↗</span></a>'
    );

    // 7. Interactive Medical Citation Badges: [1], [2], etc.
    formatted = formatted.replace(
      /(?<![\!\(])\[(\d+)\](?!\()/g,
      '<button type="button" class="citation-badge" data-citation-num="$1" aria-haspopup="dialog" aria-label="Inspect medical evidence [$1]"><span class="citation-badge-num">[$1]</span></button>'
    );

    return formatted;
  }


  // ── Professional Medical Table Renderer ─────────────────────────────────────
  function parseTableBlock(tableLines) {
    if (!tableLines || tableLines.length < 2) return '';
    const headerLine = tableLines[0];
    const sepLine = tableLines[1];
    const bodyLines = tableLines.slice(2);

    const splitRow = (line) => {
      let raw = line.trim();
      if (raw.startsWith('|')) raw = raw.slice(1);
      if (raw.endsWith('|')) raw = raw.slice(0, -1);
      return raw.split('|').map(s => s.trim());
    };

    const headers = splitRow(headerLine);
    const seps = splitRow(sepLine);
    const alignments = seps.map((s) => {
      const left = s.startsWith(':');
      const right = s.endsWith(':');
      if (left && right) return 'center';
      if (right) return 'right';
      return 'left';
    });

    let theadHtml = '';
    headers.forEach((h, idx) => {
      const align = alignments[idx] || 'left';
      theadHtml += `<th style="text-align: ${align};">${formatInlineMarkdown(h)}</th>`;
    });

    let tbodyHtml = '';
    bodyLines.forEach((bLine) => {
      const cells = splitRow(bLine);
      if (cells.length === 0 || (cells.length === 1 && !cells[0])) return;
      let trHtml = '';
      cells.forEach((c, idx) => {
        const align = alignments[idx] || 'left';
        // Classify numerical values or dosages for tabular alignment
        const isNum = /^[\$\s<>=~±+-]*\d+(?:\.\d+)?\s*(?:%|mg|mcg|g|kg|L|mL|mm\s*Hg|mEq\/L|U\/L|°C|°F|min|hours?|days?|weeks?|months?|years?)?[\$\s]*$/i.test(c);
        const cellClass = isNum ? 'cell-num' : '';
        trHtml += `<td class="${cellClass}" style="text-align: ${align};">${formatInlineMarkdown(c)}</td>`;
      });
      tbodyHtml += `<tr>${trHtml}</tr>`;
    });

    return `
      <div class="medical-table-container" tabindex="0" role="region" aria-label="Medical Comparison Table">
        <table class="medical-table">
          <thead><tr>${theadHtml}</tr></thead>
          <tbody>${tbodyHtml}</tbody>
        </table>
      </div>
    `;
  }

  // ── Clinical Callout / Note Renderer ────────────────────────────────────────
  function parseCalloutBlock(calloutLines) {
    if (!calloutLines || calloutLines.length === 0) return '';
    const firstLine = calloutLines[0].trim();

    let type = 'key-point';
    let title = 'Clinical Key Point';
    let icon = '🩺';
    let startIndex = 0;

    // Check for GitHub alert markers: > [!NOTE], > [!WARNING], > [!IMPORTANT], > [!TIP], > [!EMERGENCY]
    const alertMatch = firstLine.match(/^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION|EMERGENCY|PEARL|EXAM)\]\s*(.*)/i);
    if (alertMatch) {
      const tag = alertMatch[1].toUpperCase();
      const rest = alertMatch[2].trim();
      startIndex = 1;
      if (tag === 'WARNING' || tag === 'CAUTION') {
        type = 'contraindication';
        title = rest || 'Contraindications & Precautions';
        icon = '⚠️';
      } else if (tag === 'IMPORTANT' || tag === 'EXAM') {
        type = 'exam-fact';
        title = rest || 'High-Yield Exam Pearl';
        icon = '🎓';
      } else if (tag === 'EMERGENCY') {
        type = 'warning';
        title = rest || 'Emergency Clinical Action';
        icon = '🚨';
      } else if (tag === 'TIP' || tag === 'PEARL') {
        type = 'pearl';
        title = rest || 'Clinical Pearl';
        icon = '💡';
      } else {
        type = 'key-point';
        title = rest || 'Clinical Key Point';
        icon = '🩺';
      }
    } else {
      // Check for bold lead labels: > **Key Clinical Point:** or > **Contraindication:**
      const boldMatch = firstLine.match(/^>\s*\*\*([^*]+)\*\*:?\s*(.*)/);
      if (boldMatch) {
        const boldTitle = boldMatch[1].trim();
        const boldRest = boldMatch[2].trim();
        const lower = boldTitle.toLowerCase();
        if (lower.includes('contraindicat') || lower.includes('warning') || lower.includes('caution')) {
          type = 'contraindication';
          title = boldTitle;
          icon = '⚠️';
        } else if (lower.includes('exam') || lower.includes('viva') || lower.includes('high-yield') || lower.includes('neet')) {
          type = 'exam-fact';
          title = boldTitle;
          icon = '🎓';
        } else if (lower.includes('pearl')) {
          type = 'pearl';
          title = boldTitle;
          icon = '💡';
        } else if (lower.includes('emergency') || lower.includes('red flag') || lower.includes('urgent')) {
          type = 'warning';
          title = boldTitle;
          icon = '🚨';
        } else {
          type = 'key-point';
          title = boldTitle;
          icon = '🩺';
        }
        startIndex = boldRest ? 0 : 1;
        if (boldRest) {
          calloutLines[0] = `> ${boldRest}`;
        }
      }
    }

    const cleanLines = calloutLines.slice(startIndex).map(l => l.replace(/^>\s?/, '').trim()).filter(Boolean);
    const bodyHtml = cleanLines.map(l => `<p>${formatInlineMarkdown(l)}</p>`).join('');
    const roleAttr = (type === 'contraindication' || type === 'warning') ? 'alert' : 'note';

    return `
      <div class="clinical-callout clinical-callout--${type}" role="${roleAttr}">
        <div class="callout-icon" aria-hidden="true">${icon}</div>
        <div class="callout-content">
          <strong class="callout-title">${escapeHtml(title)}</strong>
          <div class="callout-body">${bodyHtml || '<p>' + formatInlineMarkdown(firstLine.replace(/^>\s?/, '')) + '</p>'}</div>
        </div>
      </div>
    `;
  }

  // ── Global Helper: Copy Code Snippets ───────────────────────────────────────
  window.copyMedicalCodeSnippet = function (btn) {
    if (!btn) return;
    const container = btn.closest('.code-block-wrap');
    if (!container) return;
    const codeEl = container.querySelector('code');
    if (!codeEl) return;
    const text = codeEl.innerText || codeEl.textContent || '';
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => {
        btn.textContent = orig;
        btn.classList.remove('copied');
      }, 2000);
    });
  };

  // ── Complete Medical Answer / Markdown Parser ───────────────────────────────
  function formatSafeMarkdown(rawText) {
    if (!rawText) return '';
    const cleanText = unwrapMedicalAnswer(rawText);
    const lines = cleanText.split('\n');

    let html = '';
    let listStack = []; // stores { type: 'ul' | 'ol', indent: number }
    let tableBuffer = [];
    let calloutBuffer = [];
    let codeBuffer = [];
    let inCodeBlock = false;
    let codeLang = '';

    function closeAllLists() {
      while (listStack.length > 0) {
        const item = listStack.pop();
        html += `</${item.type}>`;
      }
    }

    function adjustList(type, indent) {
      if (listStack.length === 0) {
        const listClass = type === 'ol' ? ' class="medical-steps-list"' : '';
        html += `<${type}${listClass}>`;
        listStack.push({ type, indent });
      } else {
        const top = listStack[listStack.length - 1];
        if (indent > top.indent) {
          html += `<${type}>`;
          listStack.push({ type, indent });
        } else if (indent < top.indent) {
          while (listStack.length > 0 && listStack[listStack.length - 1].indent > indent) {
            const popped = listStack.pop();
            html += `</${popped.type}>`;
          }
          if (listStack.length === 0 || listStack[listStack.length - 1].type !== type) {
            html += `<${type}>`;
            listStack.push({ type, indent });
          }
        } else if (top.type !== type) {
          html += `</${top.type}><${type}>`;
          listStack[listStack.length - 1] = { type, indent };
        }
      }
    }

    function flushTable() {
      if (tableBuffer.length > 0) {
        html += parseTableBlock(tableBuffer);
        tableBuffer = [];
      }
    }

    function flushCallout() {
      if (calloutBuffer.length > 0) {
        html += parseCalloutBlock(calloutBuffer);
        calloutBuffer = [];
      }
    }

    for (let i = 0; i < lines.length; i++) {
      const rawLine = lines[i];
      const trimmed = rawLine.trim();

      // 1. Code Block Handling (```lang)
      const codeFenceMatch = trimmed.match(/^```([a-zA-Z0-9_-]*)/);
      if (codeFenceMatch && !inCodeBlock) {
        closeAllLists();
        flushTable();
        flushCallout();
        inCodeBlock = true;
        codeLang = codeFenceMatch[1] || 'plaintext';
        codeBuffer = [];
        continue;
      }
      if (trimmed.startsWith('```') && inCodeBlock) {
        inCodeBlock = false;
        const codeContent = escapeHtml(codeBuffer.join('\n'));
        html += `
          <div class="code-block-wrap">
            <div class="code-block-bar">
              <span class="code-lang-tag">${escapeHtml(codeLang)}</span>
              <button type="button" class="copy-code-btn" onclick="window.copyMedicalCodeSnippet(this)">Copy</button>
            </div>
            <pre><code class="lang-${escapeHtml(codeLang)}">${codeContent}</code></pre>
          </div>
        `;
        codeBuffer = [];
        continue;
      }
      if (inCodeBlock) {
        codeBuffer.push(rawLine);
        continue;
      }

      // 2. Table Block Detection: line containing '|' and bordered or surrounded
      const isTableRow = trimmed.includes('|') && (
        trimmed.startsWith('|') ||
        (i + 1 < lines.length && lines[i + 1].trim().includes('|') && /^[\s|:-]+$/.test(lines[i + 1].trim())) ||
        tableBuffer.length > 0
      );

      if (isTableRow) {
        closeAllLists();
        flushCallout();
        tableBuffer.push(trimmed);
        continue;
      } else {
        flushTable();
      }

      // 3. Clinical Callouts & Blockquotes (> ...)
      if (trimmed.startsWith('>')) {
        closeAllLists();
        calloutBuffer.push(trimmed);
        continue;
      } else {
        flushCallout();
      }

      // 4. Blank lines
      if (!trimmed) {
        closeAllLists();
        continue;
      }

      // 5. Horizontal Dividers (--- or ***)
      if (/^(?:---|\*\*\*|___)$/.test(trimmed)) {
        closeAllLists();
        html += '<hr class="section-divider" />';
        continue;
      }

      // 6. Markdown Headings (# H1, ## H2, ### H3, #### H4)
      const h1Match = trimmed.match(/^#\s+(.*)/);
      if (h1Match) {
        closeAllLists();
        html += `<h1 class="answer-title">${formatInlineMarkdown(h1Match[1])}</h1>`;
        continue;
      }

      const h2Match = trimmed.match(/^##\s+(.*)/);
      if (h2Match) {
        closeAllLists();
        html += `<h2 class="answer-section-heading">${formatInlineMarkdown(h2Match[1])}</h2>`;
        continue;
      }

      const h3Match = trimmed.match(/^###\s+(.*)/);
      if (h3Match) {
        closeAllLists();
        html += `<h3 class="answer-sub-heading">${formatInlineMarkdown(h3Match[1])}</h3>`;
        continue;
      }

      const h4Match = trimmed.match(/^####\s+(.*)/);
      if (h4Match) {
        closeAllLists();
        html += `<h4 class="answer-item-heading">${formatInlineMarkdown(h4Match[1])}</h4>`;
        continue;
      }

      // 7. Unordered Lists (*, -, +)
      const ulMatch = rawLine.match(/^(\s*)(?:[\*\-\+])\s+(.*)/);
      if (ulMatch) {
        const indent = ulMatch[1].length;
        adjustList('ul', indent);
        html += `<li>${formatInlineMarkdown(ulMatch[2])}</li>`;
        continue;
      }

      // 8. Ordered Lists (1., 2.)
      const olMatch = rawLine.match(/^(\s*)(\d+)\.\s+(.*)/);
      if (olMatch) {
        const indent = olMatch[1].length;
        adjustList('ol', indent);
        html += `<li>${formatInlineMarkdown(olMatch[3])}</li>`;
        continue;
      }

      // 9. Regular Paragraphs
      closeAllLists();

      // Special highlight for direct clinical warning paragraphs starting with bold trigger
      if (/^\*\*(?:Contraindication|Caution|Warning|Red Flag|Emergency Alert)s?:\*\*/i.test(trimmed)) {
        html += `
          <div class="clinical-callout clinical-callout--contraindication" role="alert">
            <div class="callout-icon" aria-hidden="true">⚠️</div>
            <div class="callout-content">
              <div class="callout-body"><p>${formatInlineMarkdown(trimmed)}</p></div>
            </div>
          </div>
        `;
      } else {
        html += `<p>${formatInlineMarkdown(trimmed)}</p>`;
      }
    }

    // Flush any pending blocks
    closeAllLists();
    flushTable();
    flushCallout();

    if (inCodeBlock && codeBuffer.length > 0) {
      const codeContent = escapeHtml(codeBuffer.join('\n'));
      html += `
        <div class="code-block-wrap">
          <pre><code class="lang-${escapeHtml(codeLang)}">${codeContent}</code></pre>
        </div>
      `;
    }

    return html;
  }

  // ── Clean Text Extractor for Clipboard & Speech ─────────────────────────────
  function extractCleanTextForCopy(element) {
    if (!element) return '';
    const clone = element.cloneNode(true);
    // Remove UI interactive cards and buttons from copy payload
    clone.querySelectorAll('.interactive-mcq-card, .mcq-card-footer, .mcq-options-list, .code-block-bar, .answer-actions-toolbar, .suggested-followups-wrap').forEach(el => el.remove());
    return (clone.innerText || clone.textContent || '').trim();
  }

  // ── Speech Synthesis: Read Aloud Controller ─────────────────────────────────
  function setupReadAloud(speakBtn, bubble) {
    if (!('speechSynthesis' in window)) {
      speakBtn.style.display = 'none';
      return;
    }

    let isSpeaking = false;

    function stopSpeaking() {
      window.speechSynthesis.cancel();
      isSpeaking = false;
      speakBtn.classList.remove('active');
      speakBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read Aloud</span>
      `;
      speakBtn.setAttribute('aria-label', 'Read answer aloud');
    }

    speakBtn.addEventListener('click', () => {
      if (isSpeaking) {
        stopSpeaking();
        return;
      }

      window.speechSynthesis.cancel();
      const cleanText = extractCleanTextForCopy(bubble);
      if (!cleanText) return;

      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;

      utterance.onstart = () => {
        isSpeaking = true;
        speakBtn.classList.add('active');
        speakBtn.innerHTML = `
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="6" y="4" width="4" height="16"></rect>
            <rect x="14" y="4" width="4" height="16"></rect>
          </svg>
          <span>Stop</span>
        `;
        speakBtn.setAttribute('aria-label', 'Stop reading aloud');
      };

      utterance.onend = stopSpeaking;
      utterance.onerror = stopSpeaking;

      window.speechSynthesis.speak(utterance);
    });
  }

  // ── Copy Answer Setup ───────────────────────────────────────────────────────
  function setupCopyAnswer(copyBtn, bubble) {
    copyBtn.addEventListener('click', async () => {
      const textToCopy = extractCleanTextForCopy(bubble);
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(textToCopy);
        } else {
          const temp = document.createElement('textarea');
          temp.value = textToCopy;
          document.body.appendChild(temp);
          temp.select();
          document.execCommand('copy');
          temp.remove();
        }

        copyBtn.classList.add('copied');
        const origHtml = copyBtn.innerHTML;
        copyBtn.innerHTML = `
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span>Copied!</span>
        `;
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          copyBtn.innerHTML = origHtml;
        }, 2000);
      } catch (err) {
        console.warn('Copy action failed:', err);
      }
    });
  }

  // ── Feedback Group Setup (Helpful / Not Helpful) ────────────────────────────
  function setupFeedbackGroup(helpfulBtn, notHelpfulBtn) {
    helpfulBtn.addEventListener('click', () => {
      const wasActive = helpfulBtn.classList.contains('active');
      helpfulBtn.classList.toggle('active', !wasActive);
      helpfulBtn.setAttribute('aria-pressed', String(!wasActive));
      notHelpfulBtn.classList.remove('active');
      notHelpfulBtn.setAttribute('aria-pressed', 'false');
    });

    notHelpfulBtn.addEventListener('click', () => {
      const wasActive = notHelpfulBtn.classList.contains('active');
      notHelpfulBtn.classList.toggle('active', !wasActive);
      notHelpfulBtn.setAttribute('aria-pressed', String(!wasActive));
      helpfulBtn.classList.remove('active');
      helpfulBtn.setAttribute('aria-pressed', 'false');
    });
  }

  // ── Regenerate Answer Setup ─────────────────────────────────────────────────
  function setupRegenerate(regenBtn, userPrompt) {
    regenBtn.addEventListener('click', () => {
      if (isGenerating) return;
      const prompt = userPrompt || lastUserMessage;
      if (prompt) {
        submitMessage(prompt);
      }
    });
  }

  // ── Contextual Follow-up Suggestions Generator ──────────────────────────────
  function generateContextualFollowups(text, userPrompt) {
    const followups = [];
    const lowerText = (text || '').toLowerCase();
    const lowerPrompt = (userPrompt || '').toLowerCase();

    // 1. Check if model suggested questions in the response
    const lines = (text || '').split('\n');
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      const qMatch = trimmed.match(/^(?:[1-4]\.|\*|-)\s+(.*?\?)$/);
      if (qMatch) {
        const q = qMatch[1].trim();
        if (q.length > 15 && q.length < 120 && !followups.includes(q)) {
          followups.push(q);
          if (followups.length >= 3) break;
        }
      }
    }

    // 2. High-yield academic follow-ups based on medical topic
    if (followups.length < 3) {
      if (lowerText.includes('stemi') || lowerText.includes('myocardial infarction') || lowerPrompt.includes('stemi') || lowerPrompt.includes('chest pain')) {
        const cardOptions = [
          'ECG localization: Anterior vs Inferior vs Lateral MI',
          'Reperfusion criteria: Primary PCI vs Fibrinolysis windows',
          'Post-MI mechanical complications timeline (Papillary muscle vs VSD)'
        ];
        cardOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else if (lowerText.includes('diabetic nephropathy') || lowerText.includes('glomerulosclerosis') || lowerPrompt.includes('nephropathy')) {
        const renalOptions = [
          'Kimmelstiel-Wilson nodule histology vs Diffuse glomerulosclerosis',
          'SGLT2 inhibitors and ACEi/ARBs in renal hemodynamic protection',
          'Microalbuminuria vs Macroalbuminuria diagnostic thresholds'
        ];
        renalOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else if (lowerText.includes('antipsychotic') || lowerText.includes('haloperidol') || lowerPrompt.includes('antipsychotic')) {
        const pharmOptions = [
          'Receptor selectivity and Extrapyramidal Symptoms (EPS) mechanism',
          'Neuroleptic Malignant Syndrome (NMS) vs Serotonin Syndrome criteria',
          'Metabolic syndrome and monitoring for Olanzapine & Clozapine'
        ];
        pharmOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else if (lowerText.includes('fever of unknown origin') || lowerText.includes('fuo') || lowerPrompt.includes('fuo')) {
        const fuoOptions = [
          'Diagnostic workup algorithm and minimum investigation criteria for FUO',
          'Infectious vs Autoimmune etiologies of prolonged fever in India',
          'Role of PET-CT and bone marrow biopsy in occult fever evaluation'
        ];
        fuoOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else if (lowerText.includes('shock') || lowerPrompt.includes('shock')) {
        const shockOptions = [
          'Fluid resuscitation endpoints: MAP, ScvO2, and Lactate clearance',
          'Vasopressor receptor profiles: Norepinephrine vs Vasopressin',
          'Septic Shock vs Cardiogenic Shock hemodynamic profiles'
        ];
        shockOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else if (lowerText.includes('viva') || lowerPrompt.includes('viva') || lowerText.includes('oral exam')) {
        const vivaOptions = [
          'Ask me the next high-yield viva question',
          'Explain the underlying molecular and pathological mechanism',
          'What are the key exam pitfalls and common examiner cross-questions?'
        ];
        vivaOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      } else {
        const generalOptions = [
          'Key clinical contraindications and drug interactions',
          'High-yield NEET-PG / USMLE practice question on this topic',
          'Diagnostic algorithm and gold standard confirmatory tests'
        ];
        generalOptions.forEach(opt => { if (!followups.includes(opt) && followups.length < 3) followups.push(opt); });
      }
    }

    return followups.slice(0, 3);
  }

  // ── Answer Actions Toolbar Builder ──────────────────────────────────────────
  function createAnswerActionToolbar(row, bubble, text, sources, userPrompt) {
    const toolbar = document.createElement('div');
    toolbar.className = 'answer-actions-toolbar';
    toolbar.setAttribute('role', 'toolbar');
    toolbar.setAttribute('aria-label', 'Answer actions');

    // 1. Copy Answer
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'answer-action-btn copy-btn';
    copyBtn.title = 'Copy answer to clipboard';
    copyBtn.setAttribute('aria-label', 'Copy answer');
    copyBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
    `;
    setupCopyAnswer(copyBtn, bubble);
    toolbar.appendChild(copyBtn);

    // 2. Regenerate Answer
    const regenBtn = document.createElement('button');
    regenBtn.type = 'button';
    regenBtn.className = 'answer-action-btn regen-btn';
    regenBtn.title = 'Regenerate this response';
    regenBtn.setAttribute('aria-label', 'Regenerate response');
    regenBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>Regenerate</span>
    `;
    setupRegenerate(regenBtn, userPrompt);
    toolbar.appendChild(regenBtn);

    // 3. Feedback Group (Helpful / Not Helpful)
    const feedbackGroup = document.createElement('div');
    feedbackGroup.className = 'feedback-btn-group';
    feedbackGroup.setAttribute('role', 'group');
    feedbackGroup.setAttribute('aria-label', 'Response feedback');

    const helpfulBtn = document.createElement('button');
    helpfulBtn.type = 'button';
    helpfulBtn.className = 'answer-action-btn feedback-btn helpful-btn';
    helpfulBtn.title = 'Helpful explanation';
    helpfulBtn.setAttribute('aria-label', 'Mark as helpful');
    helpfulBtn.setAttribute('aria-pressed', 'false');
    helpfulBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
      </svg>
    `;

    const notHelpfulBtn = document.createElement('button');
    notHelpfulBtn.type = 'button';
    notHelpfulBtn.className = 'answer-action-btn feedback-btn not-helpful-btn';
    notHelpfulBtn.title = 'Not helpful';
    notHelpfulBtn.setAttribute('aria-label', 'Mark as not helpful');
    notHelpfulBtn.setAttribute('aria-pressed', 'false');
    notHelpfulBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
      </svg>
    `;

    setupFeedbackGroup(helpfulBtn, notHelpfulBtn);
    feedbackGroup.appendChild(helpfulBtn);
    feedbackGroup.appendChild(notHelpfulBtn);
    toolbar.appendChild(feedbackGroup);

    // 4. Read Aloud (if browser supports SpeechSynthesis)
    if ('speechSynthesis' in window) {
      const speakBtn = document.createElement('button');
      speakBtn.type = 'button';
      speakBtn.className = 'answer-action-btn speak-btn';
      speakBtn.title = 'Read answer aloud';
      speakBtn.setAttribute('aria-label', 'Read answer aloud');
      speakBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read Aloud</span>
      `;
      setupReadAloud(speakBtn, bubble);
      toolbar.appendChild(speakBtn);
    }

    // 5. View Sources (only when sources exist)
    if (Array.isArray(sources) && sources.length > 0) {
      const sourcesBtn = document.createElement('button');
      sourcesBtn.type = 'button';
      sourcesBtn.className = 'answer-action-btn view-sources-btn';
      sourcesBtn.title = 'View verified medical citations';
      sourcesBtn.setAttribute('aria-label', `View ${sources.length} verified sources`);
      sourcesBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5z"/>
        </svg>
        <span>Sources (${sources.length})</span>
      `;
      sourcesBtn.addEventListener('click', () => {
        const sourcesContainer = row.querySelector('.message-sources');
        if (sourcesContainer) {
          sourcesContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          const detailsList = sourcesContainer.querySelectorAll('details');
          detailsList.forEach(d => { d.open = true; });
        }
      });
      toolbar.appendChild(sourcesBtn);
    }

    return toolbar;
  }

  // ── Contextual Follow-up Suggestions Builder ────────────────────────────────
  function createFollowupSuggestions(text, userPrompt) {
    const followups = generateContextualFollowups(text, userPrompt);
    if (!followups || followups.length === 0) return null;

    const wrap = document.createElement('div');
    wrap.className = 'suggested-followups-wrap';

    const label = document.createElement('span');
    label.className = 'followups-label';
    label.textContent = 'Explore related concepts:';
    wrap.appendChild(label);

    const chipsList = document.createElement('div');
    chipsList.className = 'followup-chips';

    followups.forEach((prompt) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'followup-chip';
      chip.dataset.prompt = prompt;
      chip.setAttribute('aria-label', `Ask: ${prompt}`);
      chip.innerHTML = `
        <span class="followup-chip-icon" aria-hidden="true">💡</span>
        <span class="followup-chip-text">${escapeHtml(prompt)}</span>
      `;
      chip.addEventListener('click', () => {
        if (isGenerating) return;
        submitMessage(prompt);
      });
      chipsList.appendChild(chip);
    });

    wrap.appendChild(chipsList);
    return wrap;
  }

  // ── Input Box Auto-Grow & Char Counter ──────────────────────────────────────
  function handleInputChange() {
    if (!messageInput) return;
    messageInput.style.height = 'auto';
    const newHeight = Math.min(messageInput.scrollHeight, 140);
    messageInput.style.height = `${newHeight}px`;

    const len = messageInput.value.length;
    if (charCounter) {
      charCounter.textContent = `${len.toLocaleString()} / ${MAX_LENGTH.toLocaleString()}`;
      if (len >= MAX_LENGTH) {
        charCounter.className = 'char-counter danger';
      } else if (len >= MAX_LENGTH * 0.85) {
        charCounter.className = 'char-counter warning';
      } else {
        charCounter.className = 'char-counter';
      }
    }

    if (sendBtn) {
      sendBtn.disabled = !messageInput.value.trim() || isGenerating;
    }
  }

  messageInput?.addEventListener('input', handleInputChange);

  // Send on Enter (Shift+Enter inserts newline)
  messageInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn && !sendBtn.disabled && !isGenerating) {
        chatForm?.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    }
  });

  // Mobile Sidebar Drawer
  function openSidebar() {
    sidebar?.classList.add('open');
    sidebarBackdrop?.classList.add('active');
  }

  function closeSidebar() {
    sidebar?.classList.remove('open');
    sidebarBackdrop?.classList.remove('active');
  }

  openSidebarBtn?.addEventListener('click', openSidebar);
  closeSidebarBtn?.addEventListener('click', closeSidebar);
  sidebarBackdrop?.addEventListener('click', closeSidebar);

  // Suggestion Prompt Cards
  suggestionCards.forEach((card) => {
    card.addEventListener('click', () => {
      if (isGenerating) return;
      const prompt = card.getAttribute('data-prompt');
      if (prompt && messageInput) {
        messageInput.value = prompt;
        handleInputChange();
        chatForm?.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  });

  // Scroll to Bottom Helper
  function scrollToBottom() {
    if (chatViewport) {
      chatViewport.scrollTop = chatViewport.scrollHeight;
    }
  }

  // Interactive MCQ Card Renderer ──────────────────────────────────────────
  function createInteractiveMcqCard(mcq) {
    if (!mcq || !Array.isArray(mcq.options) || mcq.options.length === 0) {
      return null;
    }

    const card = document.createElement('div');
    card.className = 'interactive-mcq-card';
    card.setAttribute('role', 'group');
    card.setAttribute('aria-label', 'MBBS Practice Question');

    const header = document.createElement('div');
    header.className = 'mcq-card-header';

    const badge = document.createElement('span');
    badge.className = 'mcq-badge';
    badge.innerHTML = `
      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2">
        <path d="M9 11l3 3L22 4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>${escapeHtml(mcq.subject ? `Practice · ${mcq.subject}` : 'MBBS Practice Question')}</span>
    `;
    header.appendChild(badge);

    if (mcq.question_number) {
      const qnum = document.createElement('span');
      qnum.className = 'mcq-subtext';
      qnum.textContent = `Q${mcq.question_number}`;
      header.appendChild(qnum);
    } else if (mcq.competency) {
      const sub = document.createElement('span');
      sub.className = 'mcq-subtext';
      sub.textContent = mcq.competency;
      header.appendChild(sub);
    }
    card.appendChild(header);

    // Render case vignette / stem if provided
    const stemText = mcq.stem || mcq.case_vignette || '';
    if (stemText.trim()) {
      const stemEl = document.createElement('div');
      stemEl.className = 'mcq-stem';
      stemEl.innerHTML = formatSafeMarkdown(stemText);
      card.appendChild(stemEl);
    }

    // Options list
    const optionsList = document.createElement('div');
    optionsList.className = 'mcq-options-list';
    optionsList.setAttribute('role', 'radiogroup');
    optionsList.setAttribute('aria-label', 'Select the best answer');

    const isAnswered = Boolean(mcq.answered);
    const selectedKey = (mcq.user_answer || '').toUpperCase().trim();
    let submitted = isAnswered; // guard against double-submit

    const allBtns = [];

    mcq.options.forEach((opt, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'mcq-option-btn';
      btn.setAttribute('role', 'radio');
      const optKey = (opt.key || String.fromCharCode(65 + idx)).toUpperCase().trim();
      const isSelected = isAnswered && selectedKey === optKey;
      btn.setAttribute('aria-checked', isSelected ? 'true' : 'false');
      btn.setAttribute('aria-label', `Option ${optKey}: ${opt.text}`);
      btn.dataset.key = optKey;
      btn.dataset.questionId = mcq.question_id || '';
      btn.dataset.idx = idx;
      allBtns.push(btn);

      if (isAnswered) {
        btn.disabled = true;
        btn.setAttribute('aria-disabled', 'true');
        if (isSelected) btn.classList.add('selected');
      }

      btn.innerHTML = `
        <span class="mcq-option-key" aria-hidden="true">${escapeHtml(optKey)}</span>
        <span class="mcq-option-text">${formatInlineMarkdown(opt.text)}</span>
      `;

      // Click handler
      if (!isAnswered) {
        btn.addEventListener('click', () => {
          if (submitted || btn.disabled || isGenerating) return;
          submitted = true;

          // Immediately disable all options to prevent duplicate submissions
          allBtns.forEach((b) => {
            b.disabled = true;
            b.setAttribute('aria-disabled', 'true');
          });
          btn.classList.add('selected');
          btn.setAttribute('aria-checked', 'true');

          // Submit the student's chosen option
          const submissionText = `Option ${optKey}: ${opt.text}`;
          submitMessage(submissionText, {
            selected_option: optKey,
            question_id: mcq.question_id || ''
          });
        });

        // Keyboard: arrow keys move focus between options
        btn.addEventListener('keydown', (e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
            e.preventDefault();
            const next = allBtns[idx + 1];
            if (next && !next.disabled) next.focus();
          } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
            e.preventDefault();
            const prev = allBtns[idx - 1];
            if (prev && !prev.disabled) prev.focus();
          } else if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            btn.click();
          }
        });
      }

      optionsList.appendChild(btn);
    });

    card.appendChild(optionsList);

    // Footer with mandatory practice disclaimer
    const footer = document.createElement('div');
    footer.className = 'mcq-card-footer';
    const disclaimerText = mcq.disclaimer || 'MBBS Practice Question (Not Official Exam Content)';
    footer.innerHTML = `
      <span class="mcq-disclaimer">ℹ️ ${escapeHtml(disclaimerText)}</span>
    `;
    card.appendChild(footer);

    return card;
  }

  // ── Append Message Row ──────────────────────────────────────────────────────
  function appendMessage(role, text, sources = [], triage = {}, timestamp = null, interactiveMcq = null) {
    if (!messagesContainer) return;

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const timeStr = timestamp ? formatMessageTime(timestamp) : getFormattedTime();

    if (role === 'bot') {
      const avatar = document.createElement('div');
      avatar.className = 'message-avatar';
      avatar.setAttribute('aria-label', 'MediAssist AI Avatar');
      avatar.textContent = triage && triage.is_emergency ? '🚨' : 'MA';
      if (triage && triage.is_emergency) {
        avatar.style.background = '#ef4444';
        avatar.style.color = '#ffffff';
      }
      row.appendChild(avatar);
    }

    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'message-body-wrap';

    const meta = document.createElement('span');
    meta.className = 'message-meta';
    meta.textContent = role === 'user' ? `You · ${timeStr}` : `MediAssist · ${timeStr}`;
    bodyWrap.appendChild(meta);

    // Emergency Warning Card
    if (role === 'bot' && triage && triage.is_emergency) {
      const alertCard = document.createElement('div');
      alertCard.className = 'emergency-alert-card';
      alertCard.setAttribute('role', 'alert');
      const safeCondition = escapeHtml(String(triage.condition || 'Life-Threatening Emergency'));
      alertCard.innerHTML = `
        <div class="emergency-card-header">
          <span class="emergency-siren-icon">🚨</span>
          <div>
            <strong class="emergency-card-title">URGENT MEDICAL WARNING</strong>
            <span class="emergency-card-sub">Potential ${safeCondition}</span>
          </div>
        </div>
        <p class="emergency-card-body">
          The symptoms you described require immediate professional evaluation. <strong>Do not delay or attempt home treatment.</strong>
        </p>
        <div class="emergency-card-helplines">
          <a href="tel:112" class="emergency-tel-pill primary">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
            <span>Call 112 (National Emergency, India)</span>
          </a>
          <a href="tel:108" class="emergency-tel-pill">
            <span>Call 108 (Ambulance, India)</span>
          </a>
        </div>
      `;
      bodyWrap.appendChild(alertCard);
    } else if (role === 'bot' && triage && triage.severity === 'ROUTINE_GUIDANCE') {
      const guidancePill = document.createElement('div');
      guidancePill.className = 'symptom-guidance-pill';
      guidancePill.innerHTML = `
        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/></svg>
        <span>Symptom Guidance · Educational Assessment</span>
      `;
      bodyWrap.appendChild(guidancePill);
    }

    // Viva Voce mode banner
    if (role === 'bot' && triage) {
      const practiceMode = (triage.practice_mode || '').toLowerCase();
      const isVivaMode = practiceMode.includes('viva') || practiceMode.includes('oral_exam') || practiceMode.includes('oral exam');
      const lowerText = (text || '').toLowerCase();
      const looksLikeViva = !isVivaMode && (
        lowerText.includes('viva voce') ||
        lowerText.includes('oral exam') ||
        (lowerText.includes('question:') && (lowerText.includes('viva') || lowerText.includes('examiner')))
      );
      if (isVivaMode || looksLikeViva) {
        const vivaBanner = document.createElement('div');
        vivaBanner.className = 'viva-session-banner';
        vivaBanner.setAttribute('aria-label', 'Viva Voce Session');
        vivaBanner.innerHTML = `<span class="viva-icon" aria-hidden="true">🎙️</span><span>Viva Voce Session · Oral Exam Practice</span>`;
        bodyWrap.insertBefore(vivaBanner, bodyWrap.firstChild);
      }
    }

    // Message Bubble
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    if (role === 'user') {
      bubble.textContent = text;
      lastUserMessage = text;
    } else {
      bubble.innerHTML = formatSafeMarkdown(text);
      // Interactive MCQ card rendered inside message bubble
      const mcqObj = interactiveMcq || (triage && triage.interactive_mcq) || null;
      if (mcqObj) {
        const mcqCard = createInteractiveMcqCard(mcqObj);
        if (mcqCard) {
          bubble.appendChild(mcqCard);
        }
      }
    }
    bodyWrap.appendChild(bubble);

    // Chat Answer Action Toolbar & Contextual Follow-up Suggestions
    if (role === 'bot') {
      const actionToolbar = createAnswerActionToolbar(row, bubble, text, sources, lastUserMessage);
      bodyWrap.appendChild(actionToolbar);

      const followupsEl = createFollowupSuggestions(text, lastUserMessage);
      if (followupsEl) {
        bodyWrap.appendChild(followupsEl);
      }
    }

    // Verified Medical Sources & Interactive Citations
    if (role === 'bot' && Array.isArray(sources) && sources.length > 0) {
      activeEvidenceSources = sources;
      bodyWrap.appendChild(renderSourcesContainer(sources));
    }


    row.appendChild(bodyWrap);
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  // ── Render Sources Container with Interactive Citation Badges ───────────────
  function renderSourcesContainer(sources) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'message-sources';

    const sourcesHeader = document.createElement('span');
    sourcesHeader.className = 'sources-label';
    sourcesHeader.innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5z"/>
      </svg>
      <span>Verified Medical References:</span>
    `;
    sourcesContainer.appendChild(sourcesHeader);

    const chipsWrapper = document.createElement('div');
    chipsWrapper.className = 'sources-chips';

    sources.forEach((src, idx) => {
      let displayTitle = (src.title || 'Medical Reference').trim();
      if (displayTitle.endsWith('.md') || displayTitle.endsWith('.txt')) {
        displayTitle = displayTitle.replace(/\.(md|txt)$/i, '').replace(/_/g, ' ');
      }
      if (displayTitle.indexOf(' ') === -1 && displayTitle.indexOf('_') !== -1) {
        displayTitle = displayTitle.replace(/_/g, ' ');
      }

      // Accessible, interactive citation badge button
      const chipBtn = document.createElement('button');
      chipBtn.type = 'button';
      chipBtn.className = 'source-chip source-chip--interactive';
      chipBtn.setAttribute('aria-haspopup', 'dialog');
      chipBtn.setAttribute('aria-expanded', 'false');
      chipBtn.setAttribute('title', `Inspect verified evidence for: ${displayTitle}`);
      chipBtn.innerHTML = `
        <span class="citation-badge-num">[${idx + 1}]</span>
        <span class="source-chip-text">${escapeHtml(displayTitle)}</span>
        <span class="chip-ext-icon" aria-hidden="true">📖</span>
        <span class="chip-action-hint">Inspect</span>
      `;

      chipBtn.addEventListener('click', () => {
        openEvidenceDrawer(src, idx);
      });

      chipsWrapper.appendChild(chipBtn);
    });

    sourcesContainer.appendChild(chipsWrapper);
    return sourcesContainer;
  }

  // ── Smart Auto-Scroll Controller ───────────────────────────────────────────
  function smartScrollToBottom() {
    if (!chatViewport) return;
    const threshold = 140;
    const isNearBottom = (chatViewport.scrollHeight - chatViewport.scrollTop - chatViewport.clientHeight) <= threshold;
    if (isNearBottom) {
      chatViewport.scrollTo({
        top: chatViewport.scrollHeight,
        behavior: 'smooth'
      });
    }
  }

  // ── Streaming Message Row Controller ────────────────────────────────────────
  function createStreamingMessageRow() {
    const row = document.createElement('div');
    row.className = 'message-row bot';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.setAttribute('aria-label', 'MediAssist AI Avatar');
    avatar.textContent = 'MA';
    row.appendChild(avatar);

    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'message-body-wrap';

    const meta = document.createElement('span');
    meta.className = 'message-meta';
    meta.textContent = `MediAssist · ${getFormattedTime()}`;
    bodyWrap.appendChild(meta);

    // Real-time streaming status badge
    const statusBadge = document.createElement('div');
    statusBadge.className = 'streaming-status-badge';
    statusBadge.innerHTML = `
      <span class="streaming-pulse-dot" aria-hidden="true"></span>
      <span>Generating clinical response...</span>
    `;
    bodyWrap.appendChild(statusBadge);

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.setAttribute('role', 'status');
    bubble.setAttribute('aria-live', 'polite');
    bubble.innerHTML = `<span class="streaming-cursor" aria-hidden="true">▍</span>`;
    bodyWrap.appendChild(bubble);

    row.appendChild(bodyWrap);

    let isFinalized = false;

    return {
      row,
      updateText(newText) {
        if (isFinalized) return;
        const html = formatSafeMarkdown(newText);
        bubble.innerHTML = html + `<span class="streaming-cursor" aria-hidden="true">▍</span>`;
      },
      finalize(finalText, sources = [], triage = {}, timestamp = null, interactiveMcq = null) {
        if (isFinalized) return;
        isFinalized = true;

        statusBadge.remove();
        bubble.innerHTML = formatSafeMarkdown(finalText);

        if (triage && triage.is_emergency) {
          avatar.textContent = '🚨';
          avatar.style.background = '#ef4444';
          avatar.style.color = '#ffffff';

          const alertCard = document.createElement('div');
          alertCard.className = 'emergency-alert-card';
          alertCard.setAttribute('role', 'alert');
          const safeCondition = escapeHtml(String(triage.condition || 'Life-Threatening Emergency'));
          alertCard.innerHTML = `
            <div class="emergency-card-header">
              <span class="emergency-siren-icon">🚨</span>
              <div>
                <strong class="emergency-card-title">URGENT MEDICAL WARNING</strong>
                <span class="emergency-card-sub">Potential ${safeCondition}</span>
              </div>
            </div>
            <p class="emergency-card-body">
              The symptoms you described require immediate professional evaluation. <strong>Do not delay or attempt home treatment.</strong>
            </p>
            <div class="emergency-card-helplines">
              <a href="tel:112" class="emergency-tel-pill primary"><span>Call 112 (National Emergency, India)</span></a>
              <a href="tel:108" class="emergency-tel-pill"><span>Call 108 (Ambulance, India)</span></a>
            </div>
          `;
          bodyWrap.insertBefore(alertCard, bubble);
        }

        // Attach interactive MCQ if present
        const mcqObj = interactiveMcq || (triage && triage.interactive_mcq) || null;
        if (mcqObj) {
          const mcqCard = createInteractiveMcqCard(mcqObj);
          if (mcqCard) bubble.appendChild(mcqCard);
        }

        // Action toolbar
        const actionToolbar = createAnswerActionToolbar(row, bubble, finalText, sources, lastUserMessage);
        bodyWrap.appendChild(actionToolbar);

        // Contextual follow-up suggestions
        const followupsEl = createFollowupSuggestions(finalText, lastUserMessage);
        if (followupsEl) bodyWrap.appendChild(followupsEl);

        // Sources with interactive citation badges
        if (Array.isArray(sources) && sources.length > 0) {
          activeEvidenceSources = sources;
          bodyWrap.appendChild(renderSourcesContainer(sources));
        }

        smartScrollToBottom();
      }
    };
  }

  // ── Typing Indicator ────────────────────────────────────────────────────────
  function appendTypingIndicator() {
    if (!messagesContainer) return null;

    const row = document.createElement('div');
    row.className = 'message-row bot typing-row';
    row.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'MA';

    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'message-body-wrap';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble typing-bubble';
    bubble.setAttribute('role', 'status');
    bubble.setAttribute('aria-label', 'MediAssist is preparing a response');
    bubble.innerHTML = `
      <div class="typing-indicator-wrap">
        <div class="typing-dots" aria-label="MediAssist is thinking">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
        <span class="typing-status-text">Analyzing inquiry & clinical safety guidelines...</span>
      </div>
    `;

    bodyWrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bodyWrap);

    messagesContainer.appendChild(row);
    scrollToBottom();

    const statusEl = bubble.querySelector('.typing-status-text');
    const stages = [
      { delay: 1100, text: 'Searching verified medical knowledge base...' },
      { delay: 2600, text: 'Cross-referencing evidence & clinical guidelines...' },
      { delay: 4800, text: 'Synthesizing response with Gemini AI...' },
      { delay: 7800, text: 'Finalizing verified explanation & citations...' },
    ];

    const timerIds = [];
    stages.forEach(function (stage) {
      const tid = setTimeout(function () {
        if (statusEl && row.isConnected) {
          statusEl.style.opacity = '0';
          setTimeout(function () {
            if (statusEl && row.isConnected) {
              statusEl.textContent = stage.text;
              statusEl.style.opacity = '1';
            }
          }, 140);
        }
      }, stage.delay);
      timerIds.push(tid);
    });

    const originalRemove = row.remove.bind(row);
    row.remove = function () {
      timerIds.forEach(function (tid) { clearTimeout(tid); });
      originalRemove();
    };

    return row;
  }

  // ── Error Message with Retry ────────────────────────────────────────────────
  function appendErrorMessage(errorMsg) {
    if (!messagesContainer) return;

    const row = document.createElement('div');
    row.className = 'message-row bot error-msg';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '⚠️';

    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'message-body-wrap';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const errorText = document.createElement('p');
    errorText.textContent = errorMsg;
    bubble.appendChild(errorText);

    if (lastUserMessage) {
      const retryBtn = document.createElement('button');
      retryBtn.className = 'retry-action-btn';
      retryBtn.type = 'button';
      retryBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Retry request</span>
      `;
      retryBtn.addEventListener('click', () => {
        row.remove();
        submitMessage(lastUserMessage);
      });
      bubble.appendChild(retryBtn);
    }

    bodyWrap.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(bodyWrap);

    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  // ── Send Message (Non-Streaming Fallback) ────────────────────────────────────
  async function submitMessageNonStreaming(text, extraPayload = {}) {
    const typingIndicator = appendTypingIndicator();
    try {
      const csrfToken = getCsrfToken();
      const headers = { 'Content-Type': 'application/json' };
      if (csrfToken) {
        headers['X-CSRFToken'] = csrfToken;
      }

      const payload = { message: text };
      if (currentConversationId) payload.conversation_id = currentConversationId;
      if (extraPayload && extraPayload.selected_option) payload.selected_option = extraPayload.selected_option;
      if (extraPayload && extraPayload.question_id) payload.question_id = extraPayload.question_id;

      const response = await fetch('/chat', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      typingIndicator?.remove();

      if (response.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again to continue.');
        return;
      }

      if (response.status === 429) {
        appendErrorMessage('Rate limit reached. Please wait a moment before sending another message.');
        return;
      }

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const errorMsg = data.error || `Server returned error (${response.status})`;
        appendErrorMessage(errorMsg);
        return;
      }

      if (!data.response) {
        appendErrorMessage('Received empty response from MediAssist.');
        return;
      }

      const mcqPayload = data.interactive_mcq || (data.triage && data.triage.interactive_mcq) || null;
      appendMessage('bot', data.response, data.sources || [], data.triage || {}, data.created_at, mcqPayload);

      if (data.conversation_id) {
        currentConversationId = data.conversation_id;
        loadConversationDocuments(currentConversationId);
      }
      await loadConversations();
    } catch (networkErr) {
      typingIndicator?.remove();
      appendErrorMessage('Unable to reach MediAssist. Please check your network connection.');
      console.error('Non-streaming chat error:', networkErr);
    }
  }

  // ── Send Message (SSE Real-Time Streaming with Fallback) ─────────────────────
  async function submitMessage(text, extraPayload = {}) {
    if (!text || isGenerating) return;

    lastUserMessage = text;

    // Transition from welcome screen to messages container
    if (welcomeScreen && welcomeScreen.style.display !== 'none') {
      welcomeScreen.style.display = 'none';
      if (messagesContainer) messagesContainer.style.display = 'flex';
    }

    appendMessage('user', text);

    if (messageInput) {
      messageInput.value = '';
      handleInputChange();
    }

    isGenerating = true;
    chatForm?.setAttribute('aria-busy', 'true');
    if (sendBtn) sendBtn.disabled = true;

    // Verify browser ReadableStream capability
    if (!window.ReadableStream || typeof fetch !== 'function') {
      try {
        await submitMessageNonStreaming(text, extraPayload);
      } finally {
        isGenerating = false;
        chatForm?.setAttribute('aria-busy', 'false');
        handleInputChange();
      }
      return;
    }

    const streamRow = createStreamingMessageRow();
    messagesContainer.appendChild(streamRow.row);
    scrollToBottom();

    let fullStreamedText = '';
    let streamSources = [];
    let streamTriage = {};
    let streamSucceeded = false;

    try {
      const csrfToken = getCsrfToken();
      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      };
      if (csrfToken) {
        headers['X-CSRFToken'] = csrfToken;
      }

      const payload = { message: text, stream: true };
      if (currentConversationId) payload.conversation_id = currentConversationId;
      if (extraPayload && extraPayload.selected_option) payload.selected_option = extraPayload.selected_option;
      if (extraPayload && extraPayload.question_id) payload.question_id = extraPayload.question_id;

      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        streamRow.row.remove();
        handleSessionExpired('Your session has expired. Please sign in again to continue.');
        return;
      }

      if (response.status === 429) {
        streamRow.row.remove();
        appendErrorMessage('Rate limit reached. Please wait a moment before sending another message.');
        return;
      }

      const contentType = response.headers.get('Content-Type') || '';
      if (!response.ok || !contentType.includes('text/event-stream') || !response.body) {
        // Fall back to non-streaming endpoint
        streamRow.row.remove();
        await submitMessageNonStreaming(text, extraPayload);
        return;
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
        buffer = lines.pop(); // Keep partial trailing line

        for (let i = 0; i < lines.length; i++) {
          const rawLine = lines[i];
          const line = rawLine.trim();
          if (!line) {
            currentEvent = 'chunk';
            continue;
          }

          if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            try {
              const data = JSON.parse(dataStr);

              if (currentEvent === 'start') {
                if (data.conversation_id) {
                  currentConversationId = data.conversation_id;
                }
              } else if (currentEvent === 'chunk') {
                const chunkText = data.text || '';
                fullStreamedText += chunkText;
                streamRow.updateText(fullStreamedText);
                smartScrollToBottom();
              } else if (currentEvent === 'evidence') {
                streamSources = data.sources || [];
                activeEvidenceSources = streamSources;
              } else if (currentEvent === 'complete') {
                streamSucceeded = true;
                const finalReply = data.response || fullStreamedText;
                const finalSources = data.sources || streamSources;
                const finalTriage = data.triage || streamTriage;
                activeEvidenceSources = finalSources;

                streamRow.finalize(finalReply, finalSources, finalTriage, data.created_at, data.interactive_mcq);

                if (data.conversation_id) {
                  currentConversationId = data.conversation_id;
                  loadConversationDocuments(currentConversationId);
                }
                await loadConversations();
              } else if (currentEvent === 'error') {
                streamRow.row.remove();
                appendErrorMessage(data.error || 'MediAssist encountered an issue generating the response.');
                return;
              }
            } catch (jsonErr) {
              console.warn('Failed to parse SSE data frame:', jsonErr);
            }
          }
        }
      }

      if (!streamSucceeded && fullStreamedText) {
        streamRow.finalize(fullStreamedText, streamSources, streamTriage);
      }

    } catch (networkErr) {
      console.warn('Streaming fetch failed, attempting non-streaming fallback:', networkErr);
      streamRow.row.remove();
      try {
        await submitMessageNonStreaming(text, extraPayload);
      } catch (fallbackErr) {
        appendErrorMessage('Unable to reach MediAssist. Please check your network connection.');
      }
    } finally {
      isGenerating = false;
      chatForm?.setAttribute('aria-busy', 'false');
      handleInputChange();
    }
  }


  // ── Conversation History Management ─────────────────────────────────────────
  async function loadConversations() {
    if (!conversationsList) return;
    try {
      // Show non-destructive loading state if empty
      if (conversationsList.children.length === 0) {
        conversationsList.innerHTML = '<li class="history-state-item loading"><span>Loading consultations...</span></li>';
      }

      const res = await fetch('/api/conversations');
      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        conversationsList.innerHTML = `
          <li class="history-state-item session-expired">
            <span>Session expired</span>
            <a href="/login?next=/chat" class="history-retry-btn">Sign In</a>
          </li>
        `;
        return;
      }

      if (!res.ok) {
        throw new Error(`Failed to load history (${res.status})`);
      }

      const data = await res.json().catch(() => ({}));
      const convs = Array.isArray(data.conversations) ? data.conversations : [];

      if (historyCountBadge) {
        historyCountBadge.textContent = String(convs.length);
      }

      if (convs.length === 0) {
        conversationsList.innerHTML = '<li class="history-state-item empty"><span>No past consultations</span></li>';
        return;
      }

      conversationsList.innerHTML = '';
      convs.forEach((conv) => {
        const li = document.createElement('li');
        li.className = 'conversation-item';
        if (conv.id === currentConversationId) {
          li.classList.add('active');
        }
        li.dataset.convId = conv.id;
        li.setAttribute('role', 'listitem');

        const relTime = formatRelativeTime(conv.updated_at);
        const count = conv.message_count || 0;
        const metaText = relTime ? `${relTime} · ${count} ${count === 1 ? 'msg' : 'msgs'}` : `${count} msgs`;
        const titleText = conv.title || 'Consultation';

        li.innerHTML = `
          <button type="button" class="conversation-item-btn" title="${escapeHtml(titleText)}" aria-label="Open consultation: ${escapeHtml(titleText)}">
            <svg class="conv-item-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <div class="conv-text-wrap">
              <span class="conv-item-title">${escapeHtml(titleText)}</span>
              <span class="conv-item-meta">${escapeHtml(metaText)}</span>
            </div>
          </button>
          <div class="conv-actions-wrap">
            <button type="button" class="conv-action-btn conv-rename-btn" title="Rename consultation" aria-label="Rename consultation: ${escapeHtml(titleText)}">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </button>
            <button type="button" class="conv-action-btn conv-delete-btn" title="Delete consultation" aria-label="Delete consultation: ${escapeHtml(titleText)}">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
            </button>
          </div>
        `;

        li.querySelector('.conversation-item-btn').addEventListener('click', () => {
          selectConversation(conv.id);
        });

        li.querySelector('.conv-rename-btn').addEventListener('click', (e) => {
          e.stopPropagation();
          renameConversation(conv.id, titleText);
        });

        li.querySelector('.conv-delete-btn').addEventListener('click', (e) => {
          e.stopPropagation();
          deleteConversation(conv.id, titleText);
        });

        conversationsList.appendChild(li);
      });
    } catch (err) {
      console.warn('Unable to load conversations:', err);
      if (conversationsList) {
        conversationsList.innerHTML = `
          <li class="history-state-item error">
            <span>Unable to load history</span>
            <button type="button" class="history-retry-btn" id="retryHistoryBtn" aria-label="Retry loading past consultations">Retry</button>
          </li>
        `;
        document.getElementById('retryHistoryBtn')?.addEventListener('click', () => {
          loadConversations();
        });
      }
    }
  }

  async function selectConversation(convId) {
    if (isGenerating) return;
    try {
      const res = await fetch(`/api/conversations/${convId}`);
      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        return;
      }
      if (!res.ok) return;

      const data = await res.json();
      const conv = data.conversation;
      if (!conv) return;

      currentConversationId = conv.id;

      // Update active state in sidebar
      document.querySelectorAll('.conversation-item').forEach((el) => {
        if (el.dataset.convId === conv.id) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      });

      // Render messages with preserved timestamps, sources, and triage cards
      if (messagesContainer) {
        messagesContainer.innerHTML = '';
        const messages = Array.isArray(conv.messages) ? conv.messages : [];

        if (messages.length > 0) {
          if (welcomeScreen) welcomeScreen.style.display = 'none';
          messagesContainer.style.display = 'flex';

          messages.forEach((msg) => {
            const role = msg.role === 'model' ? 'bot' : 'user';
            const evidence = msg.evidence || {};
            const sources = evidence.sources || [];
            const triage = evidence.triage || {};
            const mcqPayload = evidence.interactive_mcq || (triage && triage.interactive_mcq) || null;
            appendMessage(role, msg.content, sources, triage, msg.created_at, mcqPayload);
          });
          scrollToBottom();
        } else {
          if (welcomeScreen) welcomeScreen.style.display = 'block';
          messagesContainer.style.display = 'none';
        }
      }

      await loadConversationDocuments(conv.id);
      closeSidebar();
    } catch (err) {
      console.error('Failed to load conversation details:', err);
    }
  }

  async function renameConversation(convId, currentTitle) {
    const newTitle = window.prompt('Enter new title for this consultation:', currentTitle || '');
    if (newTitle === null) return; // User cancelled
    const trimmed = newTitle.trim();
    if (!trimmed) {
      alert('Consultation title cannot be empty.');
      return;
    }
    if (trimmed.length > 200) {
      alert('Title cannot exceed 200 characters.');
      return;
    }
    try {
      const csrfToken = getCsrfToken();
      const headers = { 'Content-Type': 'application/json' };
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;

      const res = await fetch(`/api/conversations/${convId}`, {
        method: 'PATCH',
        headers,
        body: JSON.stringify({ title: trimmed }),
      });

      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        return;
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        alert(errData.error || 'Failed to rename consultation');
        return;
      }

      await loadConversations();
    } catch (err) {
      console.error('Rename failed:', err);
      alert('Network error while renaming consultation.');
    }
  }

  async function deleteConversation(convId, title) {
    if (!confirm(`Are you sure you want to delete "${title || 'this consultation'}"?`)) return;
    try {
      const csrfToken = getCsrfToken();
      const headers = {};
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;

      const res = await fetch(`/api/conversations/${convId}`, {
        method: 'DELETE',
        headers: headers,
      });

      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        return;
      }

      if (res.ok) {
        if (currentConversationId === convId) {
          currentConversationId = null;
          if (messagesContainer) {
            messagesContainer.innerHTML = '';
            messagesContainer.style.display = 'none';
          }
          if (welcomeScreen) welcomeScreen.style.display = 'block';
          if (attachmentChipsList) attachmentChipsList.innerHTML = '';
          if (attachmentBar) attachmentBar.style.display = 'none';
          if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
        }
        await loadConversations();
      } else {
        const errData = await res.json().catch(() => ({}));
        alert(errData.error || 'Failed to delete consultation.');
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      alert('Network error while deleting consultation.');
    }
  }

  // ── Form Submit Listener ────────────────────────────────────────────────────
  chatForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = messageInput ? messageInput.value.trim() : '';
    if (text) {
      submitMessage(text);
    }
  });

  // ── New Consultation Handler ────────────────────────────────────────────────
  newChatBtn?.addEventListener('click', async () => {
    if (isGenerating) return;

    currentConversationId = null;
    document.querySelectorAll('.conversation-item').forEach((el) => el.classList.remove('active'));

    try {
      const csrfToken = getCsrfToken();
      const headers = { 'Content-Type': 'application/json' };
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;
      await fetch('/reset', { method: 'POST', headers });
    } catch (_) { /* ignore */ }

    if (messagesContainer) {
      messagesContainer.innerHTML = '';
      messagesContainer.style.display = 'none';
    }
    if (welcomeScreen) {
      welcomeScreen.style.display = 'block';
    }
    if (attachmentChipsList) {
      attachmentChipsList.innerHTML = '';
    }
    if (attachmentBar) {
      attachmentBar.style.display = 'none';
    }
    if (uploadProgressBanner) {
      uploadProgressBanner.style.display = 'none';
    }
    if (messageInput) {
      messageInput.value = '';
      messageInput.placeholder = 'Ask an MBBS topic, mechanism, or question about your attached notes...';
      handleInputChange();
      messageInput.focus();
    }
    closeSidebar();
    if (chatViewport) chatViewport.scrollTop = 0;
    await loadConversations();
  });

  // ── Academic Profile Switcher & Preferences ─────────────────────────────────
  function formatProfileBadge(year, exam) {
    const yearNames = {
      '1st_year': '1st Year',
      '2nd_year': '2nd Year',
      '3rd_year': '3rd Year',
      'final_year': 'Final Year',
      'intern': 'Intern',
    };
    const examNames = {
      'university_profs': 'Profs',
      'neet_pg_next': 'NEET-PG',
      'usmle_step1': 'USMLE Step 1',
      'usmle_step2': 'USMLE Step 2',
    };
    if (year && exam) {
      return `${yearNames[year] || year} · ${examNames[exam] || exam}`;
    }
    if (year) return `${yearNames[year] || year} MBBS`;
    if (exam) return examNames[exam] || exam;
    return 'MBBS Level';
  }

  async function loadStudentProfile() {
    try {
      const res = await fetch('/api/profile');
      if (!res.ok) return;
      const data = await res.json();
      const u = data.user || {};
      if (selectAcademicYear) selectAcademicYear.value = u.academic_year || '';
      if (selectTargetExam) selectTargetExam.value = u.target_exam || '';
      if (headerProfileLabel) {
        headerProfileLabel.textContent = formatProfileBadge(u.academic_year, u.target_exam);
      }
    } catch (err) {
      console.debug('Failed to load profile:', err);
    }
  }

  profileModalToggleBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (profilePopover) {
      profilePopover.hidden = !profilePopover.hidden;
      profileModalToggleBtn.setAttribute('aria-expanded', String(!profilePopover.hidden));
    }
  });

  document.addEventListener('click', (e) => {
    if (profilePopover && !profilePopover.hidden) {
      if (!profilePopover.contains(e.target) && !profileModalToggleBtn?.contains(e.target)) {
        profilePopover.hidden = true;
        profileModalToggleBtn?.setAttribute('aria-expanded', 'false');
      }
    }
  });

  profileForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (saveProfileBtn) saveProfileBtn.disabled = true;
    if (profileSaveStatus) profileSaveStatus.textContent = 'Saving...';
    try {
      const csrfToken = getCsrfToken();
      const headers = { 'Content-Type': 'application/json' };
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;

      const payload = {
        academic_year: selectAcademicYear ? selectAcademicYear.value : null,
        target_exam: selectTargetExam ? selectTargetExam.value : null,
      };

      const res = await fetch('/api/profile', {
        method: 'PATCH',
        headers,
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        const u = data.user || {};
        if (headerProfileLabel) {
          headerProfileLabel.textContent = formatProfileBadge(u.academic_year, u.target_exam);
        }
        if (profileSaveStatus) {
          profileSaveStatus.textContent = 'Saved!';
          setTimeout(() => {
            if (profileSaveStatus) profileSaveStatus.textContent = '';
            if (profilePopover) profilePopover.hidden = true;
            if (profileModalToggleBtn) profileModalToggleBtn.setAttribute('aria-expanded', 'false');
          }, 800);
        }
      } else {
        const err = await res.json().catch(() => ({}));
        if (profileSaveStatus) profileSaveStatus.textContent = err.error || 'Failed to save';
      }
    } catch (err) {
      if (profileSaveStatus) profileSaveStatus.textContent = 'Network error';
    } finally {
      if (saveProfileBtn) saveProfileBtn.disabled = false;
    }
  });

  // ── Keyboard Accessibility: Close Sidebar on Escape ─────────────────────────
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (sidebar?.classList.contains('open')) closeSidebar();
      if (profilePopover && !profilePopover.hidden) {
        profilePopover.hidden = true;
        profileModalToggleBtn?.setAttribute('aria-expanded', 'false');
      }
    }
  });

  // ── Document Attachment & Study Note Management ─────────────────────────────
  async function loadConversationDocuments(convId) {
    if (!attachmentBar || !attachmentChipsList) return;
    if (!convId) {
      attachmentChipsList.innerHTML = '';
      attachmentBar.style.display = 'none';
      return;
    }

    try {
      const res = await fetch(`/api/conversations/${convId}/documents`);
      if (res.status === 401) {
        handleSessionExpired('Your session has expired.');
        return;
      }
      if (!res.ok) {
        attachmentChipsList.innerHTML = '';
        attachmentBar.style.display = 'none';
        return;
      }

      const data = await res.json().catch(() => ({}));
      const docs = Array.isArray(data.documents) ? data.documents : [];

      if (docs.length === 0) {
        attachmentChipsList.innerHTML = '';
        attachmentBar.style.display = 'none';
        return;
      }

      attachmentChipsList.innerHTML = '';
      docs.forEach((doc) => {
        const chip = document.createElement('div');
        chip.className = 'attachment-chip';
        chip.dataset.docId = doc.id;

        const iconSpan = document.createElement('span');
        iconSpan.className = 'attachment-chip-icon';
        iconSpan.textContent = doc.file_type === 'pdf' ? '📄' : '📝';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'attachment-chip-name';
        nameSpan.title = doc.filename;
        nameSpan.textContent = doc.filename;

        const metaSpan = document.createElement('span');
        metaSpan.className = 'attachment-chip-meta';
        if (doc.status === 'processing') {
          metaSpan.textContent = '(Indexing...)';
          metaSpan.className = 'attachment-chip-meta attachment-chip-processing';
          pollDocumentStatus(convId, doc.id, doc.filename);
        } else if (doc.status === 'error') {
          metaSpan.textContent = '(Failed)';
          metaSpan.className = 'attachment-chip-meta attachment-chip-error';
          if (doc.error_message) metaSpan.title = doc.error_message;
        } else {
          const pageLabel = doc.page_count === 1 ? '1 page' : `${doc.page_count} pages`;
          metaSpan.textContent = `(${pageLabel})`;
        }

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'attachment-chip-remove';
        removeBtn.title = `Remove ${doc.filename}`;
        removeBtn.setAttribute('aria-label', `Remove attachment ${doc.filename}`);
        removeBtn.innerHTML = '&times;';
        removeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteAttachedDocument(convId, doc.id, doc.filename);
        });

        chip.appendChild(iconSpan);
        chip.appendChild(nameSpan);
        chip.appendChild(metaSpan);
        chip.appendChild(removeBtn);
        attachmentChipsList.appendChild(chip);
      });

      attachmentBar.style.display = 'flex';
    } catch (err) {
      console.warn('Failed to load conversation documents:', err);
    }
  }

  async function deleteAttachedDocument(convId, docId, filename) {
    if (!confirm(`Are you sure you want to remove "${filename || 'this attachment'}" from this consultation?`)) {
      return;
    }

    try {
      const csrfToken = getCsrfToken();
      const headers = {};
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;

      const res = await fetch(`/api/conversations/${convId}/documents/${docId}`, {
        method: 'DELETE',
        headers,
      });

      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        return;
      }

      if (res.ok) {
        await loadConversationDocuments(convId);
        await loadConversations();
      } else {
        const errData = await res.json().catch(() => ({}));
        alert(errData.error || 'Failed to remove attachment.');
      }
    } catch (err) {
      console.error('Delete attachment error:', err);
      alert('Network error while removing attachment.');
    }
  }

  const _activePollingDocs = new Set();

  async function pollDocumentStatus(convId, docId, filename, attempts = 0) {
    if (!convId || !docId) return;
    const pollKey = `${convId}:${docId}`;
    if (attempts === 0) {
      if (_activePollingDocs.has(pollKey)) return;
      _activePollingDocs.add(pollKey);
    }

    if (attempts > 80) {
      _activePollingDocs.delete(pollKey);
      if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
      return;
    }

    try {
      const res = await fetch(`/api/conversations/${convId}/documents/${docId}/status`);
      if (res.status === 401) {
        _activePollingDocs.delete(pollKey);
        return;
      }
      if (!res.ok) {
        _activePollingDocs.delete(pollKey);
        if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
        return;
      }

      const data = await res.json();
      const doc = data.document || {};

      if (doc.status === 'ready') {
        _activePollingDocs.delete(pollKey);
        if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
        await loadConversationDocuments(convId);
        await loadConversations();
        if (messageInput) {
          messageInput.placeholder = `Ask a question about ${filename}...`;
          messageInput.focus();
        }
      } else if (doc.status === 'error') {
        _activePollingDocs.delete(pollKey);
        if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
        await loadConversationDocuments(convId);
        alert(doc.error_message || `Failed to process ${filename}.`);
      } else {
        if (uploadProgressBanner) {
          uploadProgressBanner.style.display = 'flex';
          if (uploadProgressText) {
            uploadProgressText.textContent = `Indexing "${filename}" in background...`;
          }
        }
        setTimeout(() => pollDocumentStatus(convId, docId, filename, attempts + 1), 1500);
      }
    } catch (e) {
      console.debug('Polling document status error:', e);
      setTimeout(() => pollDocumentStatus(convId, docId, filename, attempts + 1), 2000);
    }
  }

  async function uploadStudyNoteFile(file) {
    if (!file) return;

    // Client-side extension validation
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'md'].includes(ext)) {
      alert('Only PDF documents (.pdf) and plain text study notes (.txt, .md) are supported.');
      return;
    }

    // Client-side 15 MB size check
    const MAX_UPLOAD_SIZE = 15 * 1024 * 1024;
    if (file.size > MAX_UPLOAD_SIZE) {
      alert(`"${file.name}" exceeds the 15 MB file size limit. Please upload a smaller excerpt or chapter.`);
      return;
    }

    // Ensure we have an active conversation ID or create one first
    if (!currentConversationId) {
      try {
        const csrfToken = getCsrfToken();
        const headers = { 'Content-Type': 'application/json' };
        if (csrfToken) headers['X-CSRFToken'] = csrfToken;

        const baseTitle = file.name.replace(/\.[^/.]+$/, '').slice(0, 40);
        const convRes = await fetch('/api/conversations', {
          method: 'POST',
          headers,
          body: JSON.stringify({ title: `Notes: ${baseTitle}` }),
        });

        if (convRes.status === 401) {
          handleSessionExpired('Your session has expired. Please sign in again.');
          return;
        }

        if (!convRes.ok) {
          alert('Failed to initialize a consultation session for this document.');
          return;
        }

        const convData = await convRes.json().catch(() => ({}));
        if (convData.conversation && convData.conversation.id) {
          currentConversationId = convData.conversation.id;
          await loadConversations();
        } else {
          alert('Failed to obtain consultation session.');
          return;
        }
      } catch (e) {
        console.error('Failed to create consultation for upload:', e);
        alert('Network error while starting consultation.');
        return;
      }
    }

    // Show upload and processing progress UI
    if (uploadProgressBanner) {
      if (uploadProgressText) {
        uploadProgressText.textContent = `Extracting, chunking & indexing "${file.name}"...`;
      }
      uploadProgressBanner.style.display = 'flex';
    }
    if (attachBtn) attachBtn.disabled = true;

    try {
      const csrfToken = getCsrfToken();
      const headers = {};
      if (csrfToken) headers['X-CSRFToken'] = csrfToken;

      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`/api/conversations/${currentConversationId}/upload`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (res.status === 401) {
        handleSessionExpired('Your session has expired. Please sign in again.');
        return;
      }

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        alert(data.error || `Upload failed with status ${res.status}`);
        return;
      }

      // Check if asynchronous background processing was returned
      if (data.status === 'processing' && data.document && data.document.id) {
        await loadConversationDocuments(currentConversationId);
        pollDocumentStatus(currentConversationId, data.document.id, file.name);
      } else {
        // Synchronous completion
        await loadConversationDocuments(currentConversationId);
        await loadConversations();
        if (uploadProgressBanner) uploadProgressBanner.style.display = 'none';
        if (messageInput) {
          messageInput.placeholder = `Ask a question about ${file.name}...`;
          messageInput.focus();
        }
      }
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Network error during file upload.');
    } finally {
      if (uploadProgressBanner && (!data || data.status !== 'processing')) {
        uploadProgressBanner.style.display = 'none';
      }
      if (attachBtn) attachBtn.disabled = false;
      if (fileAttachmentInput) fileAttachmentInput.value = '';
    }
  }

  // File input change & attach button click
  attachBtn?.addEventListener('click', () => {
    fileAttachmentInput?.click();
  });

  fileAttachmentInput?.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      uploadStudyNoteFile(files[0]);
    }
  });

  // Drag and Drop support on composer card and chat viewport
  const dropTargets = [composerCard, chatViewport].filter(Boolean);
  dropTargets.forEach((target) => {
    target.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (composerCard) composerCard.classList.add('drag-over');
    });

    target.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (composerCard && (!e.relatedTarget || !target.contains(e.relatedTarget))) {
        composerCard.classList.remove('drag-over');
      }
    });

    target.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (composerCard) composerCard.classList.remove('drag-over');
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        uploadStudyNoteFile(files[0]);
      }
    });
  });

  // Initial focus and sizing
  handleInputChange();
  messageInput?.focus();

  // Initialize theme toggle
  initTheme();

  // Load conversation history and student profile on page load
  loadConversations();
  loadStudentProfile();

})();
