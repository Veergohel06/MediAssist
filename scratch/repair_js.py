import re

content = open('static/modern_chat.js', 'r', encoding='utf-8').read()

# The corruption: 'char-counter danger'; immediately followed by MCQ Card Renderer comment
# We need to restore the missing else-if branches, close handleInputChange,
# restore sidebar helpers and suggestion cards, then open MCQ renderer properly.

bad_marker = "char-counter danger';\n      // \u2500\u2500 Interactive MCQ Card Renderer"

good_new = """char-counter danger';
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

  // Interactive MCQ Card Renderer"""

if bad_marker in content:
    content = content.replace(bad_marker, good_new, 1)
    open('static/modern_chat.js', 'w', encoding='utf-8').write(content)
    print('Repaired successfully')
    opens = content.count('{')
    closes = content.count('}')
    print(f'Brace balance after fix: {opens - closes}')
else:
    print('Marker not found in file')
    # Show context around where it should be
    idx = content.find("char-counter danger")
    if idx >= 0:
        print(repr(content[idx:idx+150]))
