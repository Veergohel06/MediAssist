content = open('static/modern_chat.css', 'r', encoding='utf-8').read()
checks = [
    '--bubble-heading-color',
    '--bubble-subheading-color',
    '--bubble-divider-color',
    '--bubble-accent-color',
    '--bubble-code-bg',
    '--bubble-code-color',
    'viva-session-banner',
    'mcq-stem',
    'word-break: break-word',
    'data-theme',
]
all_ok = True
for check in checks:
    found = check in content
    if not found:
        print('MISSING: ' + check)
        all_ok = False
if all_ok:
    print('All CSS checks passed.')
print('Total CSS lines: ' + str(content.count('\n')+1))
