#!/usr/bin/env python3
"""Replace ASCII straight double quotes "" with Chinese curly quotes "". 
Used in LaTeX with ctex for proper CJK typography."""

import sys

CHINESE_LEFT = "\u201c"   # "
CHINESE_RIGHT = "\u201d"  # "

def fix_quotes(content: str) -> str:
    """Replace paired "" with "" in non-LaTeX-command contexts."""
    result = []
    in_quote = False
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == '"':
            # Check if this is inside a LaTeX command (\foo{..."} or \foo"...)
            # Skip if preceded by \
            if i > 0 and content[i-1] == '\\':
                result.append(ch)
                i += 1
                continue
            # Toggle
            result.append(CHINESE_LEFT if not in_quote else CHINESE_RIGHT)
            in_quote = not in_quote
        else:
            result.append(ch)
        i += 1
    return ''.join(result)

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'PAPER.tex'
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    
    new_content = fix_quotes(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Count
    old_count = content.count('"')
    new_left = new_content.count(CHINESE_LEFT)
    new_right = new_content.count(CHINESE_RIGHT)
    print(f'Replaced {old_count} straight quotes → {new_left} left + {new_right} right Chinese quotes')
    if new_left != new_right:
        print(f'WARNING: unbalanced! {new_left} " vs {new_right} "')
