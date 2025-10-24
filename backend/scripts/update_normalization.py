"""
Update the normalization function in performance_fixes.py
"""

import re

# Read the file
with open('backend/scripts/performance_fixes.py', 'r') as f:
    content = f.read()

# Find and replace the _normalize_text method
old_pattern = r'@staticmethod\s+def _normalize_text\(text: str\) -> str:.*?return text'

new_method = '''@staticmethod
    def _normalize_text(text: str) -> str:
        """
        Improved normalization for better fuzzy matching.
        
        Handles:
        - Parenthetical content (chemical names, descriptions)
        - Expiration date tags (#EXP...)
        - Special characters and punctuation
        - Multiple spaces
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove parentheses and their contents (e.g., chemical names)
        text = re.sub(r'\\([^)]*\\)', '', text)
        text = re.sub(r'\\[[^\\]]*\\]', '', text)
        text = re.sub(r'\\{[^\\}]*\\}', '', text)
        
        # Remove expiration date tags and similar metadata
        text = re.sub(r'#exp[0-9-]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'exp[0-9-]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'#[a-z0-9-]+', '', text, flags=re.IGNORECASE)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\\w\\s]', '', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text'''

# Replace using regex
content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)

# Also update threshold from 0.8 to 0.7
content = content.replace('if best_score > 0.8', 'if best_score > 0.7')
content = content.replace('# If we found a good match (>80% similarity)', '# If we found a good match (>70% similarity)')

# Write back
with open('backend/scripts/performance_fixes.py', 'w') as f:
    f.write(content)

print('✅ Updated performance_fixes.py with improved normalization and 0.7 threshold')

# Test the new normalization
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'#exp[0-9-]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'exp[0-9-]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#[a-z0-9-]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test with user's example
excel_product = 'Salonpas Plaster (methyl salicylate, menthol, and camphor) 20s'
import_product = 'SALONPAS PLASTER 20s #EXP2024-11-01'

excel_norm = normalize_text(excel_product)
import_norm = normalize_text(import_product)

print(f'\n🧪 TESTING WITH IMPROVED NORMALIZATION:')
print(f'Excel: "{excel_product}"')
print(f'  -> Normalized: "{excel_norm}"')
print(f'Import: "{import_product}"')
print(f'  -> Normalized: "{import_norm}"')

import difflib
similarity = difflib.SequenceMatcher(None, excel_norm, import_norm).ratio()
print(f'\nSimilarity: {similarity:.2f}')
print(f'Threshold: 0.70')

if similarity > 0.7:
    print('✅ RESULT: Would AUTO-EXPAND!')
else:
    print(f'❌ RESULT: Would still skip ({similarity:.2f} < 0.7)')
