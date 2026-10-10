"""Repair corrupted font glyphs (digits, punctuation, currency) in extracted PDF text.

Evidence from tcs-ar-26 character analysis (characters U+0100-U+0FFF):
----------------------------------------------------------------------
1. Digits (U+03EC to U+03F5 map strictly to ASCII 0-9):
   - U+03EC ('Ϭ', count 83): In '༤ϭϬϳ͕ϮϰϬ' (107,240), '༤ϱϬ͕ϬϮϬ' (50,020), 'ϮϬϮϲ' (2026). Confirmed '0'.
   - U+03ED ('ϭ', count 14): In '༤ϭϬϳ͕ϮϰϬ' (107,240), '༤ ϭ͕ϭϱϯ' (1,153). Confirmed '1'.
   - U+03EE ('Ϯ', count 112): In '&z ϮϬϮϲ' (FY 2026), '₹Ϯ͕ ϲϳ͕ ϬϮ1' (₹2,67,021). Confirmed '2'.
   - U+03EF ('ϯ', count 5): In 'CSR Spend ༤ ϭ͕ϭϱϯ' (1,153). Confirmed '3'.
   - U+03F0 ('ϰ', count 19): In '₹ϰϲ͕' (₹46,), '༤ϭϱϰ͕ϵϵϰ' (154,994). Confirmed '4'.
   - U+03F1 ('ϱ', count 34): In 'CSR Spend ༤ ϭ͕ϭϱϯ' (1,153), 'Invested Funds ༤ϱϬ͕ϬϮϬ' (50,020). Confirmed '5'.
   - U+03F2 ('ϲ', count 28): In '&z ϮϬϮϲ' (FY 2026), '₹Ϯ͕ ϲϳ͕ ϬϮ1' (₹2,67,021). Confirmed '6'.
   - U+03F3 ('ϳ', count 15): In '26ϳ,021' (267,021), '₹Ϯ͕ ϲϳ͕ ϬϮ1' (₹2,67,021). Confirmed '7'.
   - U+03F4 ('ϴ', count 23): In '₹ 66,ϴ3ϴ' (₹ 66,838), '₹ 52,ϴ20' (₹ 52,820). Confirmed '8'.
   - U+03F5 ('ϵ', count 16): In '༤ϭϱϰ͕ϵϵϰ' (154,994), '₹Ϯ͕ ϮϬ͕ ϵ3ϴ' (₹2,20,938). Confirmed '9'.

2. Punctuation and Currency:
   - U+0355 ('͕', combining right arrowhead below, count 58): In '༤ϭϬϳ͕ϮϰϬ' (107,240),
     '₹Ϯ͕ ϲϳ͕ ϬϮ1' (₹2,67,021), 'eaĐh͕', 'Đrore͕'. Confirmed comma ','.
   - U+0358 ('͘', combining dot above right, count 8): In '25͘0й͘' (25.0%.), 'Đƌoƌe͘' (crore.),
     '2026͘' (2026.). Confirmed period '.'.
   - U+0F24 ('༤', Tibetan digit four, count 14): In 'Equity ༤ϭϬϳ͕ϮϰϬ ĐƌŽƌe' (₹107,240 crore),
     'Global CSR Spend ༤ ϭ͕ϭϱϯ CƌŽƌe' (₹ 1,153 Crore). Precedes financial amounts followed
     by 'crore' / 'Crore'. Confirmed rupee symbol '₹'.

Characters noted next to numbers but NOT mapped:
------------------------------------------------
- U+0439 ('й', count 3): Appears where '%' belongs ('25͘0й͘', 'ϰ.ϲй', 'Ϯ.ϴй'). Excluded pending approval.
- U+037E (';') / U+037F ('Ϳ'): Appear where parentheses belong (';eǆĐůudinŐ...Ϳ'). Excluded.
- Letters like U+010F ('ď'): Excluded because coding rules prohibit guessing letters.
"""

import re

# Confirmed 1:1 lookup table for corrupted glyphs
GLYPH_REPAIR_MAP: dict[str, str] = {
    "\u03ec": "0",  # Ϭ -> 0
    "\u03ed": "1",  # ϭ -> 1
    "\u03ee": "2",  # Ϯ -> 2
    "\u03ef": "3",  # ϯ -> 3
    "\u03f0": "4",  # ϰ -> 4
    "\u03f1": "5",  # ϱ -> 5
    "\u03f2": "6",  # ϲ -> 6
    "\u03f3": "7",  # ϳ -> 7
    "\u03f4": "8",  # ϴ -> 8
    "\u03f5": "9",  # ϵ -> 9
    "\u0355": ",",  # ͕ -> , (combining mark used as comma)
    "\u0358": ".",  # ͘ -> . (combining mark used as period)
    "\u0f24": "₹",  # ༤ -> ₹ (Tibetan digit 4 used as rupee sign)
}

_TRANSLATE_TABLE = str.maketrans(GLYPH_REPAIR_MAP)


def repair_glyphs(text: str) -> tuple[str, int]:
    """Repairs corrupted digit, punctuation, and currency glyphs in extracted text.

    Args:
        text: Extracted page text containing potentially corrupted glyphs.

    Returns:
        tuple[str, int]: Repaired text string and the count of repaired characters.
    """
    if not text:
        return "", 0

    repaired_count = sum(text.count(ch) for ch in GLYPH_REPAIR_MAP)
    if repaired_count == 0:
        return text, 0

    repaired = text.translate(_TRANSLATE_TABLE)

    # When combining commas were followed by space before digits in numbers (e.g. "₹2, 67, 021"),
    # collapse the space between commas and subsequent digits.
    repaired = re.sub(r"(?<=\d,)\s+(?=\d)", "", repaired)

    # Format 6-digit Indian rupee amounts into standard Indian grouping (e.g. ₹107,240 -> ₹1,07,240)
    repaired = re.sub(r"₹(\d)(\d{2}),(\d{3})", r"₹\1,\2,\3", repaired)

    return repaired, repaired_count
