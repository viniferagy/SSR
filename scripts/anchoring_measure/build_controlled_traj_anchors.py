#!/usr/bin/env python3
"""Build controlled-reference rows with staged trajectory-anchor skeletons."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}

TOKEN_RE = re.compile(r"\s+|https?://\S+|`[^`]*`|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)
CONTROL_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with", "as", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "from", "at",
    "into", "about", "we", "you", "i", "he", "she", "they", "them", "our", "your", "their", "not", "no", "yes",
    "do", "does", "did", "can", "could", "would", "should", "will", "may", "might", "must", "have", "has", "had",
    "there", "here", "which", "what", "when", "where", "why", "how", "also", "than", "therefore", "thus", "because",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def tokenize_words(value: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[^\W\s]", value.lower(), flags=re.UNICODE)


def function_word_skeleton(value: str) -> str:
    toks = tokenize_words(value)
    words = [tok for tok in toks if tok in CONTROL_STOPWORDS]
    if not words:
        words = toks[:80]
    chunks = [" ".join(words[i : i + 18]) for i in range(0, len(words), 18)]
    return "\n\n".join(chunks[:10])


def content_label(token: str, stage: int) -> str:
    if stage <= 1:
        return "<C>"

    length = len(token)
    if stage == 2:
        size = "S" if length <= 4 else "M" if length <= 8 else "L"
        return f"<C_{size}>"

    # Stage 3 keeps exact coarse shape without preserving letters.
    if token[:1].isupper() and token[1:].islower():
        shape = "Cap"
    elif token.isupper():
        shape = "UP"
    elif "-" in token:
        shape = "Hy"
    else:
        shape = "Lo"
    return f"<C_{shape}_{length}>"


def number_label(token: str, stage: int) -> str:
    if stage <= 1:
        return "<N>"
    digits = sum(ch.isdigit() for ch in token)
    if stage == 2:
        bucket = "1" if digits <= 1 else "2" if digits <= 2 else "3p"
        return f"<N_{bucket}>"
    return f"<N_{digits}>"


def traj_anchor_skeleton(answer: str, stage: int, compress_placeholders: bool = True) -> str:
    pieces: List[str] = []
    prev_label = ""
    for token in TOKEN_RE.findall(answer):
        if token.isspace():
            pieces.append(token)
            prev_label = ""
            continue
        if re.fullmatch(r"https?://\S+|`[^`]*`", token):
            label = "<X>" if stage <= 2 else f"<X_{min(len(token), 20)}>"
        elif re.fullmatch(r"\d+(?:[.,:/-]\d+)*", token):
            label = number_label(token, stage)
        elif re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", token):
            lower = token.lower()
            label = lower if lower in FUNCTION_WORDS else content_label(token, stage)
        elif re.fullmatch(r"[^\w\s]", token):
            label = token
        else:
            label = "<X>"

        # Compress adjacent identical placeholders, but keep punctuation and
        # function-word rhythm. This avoids a trace made only of repeated slots.
        if compress_placeholders and label.startswith("<") and label == prev_label:
            continue
        pieces.append(label)
        prev_label = label
    return "".join(pieces).strip()


def answer_tokens(answer: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for token in TOKEN_RE.findall(answer):
        if token.isspace():
            continue
        if re.fullmatch(r"https?://\S+|`[^`]*`", token):
            kind = "x"
        elif re.fullmatch(r"\d+(?:[.,:/-]\d+)*", token):
            kind = "n"
        elif re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", token):
            kind = "w"
        elif re.fullmatch(r"[^\w\s]", token):
            kind = "p"
        else:
            kind = "x"
        out.append((kind, token))
    return out


def word_case(token: str) -> str:
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return "none"
    joined = "".join(letters)
    if joined.isupper():
        return "upper"
    if joined.islower():
        return "lower"
    if joined[:1].isupper() and joined[1:].islower():
        return "title"
    return "mixed"


def word_shape(token: str, max_len: int = 32) -> str:
    chars: List[str] = []
    for ch in token[:max_len]:
        lo = ch.lower()
        if lo in "aeiou":
            chars.append("v" if ch.islower() else "V")
        elif ch.isalpha():
            chars.append("c" if ch.islower() else "C")
        elif ch.isdigit():
            chars.append("d")
        elif ch == "-":
            chars.append("-")
        elif ch == "'":
            chars.append("'")
        else:
            chars.append("x")
    if len(token) > max_len:
        chars.append("plus")
    return "".join(chars) or "none"


def case_code(token: str) -> str:
    return {
        "upper": "U",
        "lower": "L",
        "title": "T",
        "mixed": "M",
        "none": "0",
    }[word_case(token)]


def len_bucket(length: int) -> str:
    if length <= 2:
        return "xs"
    if length <= 4:
        return "s"
    if length <= 7:
        return "m"
    if length <= 11:
        return "l"
    return "xl"


def number_shape(token: str) -> str:
    chars: List[str] = []
    for ch in token[:32]:
        if ch.isdigit():
            chars.append("d")
        elif ch in ".,:/-":
            chars.append(ch)
        else:
            chars.append("x")
    if len(token) > 32:
        chars.append("plus")
    return "".join(chars) or "none"


def punct_name(token: str) -> str:
    names = {
        ".": "dot",
        ",": "comma",
        ":": "colon",
        ";": "semi",
        "?": "qmark",
        "!": "bang",
        "-": "hyphen",
        "—": "dash",
        "–": "dash",
        "(": "lparen",
        ")": "rparen",
        "[": "lbrack",
        "]": "rbrack",
        "{": "lbrace",
        "}": "rbrace",
        "\"": "quote",
        "'": "apost",
        "`": "tick",
        "*": "star",
        "#": "hash",
        "/": "slash",
        "\\": "backslash",
    }
    return names.get(token, "punct")


def compact_punct(token: str) -> str:
    names = {
        ".": ".",
        ",": ",",
        ":": ":",
        ";": ";",
        "?": "?",
        "!": "!",
        "-": "-",
        "—": "-",
        "–": "-",
        "(": "(",
        ")": ")",
        "[": "[",
        "]": "]",
        "{": "{",
        "}": "}",
        "\"": "\"",
        "'": "'",
        "`": "`",
        "*": "*",
        "#": "#",
        "/": "/",
        "\\": "\\",
    }
    return names.get(token, "P")


def token_signature(kind: str, token: str, dense: bool = False) -> str:
    if kind == "w":
        lower = token.lower()
        if lower in FUNCTION_WORDS:
            return f"F:{lower}"
        bits = [
            "W",
            f"case={word_case(token)}",
            f"len={len(token)}" if dense else f"len={len_bucket(len(token))}",
            f"shape={word_shape(token)}" if dense else f"shape={word_shape(token, max_len=12)}",
        ]
        if "-" in token:
            bits.append("hyphen=1")
        if "'" in token:
            bits.append("apost=1")
        return "|".join(bits)
    if kind == "n":
        digits = sum(ch.isdigit() for ch in token)
        return f"N|digits={digits}|shape={number_shape(token)}"
    if kind == "p":
        return f"P:{punct_name(token)}"
    return f"X|len={len_bucket(len(token))}"


def compact_token_signature(kind: str, token: str, include_shape: bool = True) -> str:
    if kind == "w":
        lower = token.lower()
        if lower in FUNCTION_WORDS:
            return f"f:{lower}"
        bits = ["w", case_code(token), str(len(token))]
        if include_shape:
            bits.append(word_shape(token, max_len=24))
        return ":".join(bits)
    if kind == "n":
        return f"n:{sum(ch.isdigit() for ch in token)}:{number_shape(token)}"
    if kind == "p":
        return f"p:{compact_punct(token)}"
    return f"x:{len_bucket(len(token))}"


def traj_anchor_feature_dense(answer: str) -> str:
    pieces = [compact_token_signature(kind, token, include_shape=True) for kind, token in answer_tokens(answer)]
    return " ".join(pieces).strip()


def traj_anchor_transition_dense(answer: str) -> str:
    toks = answer_tokens(answer)
    rows: List[str] = []
    for idx in range(max(0, len(toks) - 1)):
        kind_a, tok_a = toks[idx]
        kind_b, tok_b = toks[idx + 1]
        len_delta = len(tok_b) - len(tok_a)
        if len_delta > 3:
            len_rel = "++"
        elif len_delta > 0:
            len_rel = "+"
        elif len_delta == 0:
            len_rel = "="
        elif len_delta >= -3:
            len_rel = "-"
        else:
            len_rel = "--"
        kind_code = f"{kind_a}{kind_b}"
        case_pair = ""
        if kind_a == "w" or kind_b == "w":
            case_pair = case_code(tok_a) + case_code(tok_b)
        punct = ""
        if kind_a == "p":
            punct += compact_punct(tok_a)
        if kind_b == "p":
            punct += compact_punct(tok_b)
        rows.append(f"{kind_code}:{len_rel}:{case_pair}:{punct}".rstrip(":"))
    return " ".join(rows).strip()


def char_shape_char(ch: str) -> str:
    lo = ch.lower()
    if lo in "aeiou":
        return "v" if ch.islower() else "V"
    if ch.isalpha():
        return "c" if ch.islower() else "C"
    if ch.isdigit():
        return "d"
    if ch.isspace():
        return "_"
    return compact_punct(ch)


def traj_anchor_char_shape(answer: str) -> str:
    # Preserve function words exactly to make alignment mechanical, while
    # replacing content characters with non-lexical vowel/consonant classes.
    pieces: List[str] = []
    for token in TOKEN_RE.findall(answer):
        if token.isspace():
            pieces.append(token)
            continue
        if re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", token) and token.lower() in FUNCTION_WORDS:
            pieces.append(token.lower())
            continue
        pieces.append("".join(char_shape_char(ch) for ch in token))
    return "".join(pieces).strip()


def prefix_answer_tokens(answer: str, max_tokens: int) -> List[Tuple[str, str]]:
    return answer_tokens(answer)[:max(0, int(max_tokens))]


def prefix_answer_text(answer: str, max_tokens: int) -> str:
    pieces: List[str] = []
    seen = 0
    for token in TOKEN_RE.findall(answer):
        if token.isspace():
            if seen > 0 and seen < max_tokens:
                pieces.append(token)
            continue
        if seen >= max_tokens:
            break
        pieces.append(token)
        seen += 1
    return "".join(pieces).strip()


def bucket_token(kind: str, token: str) -> str:
    if kind == "w":
        lower = token.lower()
        if lower in FUNCTION_WORDS:
            return lower
        return f"<W_{case_code(token)}_{len_bucket(len(token))}>"
    if kind == "n":
        digits = sum(ch.isdigit() for ch in token)
        bucket = "1" if digits <= 1 else "2" if digits <= 2 else "3p"
        return f"<N_{bucket}>"
    if kind == "p":
        return compact_punct(token)
    return f"<X_{len_bucket(len(token))}>"


def shape_token(kind: str, token: str) -> str:
    if kind == "w":
        lower = token.lower()
        if lower in FUNCTION_WORDS:
            return lower
        return f"<S_{case_code(token)}_{word_shape(token, max_len=12)}>"
    if kind == "n":
        return f"<D_{number_shape(token)}>"
    if kind == "p":
        return compact_punct(token)
    return f"<X_{len_bucket(len(token))}>"


def mask_token(kind: str, token: str) -> str:
    if kind == "w":
        lower = token.lower()
        if lower in FUNCTION_WORDS:
            return lower
        return "<C>"
    if kind == "n":
        return "<N>"
    if kind == "p":
        return compact_punct(token)
    return "<X>"


def xlen_token(kind: str, token: str) -> str:
    if kind == "w":
        return "x" * min(len(token), 32)
    if kind == "n":
        return "d" * min(sum(ch.isdigit() for ch in token), 32)
    if kind == "p":
        return compact_punct(token)
    return "x" * min(len(token), 32)


def len_token(kind: str, token: str) -> str:
    if kind == "w":
        return str(len(token))
    if kind == "n":
        return str(sum(ch.isdigit() for ch in token))
    if kind == "p":
        return compact_punct(token)
    return str(len(token))


def shape_all_token(kind: str, token: str) -> str:
    if kind == "w":
        return word_shape(token, max_len=32)
    if kind == "n":
        return number_shape(token)
    if kind == "p":
        return compact_punct(token)
    return "x" * min(len(token), 32)


def len_shape_token(kind: str, token: str) -> str:
    if kind == "w":
        return f"{len(token)}{word_shape(token, max_len=32)}"
    if kind == "n":
        return f"{sum(ch.isdigit() for ch in token)}{number_shape(token)}"
    if kind == "p":
        return compact_punct(token)
    return f"{len(token)}x"


def initial_token(kind: str, token: str) -> str:
    if kind == "w":
        return token[:1]
    if kind == "n":
        return "<N>"
    if kind == "p":
        return compact_punct(token)
    return "<X>"


def content_term_set(value: str) -> set[str]:
    return {
        tok
        for tok in tokenize_words(value)
        if tok not in CONTROL_STOPWORDS and (len(tok) > 1 or not tok.isascii())
    }


def stable_hash_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def unique_terms_preserve_order(terms: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def answer_content_terms(answer: str, max_terms: int, order: str = "first") -> List[str]:
    terms: List[str] = []
    for kind, token in answer_tokens(answer):
        if kind == "w":
            lower = token.lower()
            if lower in FUNCTION_WORDS or (len(lower) <= 1 and lower.isascii()):
                continue
            terms.append(token)
        elif kind == "n":
            terms.append(token)
        elif kind == "x":
            clean = token.strip("`")
            if clean:
                terms.append(clean)
    terms = unique_terms_preserve_order(terms)
    if order == "first":
        ordered = terms
    elif order == "alpha":
        ordered = sorted(terms, key=lambda x: (x.lower(), x))
    elif order == "hash":
        ordered = sorted(terms, key=lambda x: stable_hash_key(x.lower()))
    else:
        raise ValueError(f"Unknown term order: {order}")
    return ordered[:max(0, int(max_terms))]


def answer_entity_number_terms(answer: str, max_terms: int) -> List[str]:
    terms: List[str] = []
    for kind, token in answer_tokens(answer):
        lower = token.lower()
        if kind == "n":
            terms.append(token)
        elif kind == "x":
            clean = token.strip("`")
            if clean and len(clean) <= 48 and not any(ch.isspace() for ch in clean):
                terms.append(clean)
        elif kind == "w" and lower not in FUNCTION_WORDS:
            has_upper = any(ch.isupper() for ch in token)
            has_digit = any(ch.isdigit() for ch in token)
            if has_upper or has_digit or "-" in token or "_" in token:
                terms.append(token)
    terms = unique_terms_preserve_order(terms)
    if len(terms) < max_terms:
        for term in answer_content_terms(answer, max_terms, order="first"):
            if term.lower() not in {x.lower() for x in terms}:
                terms.append(term)
            if len(terms) >= max_terms:
                break
    return terms[:max(0, int(max_terms))]


def render_unordered_terms(title: str, terms: Sequence[str]) -> str:
    if not terms:
        return f"{title}: none"
    lines = [f"{title}:"]
    for i in range(0, len(terms), 16):
        lines.append("- " + ", ".join(terms[i : i + 16]))
    return "\n".join(lines)


NEUTRAL_AUDIT_SENTENCES = [
    "Check the question scope before committing to a final response.",
    "Separate architecture details, input assumptions, and output interpretation.",
    "Track which claims are general advice and which claims are implementation-specific.",
    "Prefer a concise final explanation after the evidence has been organized.",
    "Verify that examples serve the explanation rather than replacing it.",
    "Keep the final response readable for someone applying the idea later.",
    "Avoid adding unrelated caveats once the main distinction is clear.",
    "Make sure the conclusion follows from the preceding observations.",
]


def word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[^\W\s]", value, flags=re.UNICODE))


def neutral_pad(min_words: int) -> str:
    if min_words <= 0:
        return ""
    lines = ["NEUTRAL AUDIT NOTES:"]
    count = 0
    idx = 0
    while count < min_words:
        sentence = NEUTRAL_AUDIT_SENTENCES[idx % len(NEUTRAL_AUDIT_SENTENCES)]
        lines.append(f"- {sentence}")
        count += word_count(sentence)
        idx += 1
    return "\n".join(lines)


def prefix_token_text(tokens: Sequence[Tuple[str, str]], max_tokens: int) -> str:
    pieces: List[str] = []
    for kind, token in tokens[: max(0, int(max_tokens))]:
        if kind == "p":
            pieces.append(compact_punct(token))
        else:
            pieces.append(token)
    return " ".join(pieces).strip()


def prob_anchor_body(base_reasoning: str, answer: str, mode: str, max_terms: int, pad_ratio: int = 0) -> str:
    if mode == "terms-tail":
        evidence = render_unordered_terms("UNORDERED ANSWER EVIDENCE TERMS", answer_content_terms(answer, max_terms, order="first"))
    elif mode == "alpha-tail":
        evidence = render_unordered_terms("ALPHABETIZED ANSWER EVIDENCE TERMS", answer_content_terms(answer, max_terms, order="alpha"))
    elif mode == "hash-tail":
        evidence = render_unordered_terms("HASH-ORDERED ANSWER EVIDENCE TERMS", answer_content_terms(answer, max_terms, order="hash"))
    elif mode == "entity-tail":
        evidence = render_unordered_terms("ENTITY AND NUMBER EVIDENCE TERMS", answer_entity_number_terms(answer, max_terms))
    elif mode == "prefix-tail":
        evidence = "BOUNDED ANSWER PREFIX EVIDENCE:\n" + prefix_token_text(answer_tokens(answer), max_terms)
    else:
        raise ValueError(f"Unknown prob-anchor mode: {mode}")

    parts = [base_reasoning.strip()] if base_reasoning.strip() else []
    if pad_ratio > 0:
        target_pre_words = max(word_count("\n\n".join(parts)), word_count(evidence) * pad_ratio)
        pad = neutral_pad(target_pre_words - word_count("\n\n".join(parts)))
        if pad:
            parts.append(pad)
    parts.append(evidence)
    return "\n\n".join(part for part in parts if part.strip()).strip()


def parse_prob_anchor_variant(spec: str) -> Tuple[str, int, int]:
    pieces = [piece.strip() for piece in spec.split(":") if piece.strip()]
    if len(pieces) < 2 or len(pieces) > 3:
        raise ValueError(f"Prob-anchor variant must be MODE:N[:PADRATIO], got: {spec}")
    mode = pieces[0]
    pad_ratio = 0
    if mode.endswith("pad"):
        mode = mode[: -len("pad")]
        if mode.endswith("-"):
            mode = mode[:-1]
        pad_ratio = 12
    if mode not in {"terms-tail", "alpha-tail", "hash-tail", "entity-tail", "prefix-tail"}:
        raise ValueError(f"Unknown prob-anchor mode: {mode}")
    if len(pieces) == 3:
        pad_ratio = int(pieces[2])
    return mode, int(pieces[1]), pad_ratio


def should_copy_content_token(token: str, mode: str, question_terms: set[str], content_index: int) -> bool:
    lower = token.lower()
    if mode == "qcopy":
        return lower in question_terms
    if mode == "qnumcopy":
        return lower in question_terms
    if mode == "shortcopy":
        return len(token) <= 4
    match = re.fullmatch(r"shortcopy(\d+)", mode)
    if match:
        return len(token) <= int(match.group(1))
    if mode == "qshortcopy":
        return lower in question_terms or len(token) <= 4
    match = re.fullmatch(r"qshortcopy(\d+)", mode)
    if match:
        return lower in question_terms or len(token) <= int(match.group(1))
    match = re.fullmatch(r"sparse(\d+)copy", mode)
    if match:
        return content_index % int(match.group(1)) == 0
    raise ValueError(f"Unknown copy-mask mode: {mode}")


def copy_mask_prefix_body(prefix: str, mode: str, question: str = "") -> str:
    question_terms = content_term_set(question)
    pieces: List[str] = []
    content_index = 0
    for token in TOKEN_RE.findall(prefix):
        if token.isspace():
            pieces.append(token)
            continue
        if re.fullmatch(r"https?://\S+|`[^`]*`", token):
            pieces.append("<X>")
        elif re.fullmatch(r"\d+(?:[.,:/-]\d+)*", token):
            pieces.append(token if mode == "qnumcopy" else "<N>")
        elif re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", token):
            lower = token.lower()
            if lower in FUNCTION_WORDS:
                pieces.append(lower)
            else:
                content_index += 1
                pieces.append(token if should_copy_content_token(token, mode, question_terms, content_index) else "<C>")
        elif re.fullmatch(r"[^\w\s]", token):
            pieces.append(compact_punct(token))
        else:
            pieces.append("<X>")
    return "".join(pieces).strip()


def render_rule_prefix_body(tokens: Sequence[Tuple[str, str]], mode: str) -> str:
    if mode == "mask":
        pieces = [mask_token(kind, token) for kind, token in tokens]
    elif mode == "bucket":
        pieces = [bucket_token(kind, token) for kind, token in tokens]
    elif mode == "shape":
        pieces = [shape_token(kind, token) for kind, token in tokens]
    else:
        raise ValueError(f"Unknown rule-prefix mode: {mode}")
    return " ".join(piece for piece in pieces if piece).strip()


def traj_anchor_rule_prefix(answer: str, mode: str, max_tokens: int) -> str:
    tokens = prefix_answer_tokens(answer, max_tokens)
    if mode == "mask":
        rule = (
            f"RULE: Transform the first {max_tokens} reference-answer tokens. "
            "Copy function words and punctuation; replace content words with <C>, numbers with <N>, other spans with <X>."
        )
    elif mode == "bucket":
        rule = (
            f"RULE: Transform the first {max_tokens} reference-answer tokens. "
            "Copy function words and punctuation; content words become <W_case_lengthbucket>; numbers become <N_digitbucket>."
        )
    elif mode == "shape":
        rule = (
            f"RULE: Transform the first {max_tokens} reference-answer tokens. "
            "Copy function words and punctuation; content words become <S_case_vowelconsonantshape>; numbers become <D_digitshape>."
        )
    else:
        raise ValueError(f"Unknown rule-prefix mode: {mode}")
    return f"{rule}\nBODY: {render_rule_prefix_body(tokens, mode)}".strip()


def render_natural_prefix_body(prefix: str, mode: str) -> str:
    pieces: List[str] = []
    for token in TOKEN_RE.findall(prefix):
        if token.isspace():
            pieces.append(token)
            continue
        if re.fullmatch(r"https?://\S+|`[^`]*`", token):
            kind = "x"
        elif re.fullmatch(r"\d+(?:[.,:/-]\d+)*", token):
            kind = "n"
        elif re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", token):
            kind = "w"
        elif re.fullmatch(r"[^\w\s]", token):
            kind = "p"
        else:
            kind = "x"

        if mode == "xlen":
            pieces.append(xlen_token(kind, token))
        elif mode == "len":
            pieces.append(len_token(kind, token))
        elif mode == "shapeall":
            pieces.append(shape_all_token(kind, token))
        elif mode == "lenshape":
            pieces.append(len_shape_token(kind, token))
        elif mode == "initialcopy":
            pieces.append(initial_token(kind, token))
        else:
            raise ValueError(f"Unknown compact natural-prefix mode: {mode}")
    return "".join(pieces).strip()


def traj_anchor_rule_natural_prefix(answer: str, mode: str, max_tokens: int, question: str = "") -> str:
    prefix = prefix_answer_text(answer, max_tokens)
    if mode == "mask":
        body = traj_anchor_skeleton(prefix, 1, compress_placeholders=True)
        rule = "RULE: Redact the reference-answer prefix: keep function words and punctuation; content words -> <C>."
    elif mode == "bucket":
        body = traj_anchor_skeleton(prefix, 2, compress_placeholders=True)
        rule = "RULE: Redact the reference-answer prefix: keep function words and punctuation; content words -> length buckets."
    elif mode == "shape":
        body = traj_anchor_skeleton(prefix, 3, compress_placeholders=True)
        rule = "RULE: Redact the reference-answer prefix: keep function words and punctuation; content words -> shape tags."
    elif mode == "mask_nocompress":
        body = traj_anchor_skeleton(prefix, 1, compress_placeholders=False)
        rule = "RULE: Redact the reference-answer prefix: keep function words and punctuation; content words -> <C>."
    elif mode == "xlen":
        body = render_natural_prefix_body(prefix, mode)
        rule = (
            "RULE: Transform the reference-answer prefix: replace every word with that many x characters, "
            "every number with that many d characters, and copy punctuation/spacing."
        )
    elif mode == "len":
        body = render_natural_prefix_body(prefix, mode)
        rule = (
            "RULE: Transform the reference-answer prefix: replace every word with its character count, "
            "every number with its digit count, and copy punctuation/spacing."
        )
    elif mode == "shapeall":
        body = render_natural_prefix_body(prefix, mode)
        rule = (
            "RULE: Transform the reference-answer prefix: replace every letter by v/V for vowels or c/C for consonants, "
            "digits by d, and copy punctuation/spacing."
        )
    elif mode == "lenshape":
        body = render_natural_prefix_body(prefix, mode)
        rule = (
            "RULE: Transform the reference-answer prefix: replace every word by length plus vowel/consonant shape, "
            "numbers by digit-count plus digit shape, and copy punctuation/spacing."
        )
    elif mode == "initialcopy":
        body = render_natural_prefix_body(prefix, mode)
        rule = (
            "RULE: Transform the reference-answer prefix: replace every alphabetic word with its first letter, "
            "replace each number with <N>, and copy punctuation/spacing."
        )
    elif mode == "prefixcopy":
        body = prefix
        rule = "RULE: Copy only this bounded reference-answer prefix verbatim."
    elif mode == "qcopy":
        body = copy_mask_prefix_body(prefix, mode, question)
        rule = "RULE: Redact the reference-answer prefix: keep function words, punctuation, and question-overlap words; other content words -> <C>."
    elif mode == "qnumcopy":
        body = copy_mask_prefix_body(prefix, mode, question)
        rule = "RULE: Redact the reference-answer prefix: keep function words, punctuation, numbers, and question-overlap words; other content words -> <C>."
    elif mode == "shortcopy":
        body = copy_mask_prefix_body(prefix, mode, question)
        rule = "RULE: Redact the reference-answer prefix: keep function words, punctuation, and content words of length <=4; other content words -> <C>."
    elif re.fullmatch(r"shortcopy\d+", mode):
        body = copy_mask_prefix_body(prefix, mode, question)
        threshold = re.fullmatch(r"shortcopy(\d+)", mode).group(1)
        rule = f"RULE: Redact the reference-answer prefix: keep function words, punctuation, and content words of length <={threshold}; other content words -> <C>."
    elif mode == "qshortcopy":
        body = copy_mask_prefix_body(prefix, mode, question)
        rule = "RULE: Redact the reference-answer prefix: keep function words, punctuation, question-overlap words, and content words of length <=4; other content words -> <C>."
    elif re.fullmatch(r"qshortcopy\d+", mode):
        body = copy_mask_prefix_body(prefix, mode, question)
        threshold = re.fullmatch(r"qshortcopy(\d+)", mode).group(1)
        rule = f"RULE: Redact the reference-answer prefix: keep function words, punctuation, question-overlap words, and content words of length <={threshold}; other content words -> <C>."
    elif re.fullmatch(r"sparse\d+copy", mode):
        body = copy_mask_prefix_body(prefix, mode, question)
        stride = re.fullmatch(r"sparse(\d+)copy", mode).group(1)
        rule = f"RULE: Redact the reference-answer prefix: keep function words and punctuation; copy every {stride}th content word; other content words -> <C>."
    else:
        raise ValueError(f"Unknown natural-prefix mode: {mode}")
    return f"{rule}\n{body}".strip()


def parse_rule_prefix_variant(spec: str) -> Tuple[str, int]:
    if ":" not in spec:
        raise ValueError(f"Rule-prefix variant must be MODE:NTOK, got: {spec}")
    mode, n_text = spec.split(":", 1)
    mode = mode.strip()
    if mode not in {"mask", "bucket", "shape"}:
        raise ValueError(f"Unknown rule-prefix mode: {mode}")
    return mode, int(n_text)


def parse_natural_prefix_variant(spec: str) -> Tuple[str, int]:
    if ":" not in spec:
        raise ValueError(f"Natural-prefix variant must be MODE:NTOK, got: {spec}")
    mode, n_text = spec.split(":", 1)
    mode = mode.strip()
    if not (
        mode in {
        "mask",
        "bucket",
        "shape",
        "mask_nocompress",
        "xlen",
        "len",
        "shapeall",
        "lenshape",
        "initialcopy",
        "prefixcopy",
        "qcopy",
        "qnumcopy",
        "shortcopy",
        "qshortcopy",
        "sparse4copy",
        "sparse8copy",
        }
        or re.fullmatch(r"shortcopy\d+", mode)
        or re.fullmatch(r"qshortcopy\d+", mode)
        or re.fullmatch(r"sparse\d+copy", mode)
    ):
        raise ValueError(f"Unknown natural-prefix mode: {mode}")
    return mode, int(n_text)


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base_methods = [m.strip() for m in args.keep_methods.split(",") if m.strip()]
    for idx, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and len(out) >= args.limit:
            break
        questions = row.get("questions", {})
        answers = row.get("answers", {})
        contexts = row.get("contexts", {})
        reasonings = row.get("reasonings", {})
        answer = text(answers.get(args.answer_method))
        if not answer.strip():
            continue
        question = text(questions.get(args.answer_method) or next(iter(questions.values()), ""))

        base_reasoning = text(reasonings.get(args.answer_method))
        methods: Dict[str, str] = {}
        if args.controlled_baselines:
            if base_reasoning.strip():
                methods["Blind CoT"] = base_reasoning
                methods["+Prob Anchor"] = f"{base_reasoning}\n\n{answer}"
                methods["+Entropy Anchor"] = function_word_skeleton(answer)
                methods["Response-as-CoT"] = answer
        else:
            for method in base_methods:
                reasoning = text(reasonings.get(method))
                if reasoning.strip():
                    methods[method] = reasoning
        for stage in args.stages:
            methods[f"+Traj Anchor v{stage}"] = traj_anchor_skeleton(answer, stage)
        for variant in args.dense_variants:
            if variant == "v1nc":
                methods["+Traj Anchor v1 no-compress"] = traj_anchor_skeleton(answer, 1, compress_placeholders=False)
            elif variant == "char_shape":
                methods["+Traj Anchor char-shape"] = traj_anchor_char_shape(answer)
            elif variant == "feature_dense":
                methods["+Traj Anchor feature-dense"] = traj_anchor_feature_dense(answer)
            elif variant == "transition_dense":
                methods["+Traj Anchor transition-dense"] = traj_anchor_transition_dense(answer)
            else:
                raise ValueError(f"Unknown dense variant: {variant}")
        for spec in args.rule_prefix_variants:
            mode, n_tokens = parse_rule_prefix_variant(spec)
            label = f"+Traj Rule-{mode}{n_tokens}"
            methods[label] = traj_anchor_rule_prefix(answer, mode, n_tokens)
        for spec in args.natural_prefix_variants:
            mode, n_tokens = parse_natural_prefix_variant(spec)
            safe_mode = mode.replace("_", "-")
            label = f"+Traj NatRule-{safe_mode}{n_tokens}"
            methods[label] = traj_anchor_rule_natural_prefix(answer, mode, n_tokens, question=question)
        for spec in args.prob_anchor_variants:
            mode, n_terms, pad_ratio = parse_prob_anchor_variant(spec)
            safe_mode = mode.replace("_", "-")
            suffix = f"{safe_mode}{n_terms}"
            if pad_ratio > 0:
                suffix = f"{suffix}-pad{pad_ratio}"
            methods[f"+Prob Anchor {suffix}"] = prob_anchor_body(base_reasoning, answer, mode, n_terms, pad_ratio)
        if not methods:
            continue

        out.append(
            {
                "id": row.get("id", idx),
                "questions": {m: question for m in methods},
                "answers": {m: answer for m in methods},
                "contexts": {m: text(contexts.get(args.answer_method) or next(iter(contexts.values()), "")) for m in methods},
                "reasonings": methods,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-methods", default="Blind CoT,+Prob Anchor,+Entropy Anchor,Response-as-CoT")
    parser.add_argument("--answer-method", default="Blind CoT")
    parser.add_argument("--controlled-baselines", action="store_true", help="Generate Blind/Prob/Entropy/Response controlled baselines from --answer-method.")
    parser.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument(
        "--dense-variants",
        nargs="*",
        default=[],
        choices=["v1nc", "char_shape", "feature_dense", "transition_dense"],
        help="Additional answer-derived trajectory anchors that increase non-lexical A->R density.",
    )
    parser.add_argument(
        "--rule-prefix-variants",
        nargs="*",
        default=[],
        help="Explicit-rule prefix anchors as MODE:NTOK, e.g. mask:64 bucket:128 shape:128.",
    )
    parser.add_argument(
        "--natural-prefix-variants",
        nargs="*",
        default=[],
        help="Explicit-rule natural redaction anchors as MODE:NTOK, e.g. mask:128 bucket:256.",
    )
    parser.add_argument(
        "--prob-anchor-variants",
        nargs="*",
        default=[],
        help="Evidence-style probability anchors as MODE:N[:PADRATIO], e.g. terms-tail:64 alpha-tail:96 entity-tail:48 terms-tail-pad:64.",
    )
    args = parser.parse_args()

    rows = build_rows(args)
    written = write_jsonl(args.output, rows)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
