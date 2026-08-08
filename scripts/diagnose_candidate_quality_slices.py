#!/usr/bin/env python3
"""Heuristic quality-slice diagnostics for generated reasoning methods.

The diagnostics are intentionally lightweight and dependency-free:

- language-match risk scan using Unicode script and small stopword lists;
- task-type stratification for per-record A_prob;
- simple-surface-task overreasoning scan from prompt/answer length and trace length.

These are not replacement metrics. They are risk screens for comparing methods
on the same sample order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FRIENDLY_LABELS = {
    "NEU": "NEU",
    "SUP": "SUP",
    "AUG-SUP": "AUG-SUP",
    "SSR": "SSR",
    "SSR-SCHEMA": "SSR-SCHEMA",
    "SSR-DENSE": "SSR-DENSE",
}

STOPWORDS = {
    "en": {
        "the",
        "and",
        "is",
        "are",
        "what",
        "why",
        "how",
        "can",
        "you",
        "to",
        "of",
        "in",
        "for",
        "that",
        "with",
        "as",
        "on",
        "this",
        "it",
        "be",
        "not",
        "should",
        "will",
        "from",
        "about",
        "if",
        "or",
        "by",
        "do",
        "does",
        "need",
        "task",
        "user",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "y",
        "o",
        "de",
        "que",
        "en",
        "para",
        "por",
        "con",
        "como",
        "puedes",
        "puede",
        "quiero",
        "necesito",
        "esto",
        "pero",
        "si",
        "del",
        "al",
        "me",
        "lo",
    },
    "pt": {
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "e",
        "de",
        "que",
        "para",
        "por",
        "com",
        "como",
        "voce",
        "você",
        "preciso",
        "isso",
        "mas",
        "não",
        "nao",
    },
    "fr": {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "et",
        "ou",
        "de",
        "du",
        "que",
        "en",
        "pour",
        "avec",
        "comment",
        "je",
        "tu",
        "vous",
        "est",
        "pas",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "oder",
        "ein",
        "eine",
        "ist",
        "ich",
        "du",
        "sie",
        "nicht",
        "mit",
        "für",
        "fur",
        "wie",
        "was",
        "warum",
        "zu",
    },
    "it": {
        "il",
        "lo",
        "la",
        "gli",
        "le",
        "un",
        "una",
        "e",
        "o",
        "di",
        "che",
        "in",
        "per",
        "con",
        "come",
        "questo",
        "ma",
        "non",
    },
    "pl": {
        "i",
        "w",
        "z",
        "że",
        "ze",
        "to",
        "na",
        "nie",
        "mi",
        "mnie",
        "dla",
        "jak",
        "czy",
        "możesz",
        "mozesz",
        "zrób",
        "zrob",
        "tylko",
        "będzie",
        "bedzie",
    },
    "nl": {
        "de",
        "het",
        "een",
        "en",
        "of",
        "van",
        "in",
        "voor",
        "met",
        "dat",
        "niet",
        "ik",
        "je",
    },
    "tr": {
        "ve",
        "bir",
        "bu",
        "şu",
        "su",
        "ile",
        "için",
        "icin",
        "de",
        "da",
        "ne",
        "nasıl",
        "nasil",
        "mi",
        "mı",
    },
}

DIACRITIC_HINTS = {
    "pl": set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"),
    "es": set("ñáéíóúü¿¡ÑÁÉÍÓÚÜ"),
    "pt": set("ãõâêôçáéíóúàÃÕÂÊÔÇÁÉÍÓÚÀ"),
    "fr": set("àâçéèêëîïôûùüÿœæÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ"),
    "de": set("äöüßÄÖÜẞ"),
    "tr": set("çğıöşüÇĞİÖŞÜ"),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_text(mapping: Dict[str, Any]) -> str:
    for value in mapping.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def compact(text: str, width: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= width else text[: width - 3].rstrip() + "..."


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]+", text.lower())


def script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        code = ord(ch)
        name = unicodedata.name(ch, "")
        if 0x4E00 <= code <= 0x9FFF:
            counts["zh"] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts["ja"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["ko"] += 1
        elif "CYRILLIC" in name:
            counts["cyrl"] += 1
        elif "ARABIC" in name:
            counts["arab"] += 1
        elif "HEBREW" in name:
            counts["hebr"] += 1
        elif "DEVANAGARI" in name:
            counts["deva"] += 1
        elif "THAI" in name:
            counts["thai"] += 1
        elif "GREEK" in name:
            counts["grek"] += 1
        elif "LATIN" in name or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def detect_language(text: str) -> Tuple[str, float, str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    counts = script_counts(text)
    total_letters = sum(counts.values())
    if total_letters == 0:
        return "unknown", 0.0, "no_letters"

    non_latin = {k: v for k, v in counts.items() if k != "latin"}
    if non_latin:
        script, count = max(non_latin.items(), key=lambda item: item[1])
        ratio = count / total_letters
        if count >= 8 or ratio >= 0.20:
            if script == "zh" and counts.get("ja", 0) >= 3:
                script = "ja"
            return script, ratio, "script"

    words = tokenize_words(text)
    if len(words) < 3:
        for lang, chars in DIACRITIC_HINTS.items():
            if any(ch in chars for ch in text):
                return lang, 0.55, "diacritic"
        return "unknown", 0.0, "too_short_latin"

    scores: Counter[str] = Counter()
    word_set = set(words)
    for lang, stops in STOPWORDS.items():
        scores[lang] = sum(1 for word in words if word in stops)
        scores[lang] += sum(2 for word in word_set if word in stops and len(word) > 2)
    for lang, chars in DIACRITIC_HINTS.items():
        if any(ch in chars for ch in text):
            scores[lang] += 4

    best, score = scores.most_common(1)[0]
    runner_up = scores.most_common(2)[1][1] if len(scores) > 1 else 0
    if score < 2 and len(words) < 8:
        return "unknown", 0.0, "weak_latin"
    if score < 3 and score - runner_up < 2:
        return "unknown", 0.0, "ambiguous_latin"
    confidence = min(1.0, score / max(6.0, len(words) * 0.25))
    return best, confidence, "stopwords"


def lang_family(lang: str) -> str:
    if lang in {"zh", "ja", "ko"}:
        return lang
    if lang in {"cyrl", "arab", "hebr", "deva", "thai", "grek"}:
        return lang
    return lang


def language_match(question: str, reasoning: str) -> Dict[str, Any]:
    q_lang, q_conf, q_basis = detect_language(question)
    r_lang, r_conf, r_basis = detect_language(reasoning[:1800])
    comparable = q_lang != "unknown" and r_lang != "unknown"
    match = comparable and lang_family(q_lang) == lang_family(r_lang)
    return {
        "question_lang": q_lang,
        "reasoning_lang": r_lang,
        "question_conf": q_conf,
        "reasoning_conf": r_conf,
        "question_basis": q_basis,
        "reasoning_basis": r_basis,
        "comparable": comparable,
        "match": match,
        "english_drift": comparable and q_lang != "en" and r_lang == "en",
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def is_code_like(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(r"```|\\b(json|python|javascript|typescript|react|redux|sql|html|css|api|function|class|bug|code|query|elasticsearch|jest|docker|aws|azure|regex|script)\\b", lower)
        or re.search(r"[{};<>]{3,}", text)
    )


def task_type(question: str, answer: str) -> str:
    q = question.lower()
    a = answer.lower()
    qa = q + "\n" + a[:1200]

    if re.search(r"\b(translate|translation|traduc|traduce|перев|翻译|catalan|català|рус|англ|english version|natural english)\b", q):
        return "translation_language"
    if re.search(
        r"\\begin\{|\\\(|\$\$|\b(matrix|calculate|compute|solve|equation|ratio|percent|probability|derivative|integral)\b"
        r"|mod\s+\d+|\b\d+\s*/\s*\d+\b|\b\d+(?:\.\d+)?\s*[+*/^=]\s*-?\d+(?:\.\d+)?\b"
        r"|\b\d+(?:\.\d+)?\s+-\s+\d+(?:\.\d+)?\b",
        qa,
    ):
        return "math_numeric"
    if is_code_like(qa) or re.search(
        r"\b(resnet|regnet|fine[- ]?tune|finetune|checkpoint|model weights?|neural|transformer|dataset|training|inference)\b",
        qa,
    ):
        return "code_data"
    if re.search(
        r"\b(calorie|weight loss|lose weight|body weight|\d+\s*kg|workout|prednisone|glucocorticoid|medical|symptom|dose|health|fitness|diet)\b",
        qa,
    ):
        return "health_fitness_medical"
    if re.search(r"\b(policy|regulation|government|political|governance|law|legal|china|chinese society|监管|政策|法律|治理)\b", qa):
        return "policy_law_analysis"
    if re.search(r"\b(write|draft|create|make|script|essay|business case|template|email|paragraph|story|poem|presentation|rewrite|rephrase)\b", q):
        return "writing_artifact"
    if re.search(r"\b(should i|recommend|suggest|best|option|advice|relationship|dating|career|choose|which one)\b", q):
        return "advice_decision"
    if re.search(r"\b(why|how|what is|what are|explain|tell me about)\b", q):
        return "factual_explanation"
    if word_count(question) <= 12 or len(question.strip()) <= 80:
        return "short_direct"
    return "other"


def is_simple_surface_task(question: str, answer: str) -> bool:
    q_words = word_count(question)
    a_words = word_count(answer)
    q = question.lower()
    direct_pattern = bool(
        re.search(
            r"\b(no empty line|empty line|translate|rename|назови|ponermelo|parrafos|catalan|make it|shorter|fix typo|format|only|just|ajaja)\b",
            q,
        )
    )
    short_request = q_words <= 12 or len(question.strip()) <= 80
    broad_question = bool(re.search(r"\bwhy|explain|tell me about|what is|how does\b", q)) and a_words > 160
    complex_surface = is_code_like(question) or bool(re.search(r"\\begin\{|matrix|equation|derive|prove", question.lower()))
    return (direct_pattern or (short_request and a_words <= 160)) and not broad_question and not complex_surface


def load_reasonings(inputs: List[Path]) -> Tuple[Dict[Tuple[int, str], str], Dict[Tuple[int, str], str], Dict[Tuple[int, str], str]]:
    traces: Dict[Tuple[int, str], str] = {}
    questions: Dict[Tuple[int, str], str] = {}
    answers: Dict[Tuple[int, str], str] = {}
    for path in inputs:
        rows = read_jsonl(path)
        for idx, row in enumerate(rows):
            for method, trace in row.get("reasonings", {}).items():
                if isinstance(trace, str) and trace.strip():
                    traces[(idx, method)] = trace
                    questions[(idx, method)] = row.get("questions", {}).get(method) or first_text(row.get("questions", {}))
                    answers[(idx, method)] = row.get("answers", {}).get(method) or first_text(row.get("answers", {}))
    return traces, questions, answers


def load_metrics(paths: List[Path]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    metrics: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx = int(row["sample_idx"])
                method = row["method"]
                metrics[(idx, method)] = row
    return metrics


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[int((len(values) - 1) * p)]


def pct(value: float) -> float:
    return round(value * 100.0, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", required=True, type=Path)
    parser.add_argument("--input-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--metrics-csv", action="append", required=True, type=Path)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_input)
    traces, questions_by_method, answers_by_method = load_reasonings(args.input_jsonl)
    metrics = load_metrics(args.metrics_csv)

    requested = [part.strip() for part in args.methods.split(",") if part.strip()]
    if requested:
        methods = requested
    else:
        methods = sorted({method for _, method in metrics})

    base_questions = [first_text(row.get("questions", {})) for row in base_rows]
    base_answers = [first_text(row.get("answers", {})) for row in base_rows]
    base_types = [task_type(q, a) for q, a in zip(base_questions, base_answers)]
    base_simple = [is_simple_surface_task(q, a) for q, a in zip(base_questions, base_answers)]

    # Language match.
    lang_rows: List[Dict[str, Any]] = []
    lang_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for method in methods:
        items = [(idx, traces[(idx, method)]) for idx in range(len(base_rows)) if (idx, method) in traces]
        n = len(items)
        comparable = matches = english_drift = non_en_q = unknown_q = unknown_r = 0
        q_lang_counts: Counter[str] = Counter()
        r_lang_counts: Counter[str] = Counter()
        for idx, trace in items:
            question = questions_by_method.get((idx, method)) or base_questions[idx]
            diag = language_match(question, trace)
            q_lang_counts[diag["question_lang"]] += 1
            r_lang_counts[diag["reasoning_lang"]] += 1
            unknown_q += int(diag["question_lang"] == "unknown")
            unknown_r += int(diag["reasoning_lang"] == "unknown")
            non_en_q += int(diag["question_lang"] not in {"unknown", "en"})
            comparable += int(diag["comparable"])
            matches += int(diag["match"])
            english_drift += int(diag["english_drift"])
            if diag["comparable"] and not diag["match"] and len(lang_examples[method]) < args.max_examples:
                lang_examples[method].append(
                    {
                        "sample_idx": idx,
                        "id": base_rows[idx].get("id", idx),
                        "question_lang": diag["question_lang"],
                        "reasoning_lang": diag["reasoning_lang"],
                        "question": compact(question),
                        "reasoning_start": compact(trace),
                    }
                )
        lang_rows.append(
            {
                "Method": method,
                "Label": FRIENDLY_LABELS.get(method, method),
                "N": n,
                "Comparable_N": comparable,
                "Match_Rate": round(matches / comparable, 6) if comparable else "",
                "Mismatch_Rate": round((comparable - matches) / comparable, 6) if comparable else "",
                "NonEnglish_Q_N": non_en_q,
                "English_Drift_N": english_drift,
                "English_Drift_Rate_Among_NonEnglish_Q": round(english_drift / non_en_q, 6) if non_en_q else "",
                "Unknown_Q_N": unknown_q,
                "Unknown_R_N": unknown_r,
                "Top_Q_Langs": json.dumps(q_lang_counts.most_common(6), ensure_ascii=False),
                "Top_R_Langs": json.dumps(r_lang_counts.most_common(6), ensure_ascii=False),
            }
        )

    write_csv(
        args.output_dir / "language_match_summary.csv",
        lang_rows,
        [
            "Method",
            "Label",
            "N",
            "Comparable_N",
            "Match_Rate",
            "Mismatch_Rate",
            "NonEnglish_Q_N",
            "English_Drift_N",
            "English_Drift_Rate_Among_NonEnglish_Q",
            "Unknown_Q_N",
            "Unknown_R_N",
            "Top_Q_Langs",
            "Top_R_Langs",
        ],
    )
    with (args.output_dir / "language_mismatch_examples.json").open("w", encoding="utf-8") as f:
        json.dump(lang_examples, f, ensure_ascii=False, indent=2)

    # Task-type A_prob stratification.
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for method in methods:
        for idx in range(len(base_rows)):
            row = metrics.get((idx, method))
            if not row:
                continue
            strata[(method, base_types[idx])].append(row)

    task_rows: List[Dict[str, Any]] = []
    for (method, typ), rows in sorted(strata.items(), key=lambda item: (FRIENDLY_LABELS.get(item[0][0], item[0][0]), item[0][1])):
        aprobs = [float(row["A_prob"]) * 100.0 for row in rows]
        b100 = [float(row["B_100"]) for row in rows]
        toks = [float(row["reasoning_tokens"]) for row in rows]
        alex = [float(row["A_lex_QF"]) * 100.0 for row in rows]
        task_rows.append(
            {
                "Method": method,
                "Label": FRIENDLY_LABELS.get(method, method),
                "Task_Type": typ,
                "N": len(rows),
                "A_prob_Mean": round(mean(aprobs), 6),
                "A_prob_Median": round(median(aprobs), 6),
                "A_prob_P90": round(percentile(aprobs, 0.90), 6),
                "B_100_Mean": round(mean(b100), 6),
                "A_lex_QF_Mean": round(mean(alex), 6),
                "Avg_R_tokens": round(mean(toks), 3),
            }
        )

    write_csv(
        args.output_dir / "task_type_aprob_summary.csv",
        task_rows,
        [
            "Method",
            "Label",
            "Task_Type",
            "N",
            "A_prob_Mean",
            "A_prob_Median",
            "A_prob_P90",
            "B_100_Mean",
            "A_lex_QF_Mean",
            "Avg_R_tokens",
        ],
    )

    # Simple-surface overreasoning.
    simple_rows: List[Dict[str, Any]] = []
    simple_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    simple_indices = [idx for idx, value in enumerate(base_simple) if value]
    for method in methods:
        vals: List[float] = []
        aprobs: List[float] = []
        for idx in simple_indices:
            row = metrics.get((idx, method))
            if not row:
                continue
            tok = float(row["reasoning_tokens"])
            vals.append(tok)
            aprobs.append(float(row["A_prob"]) * 100.0)
            if tok >= 300 and len(simple_examples[method]) < args.max_examples:
                trace = traces.get((idx, method), "")
                simple_examples[method].append(
                    {
                        "sample_idx": idx,
                        "id": base_rows[idx].get("id", idx),
                        "question": compact(base_questions[idx]),
                        "answer": compact(base_answers[idx]),
                        "reasoning_tokens": tok,
                        "A_prob": round(float(row["A_prob"]) * 100.0, 6),
                        "reasoning_start": compact(trace),
                    }
                )
        simple_rows.append(
            {
                "Method": method,
                "Label": FRIENDLY_LABELS.get(method, method),
                "Simple_N": len(vals),
                "Simple_Avg_R_tokens": round(mean(vals), 3),
                "Simple_Median_R_tokens": round(median(vals), 3),
                "Simple_P90_R_tokens": round(percentile(vals, 0.90), 3),
                "Over_250_Rate": round(sum(v >= 250 for v in vals) / len(vals), 6) if vals else "",
                "Over_300_Rate": round(sum(v >= 300 for v in vals) / len(vals), 6) if vals else "",
                "Over_400_Rate": round(sum(v >= 400 for v in vals) / len(vals), 6) if vals else "",
                "Simple_A_prob_Mean": round(mean(aprobs), 6),
            }
        )

    write_csv(
        args.output_dir / "simple_overreasoning_summary.csv",
        simple_rows,
        [
            "Method",
            "Label",
            "Simple_N",
            "Simple_Avg_R_tokens",
            "Simple_Median_R_tokens",
            "Simple_P90_R_tokens",
            "Over_250_Rate",
            "Over_300_Rate",
            "Over_400_Rate",
            "Simple_A_prob_Mean",
        ],
    )
    with (args.output_dir / "simple_overreasoning_examples.json").open("w", encoding="utf-8") as f:
        json.dump(simple_examples, f, ensure_ascii=False, indent=2)

    # Task inventory helps interpret heuristic buckets.
    inv_rows = []
    type_counts = Counter(base_types)
    simple_type_counts = Counter(base_types[idx] for idx in simple_indices)
    for typ, count in type_counts.most_common():
        inv_rows.append(
            {
                "Task_Type": typ,
                "N": count,
                "Simple_Surface_N": simple_type_counts.get(typ, 0),
                "Example_Questions": json.dumps(
                    [compact(base_questions[i], 140) for i, t in enumerate(base_types) if t == typ][:5],
                    ensure_ascii=False,
                ),
            }
        )
    write_csv(args.output_dir / "task_type_inventory.csv", inv_rows, ["Task_Type", "N", "Simple_Surface_N", "Example_Questions"])

    report_lines = [
        "# Candidate Quality Slice Diagnostics",
        "",
        "Heuristic diagnostics; language detection is dependency-free and should be read as a risk scan, not a formal metric.",
        "",
        "## Language Match",
        "",
        "| Method | Comparable N | Match | Non-English Q | English Drift |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in lang_rows:
        report_lines.append(
            f"| {row['Label']} | {row['Comparable_N']} | {row['Match_Rate']} | {row['NonEnglish_Q_N']} | {row['English_Drift_Rate_Among_NonEnglish_Q']} |"
        )
    report_lines += ["", "## Simple-Surface Overreasoning", "", "| Method | N | Avg tokens | >300 rate | Simple A_prob |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in simple_rows:
        report_lines.append(
            f"| {row['Label']} | {row['Simple_N']} | {row['Simple_Avg_R_tokens']} | {row['Over_300_Rate']} | {row['Simple_A_prob_Mean']} |"
        )
    report_lines += [
        "",
        "## Files",
        "",
        "- `language_match_summary.csv`",
        "- `language_mismatch_examples.json`",
        "- `task_type_aprob_summary.csv`",
        "- `task_type_inventory.csv`",
        "- `simple_overreasoning_summary.csv`",
        "- `simple_overreasoning_examples.json`",
    ]
    (args.output_dir / "diagnostics_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
