"""Paper-facing prompt templates for reverse-CoT generation.

The public method names intentionally match the paper. Historical prompt-search
variants are not part of the release surface.
"""

NEU_PROMPT = """You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags."""

SUP_PROMPT = NEU_PROMPT + """ **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section."""

AUGSUP_PROMPT = SUP_PROMPT + """ **PROHIBITION**: When outputting your Explanation, you are strictly forbidden from displaying ANY discernible signs that you have peeked at the Solution. **DO NOT** explicitly output any information of Solution section in the Explanation section. Should ANY form of Solution leakage occur, you will be severely punished by the Almighty Ruler."""

QA_SUP_PROMPT = SUP_PROMPT + """

Suppression must not compromise reasoning quality. Verify that each step follows from the question and prior steps, progressively narrows the possibilities, remains globally consistent, and covers every necessary part of the task."""

PG_SUP_PROMPT = SUP_PROMPT + """

Write for a student who sees only the question and explanation. Introduce relevant concepts and constraints, guide discovery through meaningful intermediate steps, address plausible misconceptions, and verify that no logical gap requires outside information."""


SSR_PROMPT = """You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided assistant turn is context; return the reasoning trace only.

After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.

### Tags

Use these tags only:

- [PLAN]: understand the request, constraints, and answer shape.
- [RETR]: bring in needed facts, context, or remembered information.
- [INFR]: infer, calculate, transform, or connect task details.
- [EVAL]: check a risky assumption, calculation, source, or consistency point.
- [SUMM]: organize the constraints, checks, and response plan.
- [BRCH]: compare plausible approaches before selecting one.
- [RFLX]: pause on an important judgment and name what keeps it disciplined.
- [BTRK]: revise a concrete earlier path after a check fails.

### Skeleton Rules

1. Put one numbered line per step inside `<skeleton>`.
2. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.
3. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.
4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.
5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.
6. Mark at least half the lines `[HIGH]`.
7. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.
8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.
9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.
10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.
11. Do not repeat the same tag or sentence frame three times in a row.
12. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.

### Reason Rules

1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.
3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.
4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs may integrate the conclusion direction when it follows from the trace.
5. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven or information-density-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, or uncertainty boundary keeps it honest.
7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.
8. When the assistant turn contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.
9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.
10. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
11. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.
13. Always write the final closing tag `</reason>` after the last reasoning paragraph.
14. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.
"""


SSR_SCHEMA_PROMPT = """You reconstruct a structured reasoning trace for a final assistant turn in a dialogue.

Return only one valid JSON object with the top-level key "steps". Do not use markdown, XML, numbering, comments, examples, or text outside JSON. Do not output the final assistant answer.

Each item in "steps" must have exactly these fields:
- "tag": one of "PLAN", "RETR", "INFR", "EVAL", "SUMM", "BRCH", "RFLX", "BTRK".
- "importance": "HIGH" or "LOW".
- "action": one short task-specific sentence describing the reasoning operation.
- "reason": one compact paragraph explaining that operation.

Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks. Mark at least half the steps HIGH.

Keep the trace moderately developed and task-specific. Align it with the provided final assistant turn while avoiding direct copying of distinctive answer wording or answer-only phrases. Add diagnostic or reflection steps only for real ambiguity, competing approaches, missing information, verification risk, failed paths, or central judgments. Put caveats, criteria, checks, and tradeoffs in the reason field. For BRCH, EVAL, RFLX, and BTRK, name the actual uncertainty, tradeoff, evidence standard, or correction.

Before returning JSON, silently check that every reason paragraph maps to one action, the steps remain in reasoning order, and no field contains XML tags, markdown headings, placeholders, or prompt text.
"""


SSR_DENSE_PROMPT = SSR_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought.",
    "The trace should read like inner task reasoning rather than a compressed summary: more informative than a terse outline while remaining focused.",
).replace(
    "1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.",
    "1. Use the skeleton as a strict paragraph plan: write exactly one independent paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.",
).replace(
    "2. Each paragraph should be compact but informative.",
    "2. Each paragraph should develop its corresponding step from a local uncertainty, criterion, evidence need, or constraint toward the resulting choice or next move. Each paragraph should be compact but informative.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Attach checks to concrete task objects such as facts, calculations, code paths, recommendations, boundaries, or artifacts.",
)


SS_GEN_PROMPT = """You are an expert AI assistant converting an existing reasoning trace into an SSR structural skeleton.

Analyze the provided INPUT, segment it into faithful reasoning operations, and output only a concise SSR skeleton. Preserve the original logical order. Use one numbered line per operation in exactly this format: `n. [TAG][HIGH/LOW] <short action sentence>`.

Use only PLAN, RETR, INFR, EVAL, SUMM, BRCH, RFLX, and BTRK. Mark objectively hard, risky, or information-dense operations HIGH and routine operations LOW. Prefer 6-12 lines when supported by the input. Keep every line short, task-specific, and operation-level. Do not invent, correct, or reinterpret content. Output only skeleton lines in {lang}.

INPUT:
{input_text}
"""
