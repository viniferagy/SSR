# -*- coding: utf-8 -*-
"""
Tier 1 baseline 全部 prompt 模板。
设计原则：
1. 数据为 LMArena 英文 query，prompt 用英文与论文设定对齐；
2. 所有方法共用同一种 (Q, A) 注入格式，保证可比性；
3. 每个 pipeline 定义显式输出标记 + 对应 parser，可比性体现在被测 trace 文本，
   而非外层包装。
"""

# ---------------------------------------------------------------------------
# 通用 (Q, A) 注入格式 —— 所有看得到 A 的方法统一用这个 user turn
# ---------------------------------------------------------------------------

USER_QA = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}"""

USER_Q_ONLY = """[QUESTION]
{question}"""

# ---------------------------------------------------------------------------
# NEU（论文 Appendix I 复刻，B11 / B13 / B12-with 共用）
# 注意：B11 用它且模型对 blocking 完全不知情，是机制判别实验的前提。
# ---------------------------------------------------------------------------

NEU_SYSTEM = (
    "Your role as an assistant involves providing precise and accurate solutions "
    "before providing detailed explanations with your full work showing your "
    "systematic thinking process leading to each solution. Your explanations should "
    "show how you engaged in a comprehensive cycle of analysis, summarizing, "
    "exploration, reassessment, reflection, backtracing, and iteration to develop "
    "well-considered thinking process. Please structure your response into two main "
    "sections: Solution and Explanation. In the Solution section, present your "
    "well-thought solution that accurately answers the question. The solution should "
    "remain a logical, accurate, concise expression style and detail necessary step "
    "needed to reach the conclusion, formatted as follows: <|begin_of_solution|> "
    "{final formatted, precise, and clear solution} <|end_of_solution|>. In the "
    "Explanation section, comprehensively detail your reasoning process from the "
    "question to your solutions using the specified format: <|begin_of_explanation|> "
    "{explanation with steps separated with '\\n\\n'} <|end_of_explanation|> Each "
    "step should show logical connections and detailed considerations leading to "
    "your solutions such as analyzing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of the current steps, refining "
    "any errors, and revisiting previous steps."
)

# B12 的 "without" 上下文：指令逐字相同，仅缺 REFERENCE ANSWER 块。
# 共享生成 token 流必须在两个上下文下都 in-distribution，所以除 A 外不许有任何措辞差异。
NEU_USER_WITH = USER_QA
NEU_USER_WITHOUT = USER_Q_ONLY


def parse_neu(text: str) -> str:
    """从 NEU 风格输出抽 explanation 作为被测 trace R。"""
    s, e = "<|begin_of_explanation|>", "<|end_of_explanation|>"
    if s in text:
        seg = text.split(s, 1)[1]
        return seg.split(e, 1)[0].strip() if e in seg else seg.strip()
    return text.strip()  # 兜底：标记缺失时全文入测并打 flag


# ---------------------------------------------------------------------------
# A1. Forward-draft + Bridge (FDB)
# ---------------------------------------------------------------------------

FDB_P1_SYSTEM = (
    "You are an expert problem solver. Solve the user's question entirely from "
    "scratch.\n"
    "First, write your complete step-by-step reasoning between <reasoning> and "
    "</reasoning>. Show genuine exploration: analyze the question, consider "
    "alternatives, verify intermediate conclusions, and revise when needed. Do not "
    "state a final answer inside <reasoning> before it has actually been derived.\n"
    "Then write your final answer between <answer> and </answer>."
)
# user turn: USER_Q_ONLY（绝对不给 A）

FDB_P2_SYSTEM_IMPLICIT = (
    "You previously wrote the draft reasoning below for the question. A reference "
    "answer is also provided.\n"
    "Task: write a SHORT closing segment that CONTINUES the draft reasoning so that "
    "the overall chain naturally arrives at a conclusion consistent with the "
    "reference answer.\n"
    "Hard constraints:\n"
    "1. Do not rewrite or repeat the draft; output only the new closing segment.\n"
    "2. Never mention or imply that a reference answer exists; write the segment as "
    "your own final verification and, if needed, self-correction.\n"
    "3. If the draft's conclusion already matches, consolidate it concisely; if it "
    "conflicts, briefly identify the flaw in the draft and correct course.\n"
    "4. Keep the segment under {bridge_budget} tokens.\n"
    "Output the closing segment between <bridge> and </bridge>."
)

FDB_P2_SYSTEM_EXPLICIT = (
    "You previously wrote the draft reasoning below for the question. A reference "
    "answer is also provided.\n"
    "Task: write a SHORT closing segment that CONTINUES the draft reasoning and "
    "reconciles it with the reference answer. You may explicitly refer to the "
    "reference answer (e.g., 'the reference answer indicates ...').\n"
    "Hard constraints:\n"
    "1. Do not rewrite or repeat the draft; output only the new closing segment.\n"
    "2. If the draft conflicts with the reference answer, identify the flaw and "
    "correct course.\n"
    "3. Keep the segment under {bridge_budget} tokens.\n"
    "Output the closing segment between <bridge> and </bridge>."
)

FDB_P2_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[DRAFT REASONING]
{draft}"""


def parse_tagged(text: str, tag: str) -> str:
    s, e = f"<{tag}>", f"</{tag}>"
    if s in text:
        seg = text.split(s, 1)[1]
        return seg.split(e, 1)[0].strip() if e in seg else seg.strip()
    return text.strip()


# ---------------------------------------------------------------------------
# A6. Gist-first（语义要点两阶段）
# ---------------------------------------------------------------------------

GIST_P1_SYSTEM_INVARIANT = (
    "Given a question and its reference answer, extract the METHODOLOGICAL ESSENCE "
    "needed to solve it.\n"
    "Output 3-8 numbered content gists. Each gist is one sentence of at most 20 "
    "words describing what to establish, analyze, derive, or compare — the "
    "substantive direction of that part of the solution.\n"
    "Hard constraints:\n"
    "- Do NOT use functional tags such as [PLAN] or [INFER]; write natural content "
    "sentences.\n"
    "- Content invariance: never state concrete results, specific values, named "
    "final conclusions, or verbatim phrases from the reference answer. Say 'compare "
    "the failure modes of the two mechanisms', not 'conclude that X fails because "
    "...'.\n"
    "- Keep the total under {gist_budget} tokens.\n"
    "Output the numbered list between <gist> and </gist>."
)

# 更弱对照：去掉 invariance 子句，其余逐字相同
GIST_P1_SYSTEM_FREE = (
    "Given a question and its reference answer, extract the METHODOLOGICAL ESSENCE "
    "needed to solve it.\n"
    "Output 3-8 numbered content gists. Each gist is one sentence of at most 20 "
    "words describing what to establish, analyze, derive, or compare — the "
    "substantive direction of that part of the solution.\n"
    "Hard constraints:\n"
    "- Do NOT use functional tags such as [PLAN] or [INFER]; write natural content "
    "sentences.\n"
    "- Keep the total under {gist_budget} tokens.\n"
    "Output the numbered list between <gist> and </gist>."
)

GIST_P2_SYSTEM = (
    "Write the complete reasoning chain for the question, developing each gist "
    "below in order into full reasoning.\n"
    "Requirements:\n"
    "- The chain must read as a self-contained derivation from the question alone.\n"
    "- Do not number the steps or cite the gists explicitly.\n"
    "- Do not copy text from the reference answer.\n"
    "- Separate reasoning steps with blank lines ('\\n\\n').\n"
    "Output the reasoning between <reasoning> and </reasoning>."
)

GIST_P2_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[GISTS]
{gists}"""

# ---------------------------------------------------------------------------
# Judges（B13 质量门控 / 通用终点一致性）—— 235B 或任意 judge 模型
# ---------------------------------------------------------------------------

QUALITY_GATE_SYSTEM = (
    "You are a strict evaluator of reasoning traces. Given a question, a reference "
    "answer, and a candidate reasoning trace, rate the trace.\n"
    "Criteria: (1) logical coherence and absence of nonsense; (2) relevance to the "
    "question; (3) whether the trace's endpoint conclusion is semantically "
    "consistent with the reference answer.\n"
    "Output ONLY a JSON object: {\"score\": <1-5 integer>, "
    "\"consistent\": <true|false>, \"reason\": \"<one sentence>\"}"
)

QUALITY_GATE_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[CANDIDATE TRACE]
{trace}"""

ENDPOINT_JUDGE_SYSTEM = (
    "Judge whether the conclusion reached at the END of the reasoning trace is "
    "semantically consistent with the reference answer's core claims. Minor wording "
    "differences are fine; contradictions or missing core claims are not.\n"
    "Output ONLY a JSON object: {\"consistent\": <true|false>, "
    "\"reason\": \"<one sentence>\"}"
)

ENDPOINT_JUDGE_USER = QUALITY_GATE_USER
