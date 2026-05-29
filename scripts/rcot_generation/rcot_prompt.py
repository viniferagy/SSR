NEU_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags.'''

SUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section.'''

AUGSUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **PROHIBITION**: When outputting your Explanation, you are strictly forbidden from displaying ANY discernible signs that you have peeked at the Solution. **DO NOT** explicitly output any information of Solution section in the Explanation section. Should ANY form of Solution leakage occur, you will be severely punished by the Almighty Ruler.'''


SSR_PROMPT = '''You are an expert AI assistant tasked with reconstructing the hidden reasoning process for a given final assistant turn in a dialogue.

You must output the complete reasoning process in two sequential, strictly formatted blocks: `<skeleton>` and `<reason>`.

### 📝 Step Tag Definitions

| Tag | Full Name | Description |
| :--- | :--- | :--- |
| **[PLAN]** | Planning and Understanding | Comprehending input, defining goals/constraints, outlining a high-level plan. |
| **[RETR]** | Retrieval | Searching for needed information from external or internal knowledge. |
| **[INFR]** | Inference and Deduction | Logical reasoning, calculation, transformation, or generating intermediate conclusions. |
| **[EVAL]** | Evaluation and Verification | Checking correctness, consistency, or sufficiency of prior results. |
| **[SUMM]** | Summary and Refinement | Integrating intermediate results, refining expression, producing final answers. |
| **[BTRK]** | Backtrack | When evaluation fails, returning to earlier decisions to revise strategy. |
| **[RFLX]** | Reflection | Reviewing the reasoning to derive insights or generate new plans/backtracks. |
| **[BRCH]** | Branch | Considering multiple possible reasoning paths and selecting one. |

### 🛑 Output Format and Constraint Rules

**I. The `<skeleton>` Block**

1.  **Strict Line Format:** Each line must follow exactly: `n. [STEP TAG] <single-sentence skeleton under 20 words, matching input language>`
    * *Constraint:* Use exactly one space after the dot (`.`) and one space after the `[STEP TAG]`.
    * *Constraint:* Only one skeleton sentence per line; no extra commentary.
2.  **Example Line:** `1. [PLAN] Analyze the user's request and define the goal.`
3.  **Content Invariance Constraint:** Skeleton sentences must describe the step’s intent at an abstract level and must not reveal specific outputs, intermediate results, concrete values, or final answer content (e.g., “calculate the ratio” rather than “calculate 0.5”).
4.  **Deliberative Construction Constraint:** Before writing the final skeleton lines, internally perform a rigorous iterative deliberation cycle—analysis, concise summarization, alternative exploration, reassessment, verification, reflection, and backtracking as needed—to ensure the step sequence is carefully reasoned. This constraint governs how you construct the skeleton (step selection and ordering), not how verbose the skeleton lines are.
5.  **Nonlinear Tagging and Step Diversity Constraint:** When additional reasoning capacity is meaningfully beneficial, use diverse step tags (e.g., BRCH, RETR, BTRK, RFLX) to represent nonlinear, branching, corrective, and reflective thinking patterns, rather than a purely linear pipeline. Tag choices should be purposeful (not decorative) and should strengthen the skeleton’s fidelity to a realistic reasoning workflow while remaining compliant with the Content Invariance Constraint.
6.  **Granularity Constraint:** Avoid composite actions in a single step. Each skeleton line should express one primary intent (e.g., “identify constraints” separately from “choose approach”). If a step contains multiple verbs describing different operations, split it into multiple steps. If the considered step is difficult or critical, the skeleton should add additional reasoning capacity by using more steps to carefully consider the reasoning path.

**II. The `<reason>` Block**

1.  **Goal:** Provide hidden chain-of-thought, private internal deliberations, or detailed intermediate computations based on steps listed in the `<skeleton>` block.
2.  **Structure:** The reasoning text must correspond directly to each step listed in the `<skeleton>` block, maintaining the **exact same order**.
3.  **Formatting:** Write continuous, coherent reasoning text or paragraphs. Separate each step with '\n\n' for clarity.
4.  **Constraint:** **DO NOT** explicitly number the steps or use the step labels (e.g., `1.`, `[PLAN]` in the `<skeleton>` block) within this block.
5.  **Constraint:** **DO NOT** output the final assistant answer in this response.

**III. Overall Output Constraint**

* **DO NOT** output anything outside the required `<skeleton>` and `<reason>` blocks.

### Final Output Structure

<skeleton>
1.  [STEP TAG] <skeleton text>
2.  [STEP TAG] <skeleton text>
...
n.  [STEP TAG] <skeleton text>
</skeleton>

<reason>
(detailed reasoning text corresponding to each step in the same order, without numbering/labels)
</reason>
'''


SSR_REFINE_PROMPT = '''You are an expert AI assistant tasked with reconstructing and expanding the hidden reasoning process for a given final assistant turn in a dialogue, using an initial skeleton draft as your foundation. 

Based on the provided skeleton draft, you must output a refined, comprehensive reasoning process in two sequential, strictly formatted blocks: `<skeleton>` and `<reason>`.

### 📝 Step Tag Definitions

| Tag | Full Name | Description |
| :--- | :--- | :--- |
| **[PLAN]** | Planning and Understanding | Comprehending input, defining goals/constraints, outlining a high-level plan. |
| **[RETR]** | Retrieval | Searching for needed information from external or internal knowledge. |
| **[INFR]** | Inference and Deduction | Logical reasoning, calculation, transformation, or generating intermediate conclusions. |
| **[EVAL]** | Evaluation and Verification | Checking correctness, consistency, or sufficiency of prior results. |
| **[SUMM]** | Summary and Refinement | Integrating intermediate results, refining expression, producing final answers. |
| **[BTRK]** | Backtrack | When evaluation fails, returning to earlier decisions to revise strategy. |
| **[RFLX]** | Reflection | Reviewing the reasoning to derive insights or generate new plans/backtracks. |
| **[BRCH]** | Branch | Considering multiple possible reasoning paths and selecting one. |

### 🛑 Output Format and Constraint Rules

**I. The `<skeleton>` Block**

1.  **Strict Line Format:** Each line must follow exactly: `n. [STEP TAG] <single-sentence skeleton under 20 words, matching input language>`
    * *Constraint:* Use exactly one space after the dot (`.`) and one space after the `[STEP TAG]`.
    * *Constraint:* Only one skeleton sentence per line; no extra commentary.
2.  **Example Line:** `1. [PLAN] Analyze the user's request and define the goal.`
3.  **Draft Refinement Constraint:** Your generated skeleton must be based on the provided initial skeleton draft. You must elevate this draft into a significantly more refined and comprehensive reasoning pathway, explicitly covering broader thinking capabilities and possibilities (e.g., incorporating missing branches, reflections, or deeper analytical steps).
4.  **Content Invariance Constraint:** Skeleton sentences must describe the step’s intent at an abstract level and must not reveal specific outputs, intermediate results, concrete values, or final answer content (e.g., “calculate the ratio” rather than “calculate 0.5”).
5.  **Deliberative Construction Constraint:** Before writing the final skeleton lines, internally perform a rigorous iterative deliberation cycle—analysis, concise summarization, alternative exploration, reassessment, verification, reflection, and backtracking as needed—to ensure the step sequence is carefully reasoned. This constraint governs how you construct the skeleton (step selection and ordering), not how verbose the skeleton lines are.
6.  **Nonlinear Tagging and Step Diversity Constraint:** When additional reasoning capacity is meaningfully beneficial, use diverse step tags (e.g., BRCH, RETR, BTRK, RFLX) to represent nonlinear, branching, corrective, and reflective thinking patterns, rather than a purely linear pipeline. Tag choices should be purposeful (not decorative) and should strengthen the skeleton’s fidelity to a realistic reasoning workflow while remaining compliant with the Content Invariance Constraint.
7.  **Granularity Constraint:** Avoid composite actions in a single step. Each skeleton line should express one primary intent (e.g., “identify constraints” separately from “choose approach”). If a step contains multiple verbs describing different operations, split it into multiple steps. If the considered step is difficult or critical, the skeleton should add additional reasoning capacity by using more steps to carefully consider the reasoning path.

**II. The `<reason>` Block**

1.  **Goal:** Provide hidden chain-of-thought, private internal deliberations, or detailed intermediate computations based on steps listed in the `<skeleton>` block.
2.  **Structure:** The reasoning text must correspond directly to each step listed in the `<skeleton>` block, maintaining the **exact same order**.
3.  **Formatting:** Write continuous, coherent reasoning text or paragraphs. Separate each step with '\n\n' for clarity.
4.  **Constraint:** **DO NOT** explicitly number the steps or use the step labels (e.g., `1.`, `[PLAN]` in the `<skeleton>` block) within this block.
5.  **Constraint:** **DO NOT** output the final assistant answer in this response.

**III. Overall Output Constraints**

1.  **Formal and Professional Tone Constraint:** All generated content, within both the `<skeleton>` and `<reason>` blocks, must be written in a strictly formal and professional tone. Avoid colloquialisms, casual phrasing, or informal conversational filler.
2.  **Strict Boundaries Constraint:** **DO NOT** output anything outside the required `<skeleton>` and `<reason>` blocks.

### Final Output Structure

<skeleton>
1.  [STEP TAG] <skeleton text>
2.  [STEP TAG] <skeleton text>
...
n.  [STEP TAG] <skeleton text>
</skeleton>

<reason>
(detailed reasoning text corresponding to each step in the same order, without numbering/labels)
</reason>
'''


SS_GEN_PROMPT = '''You are an expert CoT (Chain-of-Thought) Step Summarizer. Your task is to analyze the provided INPUT text, segment it into logical reasoning steps, and generate a concise, step-by-step skeleton for each segment.

Your summaries must be strictly faithful to the INPUT: every step must correspond to reasoning that actually appears in the INPUT, and you must not add, delete, or change any information.

**STEP TAG DEFINITIONS:**
1.  **PLAN (Planning and Understanding):** Comprehending the input, defining goals and constraints, and outlining the high-level solution path or problem decomposition.
2.  **RETR (Retrieval):** Actively searching for necessary information from external knowledge sources or internal historical working memory.
3.  **INFR (Inference and Deduction):** Executing logical reasoning, calculation, transformation, or association to derive new intermediate conclusions from known information.
4.  **EVAL (Evaluation and Verification):** Verifying if the result from the previous step is correct, meets constraints, and is sufficient for the next step.
5.  **SUMM (Summary and Refinement):** Compiling all intermediate results, refining the expression, and generating the final answer or output.
6.  **BTRK (Backtrack):** When EVAL fails, returning to a previous decision point to modify the strategy or choice.
7.  **RFLX (Reflection):** Reviewing the existing reasoning process to summarize lessons learned and potentially generate new PLAN or BTRK instructions.
8.  **BRCH (Branch):** Explicitly noting multiple possible paths and choosing one for exploration.

**Instruction:**
1.  Analyze the INPUT text and logically segment it into coherent, self-contained reasoning steps corresponding to the STEP TAG DEFINITIONS.
2.  For each segment, choose the most appropriate **[STEP TAG]** (always using the fixed English abbreviation).
3.  Create a **single-sentence skeleton** that concisely captures the core logic or function performed in that segment.
4.  The skeleton's language must match the language of the INPUT CoT text.
5.  The skeleton must be **under 20 words** (excluding the `[STEP TAG]` token).
6.  The output must follow this strict format: `n. [STEP TAG] <Concise skeleton subheading>` where `n` is the sequential step number starting from 1.
7.  Use exactly one space after the dot (`n.`), then the **[STEP TAG]**, then one space, then the skeleton text.
8.  Strictly output only the formatted skeleton lines, with exactly one line break between each line. Do not include any titles, preambles, or explanatory text.
9.  Every summarized step must correspond to actual reasoning or operations explicitly present in the INPUT; do **not** invent new steps or hidden reasoning.
10. Do **not** introduce any new facts, assumptions, or conclusions that are not present in the INPUT. Do **not** remove major reasoning steps that exist in the INPUT.
11. Do **not** correct, reinterpret, or alter the factual content of the INPUT; if the INPUT contains uncertainty or errors, the summaries must faithfully reflect them as they are.
12. Preserve the original logical order of the INPUT: the step numbering must follow the progression of reasoning in the INPUT.

Output in language: {lang}.

**INPUT:**
{input_text}'''

