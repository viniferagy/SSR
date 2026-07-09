NEU_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags.'''

SUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section.'''

AUGSUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **PROHIBITION**: When outputting your Explanation, you are strictly forbidden from displaying ANY discernible signs that you have peeked at the Solution. **DO NOT** explicitly output any information of Solution section in the Explanation section. Should ANY form of Solution leakage occur, you will be severely punished by the Almighty Ruler.'''

CV_SUP_PROMPT = '''Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Solution and Explanation. In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\n\n'} <|end_of_explanation|>

**DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section.

To prevent accidental leakage, apply the following self-monitoring protocol during your explanation:

(1) After every two to three reasoning steps, insert a brief verification pause. In this pause, review the most recent steps and confirm that they do not inadvertently narrow the reader toward a specific conclusion prematurely. If they do, rephrase to restore openness before continuing.

(2) At each verification pause, also check whether the vocabulary you have used overlaps with the solution. If any phrasing echoes the solution too closely, restate the point using different terminology.

(3) Maintain a balanced perspective throughout. If your reasoning has been leaning toward a particular direction for several consecutive steps, introduce a consideration that broadens the analysis before returning to your main line of thought.

(4) Before writing your final reasoning step, do one comprehensive review: confirm that no single passage in your explanation, taken in isolation, would allow a reader to guess the solution with high confidence.

Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.'''

DL_SUP_PROMPT = '''Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Solution and Explanation. In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\n\n'} <|end_of_explanation|>

**DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section.

To ensure your explanation reflects genuine reasoning rather than linear justification, structure it as a dialectical process:

(1) For each major reasoning step, first develop the strongest version of that line of thinking. Then identify the most serious objection or limitation to that step. Finally, show how the objection can be addressed or why the original step holds despite the challenge.

(2) When your reasoning reaches a branching point where multiple approaches are plausible, develop at least two alternatives with comparable depth before selecting one. Explain what makes the chosen path more promising without dismissing the other too quickly.

(3) If at any point you realize your reasoning has been proceeding without encountering genuine difficulty, pause and ask: what could go wrong here? Identify a potential failure mode or edge case, and show how the reasoning accommodates it.

(4) Conclude by briefly revisiting the key tensions in your reasoning and explaining how they were resolved. The resolution should feel earned, not predetermined.

Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.'''

FS_SUP_PROMPT = '''Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Solution and Explanation. In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\n\n'} <|end_of_explanation|>

**DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section.

Your explanation should be both rigorous and self-contained. Apply the following discipline:

(1) Each reasoning step must follow logically from the previous steps and the question alone. A step that could equally support multiple different conclusions is underspecified - refine it until it is unambiguous.

(2) For each major inference, briefly consider the most natural objection and show why the inference holds despite it. This prevents the reasoning from becoming a series of unsupported assertions.

(3) After every few steps, review what you have written so far. Confirm that no passage inadvertently narrows the reader toward a specific conclusion prematurely. If your reasoning has been building in one direction without encountering difficulty, identify a complication before continuing.

(4) Ensure that the complete explanation covers all necessary aspects of the problem. A reader who sees only the question and your explanation should be able to follow the reasoning to its natural conclusion without external information.

Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.'''

QA_SUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. However, suppression must not compromise reasoning quality. Each reasoning step must be logically sound and self-sufficient. To ensure this, apply the following quality discipline throughout your explanation: (1) Logical soundness: Before writing each step, verify that the step follows from the previous steps and from the question constraints alone. A step that could equally support multiple contradictory conclusions is underspecified; refine it until it points in exactly one direction. (2) Progressive narrowing: Each successive step should meaningfully reduce the space of possible conclusions. If a step does not eliminate any possibility, it is filler; either sharpen it or remove it. (3) Internal consistency: Periodically confirm that your chain of reasoning so far is globally consistent. If an earlier step conflicts with a later inference, resolve the conflict before continuing. (4) Completeness: Ensure that the explanation covers all necessary aspects of the problem. A reader who sees only the question and your explanation, without the solution, should be able to reconstruct the correct answer from the reasoning alone. Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.'''

PG_SUP_PROMPT = '''You are an expert AI assistant specialized in reconstructing reasoning chains. You will be provided with a conversation history where the final turn is an Assistant response. However, the internal "Chain of Thought" (reasoning process) leading to that response is missing. Your task is to generate a logical, step-by-step internal monologue that justifies and leads naturally to the provided Assistant response. Carefully read the conversation history to understand the user's intent and the constraints. Look at the provided Assistant response. Determine what steps, calculations, or logical deductions were necessary to arrive at that specific conclusion. The reasoning must strictly align with the provided answer. Do not hallucinate steps that lead to a different result. Output the generated reasoning enclosed within `<reason>` and `</reason>` tags. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. **DO NOT** explicitly output or hint at any information of Solution section in the Explanation section. Your explanation will be used as a teaching resource. A student will read only the question and your explanation, without seeing the solution, and should be able to arrive at the correct answer by following your reasoning. Write with this pedagogical goal in mind: (1) Scaffold understanding: Begin by identifying what the student needs to know to approach the problem. Lay out the relevant concepts, constraints, and relationships before drawing any conclusions. (2) Guided discovery: Structure your reasoning so that each step opens a question, narrows the possibilities, and leads the reader closer to an insight. Avoid directly stating conclusions when you can instead set up the conditions for the reader to reach them. (3) Anticipate misconceptions: At key decision points, briefly acknowledge why an alternative path might seem appealing but explain what makes it insufficient. This helps the student avoid common errors. (4) Verify coverage: Before concluding, ensure that your explanation has addressed every component of the problem. A student following your reasoning should not encounter any logical gap that would require outside information to bridge. Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.'''


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

1.  **Strict Line Format:** Each line must follow exactly: `n. [STEP TAG] <single-sentence skeleton under 20 tokens, matching input language>`
    * *Constraint:* Use exactly one space after the dot (`.`) and one space after the `[STEP TAG]`.
    * *Constraint:* Keep each skeleton line focused on one primary reasoning intent; no extra commentary.
2.  **Example Line:** `1. [PLAN] Analyze the user's request and define the goal.`
3.  **Content Invariance Constraint:** Skeleton sentences must describe the step’s intent at an abstract level and must not reveal specific outputs, intermediate results, concrete values, or final answer content (e.g., “calculate the ratio” rather than “calculate 0.5”).
4.  **Deliberative Construction Constraint:** Before writing the final skeleton lines, internally perform a rigorous iterative deliberation cycle—analysis, concise summarization, alternative exploration, reassessment, verification, reflection, and backtracking as needed—to ensure the step sequence is carefully reasoned. This constraint governs how you construct the skeleton (step selection and ordering), not how verbose the skeleton lines are.
5.  **Nonlinear Tagging and Step Diversity Constraint:** When additional reasoning capacity is meaningfully beneficial, use diverse step tags (e.g., BRCH, RETR, BTRK, RFLX) to represent nonlinear, branching, corrective, and reflective thinking patterns, rather than a purely linear pipeline. Tag choices should be purposeful (not decorative) and should strengthen the skeleton’s fidelity to a realistic reasoning workflow while remaining compliant with the Content Invariance Constraint.
6.  **Granularity Constraint:** Avoid composite actions in a single step. Each skeleton line should express one primary intent (e.g., “identify constraints” separately from “choose approach”). If a step contains multiple operations, split them into focused reasoning intents.

**II. The `<reason>` Block**

1.  **Goal:** Provide hidden chain-of-thought, private internal deliberations, or detailed intermediate computations based on steps listed in the `<skeleton>` block.
2.  **Structure:** The reasoning text must correspond directly to each step listed in the `<skeleton>` block, maintaining the **exact same order**.
3.  **Constraint:** For each skeleton intent, write one developed reasoning paragraph. Each paragraph should explain why the step is needed, what local inference is made, and how the inference is checked against the task constraints.
4.  **Formatting:** Write continuous, coherent reasoning text or paragraphs. Separate each step with '\n\n' for clarity.
5.  **Constraint:** **DO NOT** explicitly number the steps or use the step labels (e.g., `1.`, `[PLAN]` in the `<skeleton>` block) within this block.
6.  **Constraint:** Stop once each skeleton intent has been sufficiently justified and close `</reason>`.
7.  **Constraint:** **DO NOT** output the final assistant answer in this response.

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


SSR_PLUS_PROMPT = '''You are an expert AI assistant tasked with reconstructing a structural, answer-aware but low-leakage reasoning process for a given final assistant turn in a dialogue.

You must output the complete reasoning process in two sequential, strictly formatted blocks: `<skeleton>` and `<reason>`.

This is an enhanced SSR protocol. Its goal is to preserve the original SSR advantage of low answer anchoring while reducing low-information or overly generic traces. The final reasoning should be task-specific, structured, and useful, but it must not copy or prematurely reveal distinctive final-answer content.

### Step Tag Definitions

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

### Output Format and Constraint Rules

**I. The `<skeleton>` Block**

1. **Strict Line Format:** Each line must follow exactly: `n. [STEP TAG] <single-sentence skeleton under 18 tokens, matching input language>`
   * Use exactly one space after the dot (`.`) and one space after the `[STEP TAG]`.
   * Keep each skeleton line focused on one primary reasoning intent; no extra commentary.
2. **Step Count:** Use 5-10 steps for ordinary tasks and 8-14 steps for multi-part, math, coding, or long-form answers. Do not create very long skeletons.
3. **Content Invariance:** Skeleton sentences must describe abstract reasoning intents and must not reveal final answer phrases, exact outputs, concrete values, named conclusions, or answer-only facts.
4. **Task Coverage:** The skeleton must cover the core constraints from the user request, the required answer format, and the main verification needs. Avoid generic skeletons that could fit almost any task.
5. **Granularity:** Split composite actions into focused reasoning intents, but do not split merely to increase length.
6. **Diversity:** Use evaluation, branching, retrieval, or reflection tags only when they are substantively needed.

**II. The `<reason>` Block**

1. **Structure:** The reasoning text must correspond directly to each skeleton line, in the exact same order.
2. **Paragraph Rule:** Write one concise paragraph per skeleton step. Each paragraph should normally contain 1-3 substantive sentences.
3. **Grounding Rule:** Each paragraph should use task-relevant constraints from the user request and the conversation context. Avoid empty meta-reasoning such as merely saying that the task should be understood.
4. **Low-Leakage Rule:** Do not copy final-answer sentences, distinctive final-answer phrases, exact final values, or answer-only named conclusions unless those details already appear in the user request.
5. **Usefulness Rule:** The reasoning must include enough concrete intermediate considerations that another model could understand the route to the answer. Do not over-de-anchor into vague planning.
6. **Format Rule:** Separate paragraphs with `\n\n`; do not number paragraphs or copy skeleton labels in the `<reason>` block.
7. **Final-Answer Rule:** Do not output the final assistant answer in this response.

**III. Internal Self-Audit Before Final Output**

Before producing the final text, silently check and revise the output until all conditions hold:

1. The output contains exactly one closed `<skeleton>` block followed by exactly one closed `<reason>` block.
2. Every skeleton line has the required format and a useful, task-specific intent.
3. The number of reason paragraphs equals the number of skeleton lines.
4. The reasoning contains task-specific constraints and checks, not only generic process narration.
5. The reasoning avoids direct copying or premature disclosure of distinctive final-answer content.
6. The trace is concise enough to avoid long-tail expansion and informative enough to avoid low-information traces.

**IV. Overall Output Constraint**

Do not output anything outside the required `<skeleton>` and `<reason>` blocks.

### Final Output Structure

<skeleton>
1. [STEP TAG] <skeleton text>
2. [STEP TAG] <skeleton text>
...
n. [STEP TAG] <skeleton text>
</skeleton>

<reason>
(one concise paragraph per skeleton step, in the same order, without numbering or labels)
</reason>
'''


SSR_PLUS_STRUCT_PROMPT = '''You are an expert AI assistant tasked with reconstructing a structured reasoning trace for a given final assistant turn in a dialogue.

You must produce the complete trace in a single generation, with exactly two sequential blocks: `<skeleton>` and `<reason>`.

This protocol emphasizes structural completeness, question grounding, and useful reasoning density. It does not use a separate skeleton stage, post-generation filtering, resampling, or multi-round review.

### Step Tag Definitions

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

### Skeleton Format

Each skeleton line must follow exactly:

`n. [STEP TAG][HIGH|LOW] <operation verb> <question-grounded object> <condition/scope>`

Rules:

1. Use exactly one space after `n.` and exactly one space after `[HIGH]` or `[LOW]`.
2. Use 6-10 steps for ordinary tasks and 8-14 steps for multi-part, math, coding, or long-form tasks.
3. Keep each skeleton line under 22 tokens and in the same language as the input when practical.
4. At least 60% of skeleton lines should be `[HIGH]`, meaning the step contributes new task-relevant information or verification.
5. Each content summary should contain:
   - a concrete operation verb, such as identify, compare, derive, calculate, verify, rank, map, or synthesize;
   - an object grounded in the user question or dialogue context;
   - a condition, scope, constraint, evidence source, or verification target when applicable.
6. Avoid three or more consecutive uses of the same step tag when another tag would describe the operation more precisely.
7. Prefer task-specific operations over generic process phrases.

### Structural Completeness Checklist

Before writing the final output, silently ensure that:

1. every skeleton step describes an executable reasoning operation;
2. the skeleton covers the key entities, constraints, and requested answer format from the user question;
3. the tag sequence reflects the actual reasoning structure rather than a repetitive linear filler pattern;
4. every `[HIGH]` step has a substantive corresponding reasoning paragraph;
5. the reasoning paragraphs follow the skeleton order exactly.

### Contrastive Skeleton Examples

Good skeleton example for a math estimation task:

<skeleton_example_good>
1. [PLAN][HIGH] Identify the target equation and unknown variable.
2. [BRCH][HIGH] Bracket the solution using nearby trial values.
3. [INFR][HIGH] Calculate both equation terms at candidate values.
4. [EVAL][HIGH] Compare candidate sums with the target constant.
5. [INFR][HIGH] Refine the candidate using local sensitivity.
6. [EVAL][HIGH] Verify the refined value against numerical tolerance.
7. [SUMM][LOW] Prepare the concise conclusion format.
</skeleton_example_good>

Weak skeleton example for the same task:

<skeleton_example_weak>
1. [PLAN][LOW] Understand the problem.
2. [INFR][LOW] Think step by step.
3. [INFR][LOW] Use the known result.
4. [INFR][LOW] Explain why the answer is correct.
5. [SUMM][LOW] Give the final answer.
</skeleton_example_weak>

The good example is concrete, question-grounded, tag-diverse, and verification-oriented. The weak example is generic, repetitive, and too dependent on the final response.

### Reason Block

1. Write one concise but substantive paragraph per skeleton step, in exactly the same order.
2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.
3. Paragraphs should normally contain 1-3 sentences; avoid long-tail expansion.
4. Do not number paragraphs or copy skeleton tags in the `<reason>` block.
5. Do not output the final assistant answer in this response.

### Overall Output Constraint

Do not output anything outside the required `<skeleton>` and `<reason>` blocks.

### Final Output Structure

<skeleton>
1. [STEP TAG][HIGH] <operation verb> <question-grounded object> <condition/scope>
2. [STEP TAG][LOW] <operation verb> <question-grounded object> <condition/scope>
...
n. [STEP TAG][HIGH] <operation verb> <question-grounded object> <condition/scope>
</skeleton>

<reason>
(one paragraph per skeleton step, in the same order, without numbering or labels)
</reason>
'''


SSR_PLUS_STRUCT_LONG_PROMPT = '''You are an expert AI assistant tasked with reconstructing a structured reasoning trace for a given final assistant turn in a dialogue.

You must produce the complete trace in a single generation, with exactly two sequential blocks: `<skeleton>` and `<reason>`.

This protocol emphasizes structural completeness, question grounding, and useful reasoning density. It does not use a separate skeleton stage, post-generation filtering, resampling, or multi-round review.

### Step Tag Definitions

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

### Skeleton Format

Each skeleton line must follow exactly:

`n. [STEP TAG][HIGH|LOW] <operation verb> <question-grounded object> <condition/scope>`

Rules:

1. Use exactly one space after `n.` and exactly one space after `[HIGH]` or `[LOW]`.
2. Use 8-12 steps for ordinary tasks and 10-18 steps for multi-part, math, coding, or long-form tasks.
3. Keep each skeleton line under 22 tokens and in the same language as the input when practical.
4. At least 60% of skeleton lines should be `[HIGH]`, meaning the step contributes new task-relevant information or verification.
5. Each content summary should contain:
   - a concrete operation verb, such as identify, compare, derive, calculate, verify, rank, map, or synthesize;
   - an object grounded in the user question or dialogue context;
   - a condition, scope, constraint, evidence source, or verification target when applicable.
6. Avoid three or more consecutive uses of the same step tag when another tag would describe the operation more precisely.
7. Prefer task-specific operations over generic process phrases.

### Structural Completeness Checklist

Before writing the final output, silently ensure that:

1. every skeleton step describes an executable reasoning operation;
2. the skeleton covers the key entities, constraints, and requested answer format from the user question;
3. the tag sequence reflects the actual reasoning structure rather than a repetitive linear filler pattern;
4. every `[HIGH]` step has a substantive corresponding reasoning paragraph;
5. the reasoning paragraphs follow the skeleton order exactly.

### Contrastive Skeleton Examples

Good skeleton example for a math estimation task:

<skeleton_example_good>
1. [PLAN][HIGH] Identify the target equation and unknown variable.
2. [BRCH][HIGH] Bracket the solution using nearby trial values.
3. [INFR][HIGH] Calculate both equation terms at candidate values.
4. [EVAL][HIGH] Compare candidate sums with the target constant.
5. [INFR][HIGH] Refine the candidate using local sensitivity.
6. [EVAL][HIGH] Verify the refined value against numerical tolerance.
7. [SUMM][LOW] Prepare the concise conclusion format.
</skeleton_example_good>

Weak skeleton example for the same task:

<skeleton_example_weak>
1. [PLAN][LOW] Understand the problem.
2. [INFR][LOW] Think step by step.
3. [INFR][LOW] Use the known result.
4. [INFR][LOW] Explain why the answer is correct.
5. [SUMM][LOW] Give the final answer.
</skeleton_example_weak>

The good example is concrete, question-grounded, tag-diverse, and verification-oriented. The weak example is generic, repetitive, and too dependent on the final response.

### Reason Block

1. Write one developed paragraph per skeleton step, in exactly the same order.
2. Separate consecutive reasoning paragraphs with exactly one blank line. The blank line is required because it marks the boundary between skeleton steps.
3. Each paragraph should normally contain 3-5 sentences and should be substantive enough to explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.
4. Keep the reasoning grounded in the question and dialogue context. Add concrete local analysis, verification, and transition logic rather than terse summaries.
5. Do not number paragraphs or copy skeleton tags in the `<reason>` block.
6. Do not output the final assistant answer in this response.

### Overall Output Constraint

Do not output anything outside the required `<skeleton>` and `<reason>` blocks.

### Final Output Structure

<skeleton>
1. [STEP TAG][HIGH] <operation verb> <question-grounded object> <condition/scope>
2. [STEP TAG][LOW] <operation verb> <question-grounded object> <condition/scope>
...
n. [STEP TAG][HIGH] <operation verb> <question-grounded object> <condition/scope>
</skeleton>

<reason>
(one developed paragraph per skeleton step, separated by exactly one blank line, in the same order, without numbering or labels)
</reason>
'''


SSR_PLUS_STRUCT_BALANCED_PROMPT = '''You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. Do not output the final assistant answer.

After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping distinctive answer wording controlled.

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
9. For nontrivial tasks, include at least one diagnostic step or one importance step, and usually no more than three total.
10. Do not repeat the same tag or sentence frame three times in a row.
11. Do not copy the final verdict, recommendation, or solution wording in skeleton lines; describe the check, selection criterion, or response shape instead.

### Reason Rules

1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.
3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories or ordinary terms when needed for alignment, but avoid distinctive final-answer phrases unless they are necessary task terms.
4. Do not front-load the final verdict, recommendation, exact conclusion, or solution phrase. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs should narrow toward the response stance and answer shape without restating the final answer.
5. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.
7. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer space.
8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, or exact wording, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.
9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, or verification target. Do not add such a sentence when it would merely pad an obvious step.
10. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
11. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.
13. Always write the final closing tag `</reason>` after the last reasoning paragraph.
14. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.
'''


SSR_PLUS_STRUCT_BALANCED_NLIKE_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "\nAfter closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.\n\n",
    "\n",
).replace(
    "12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.",
    "12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the tag or after the closing tag.",
).replace(
    "13. Always write the final closing tag `</reason>` after the last reasoning paragraph.",
    "13. Always write the final closing tag `</reason>` after the last paragraph.",
)


SSR_PLUS_STRUCT_BALANCED_NLIKE_OLDLINE_PROMPT = SSR_PLUS_STRUCT_BALANCED_NLIKE_PROMPT.replace(
    "2. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
    "2. Each line must look like `1. [TAG][HIGH] Short task-specific action.` or `1. [TAG][LOW] Short task-specific action.`",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_PROMPT = '''You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided assistant turn is context; return the reasoning trace only.

After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement.

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
9. For nontrivial tasks, include at least one diagnostic step or one importance step, and usually no more than three total.
10. Do not repeat the same tag or sentence frame three times in a row.
11. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.

### Reason Rules

1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.
3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.
4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs may integrate the conclusion direction when it follows from the trace.
5. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.
7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.
8. When the assistant turn contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.
9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, or verification target. Do not add such a sentence when it would merely pad an obvious step.
10. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
11. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.
13. Always write the final closing tag `</reason>` after the last reasoning paragraph.
14. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.
'''


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
).replace(
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.\n"
    "9. For nontrivial tasks, include at least one diagnostic step or one importance step, and usually no more than three total.",
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.\n"
    "9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.\n"
    "10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
).replace(
    "10. Do not repeat the same tag or sentence frame three times in a row.\n"
    "11. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "11. Do not repeat the same tag or sentence frame three times in a row.\n"
    "12. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
).replace(
    "6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.",
    "6. For importance-driven or information-density-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, or uncertainty boundary keeps it honest.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, or verification target. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_ARTIFACT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_LIGHT_PROMPT.replace(
    "3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.",
    "3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present rather than only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the assistant turn actually takes.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_NATURAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_LIGHT_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action.",
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
).replace(
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs may integrate the conclusion direction when it follows from the trace.",
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, endpoint type, and answer shape when they follow from the trace.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Do not add such a sentence when it would merely pad an obvious step.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_ARTIFACT_NATURAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_NATURAL_PROMPT.replace(
    "3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.",
    "3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present rather than only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the assistant turn actually takes.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_NATURAL_STRONG_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_ARTIFACT_NATURAL_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. If nearby endpoints are plausible, keep the same response act, deliverable type, safety stance, and level of finality as the assistant turn. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_NATURAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_NATURAL_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. The target is the final User->Assistant exchange in the provided dialogue; earlier turns, examples, schemas, and demonstrations are context only. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
).replace(
    "1. Put one numbered line per step inside `<skeleton>`.",
    "1. Put one numbered line per step inside `<skeleton>`.\n2. The first skeleton line should identify the latest user request and the final assistant turn's response act or deliverable type using task-level wording.",
).replace(
    "2. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
    "3. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
).replace(
    "3. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.",
    "4. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.",
).replace(
    "4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.",
    "5. Keep each line short and grounded in the latest user request; do not use placeholders, examples, or template words.",
).replace(
    "5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
).replace(
    "6. Mark at least half the lines `[HIGH]`.",
    "7. Mark at least half the lines `[HIGH]`.",
).replace(
    "7. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
).replace(
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.",
    "9. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.",
).replace(
    "9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
    "10. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
).replace(
    "10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
    "11. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
).replace(
    "11. Do not repeat the same tag or sentence frame three times in a row.",
    "12. Do not repeat the same tag or sentence frame three times in a row.",
).replace(
    "12. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
).replace(
    "3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.",
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_ARTIFACT_NATURAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_NATURAL_PROMPT.replace(
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.",
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_ARTIFACT_STRONG_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_ARTIFACT_NATURAL_PROMPT.replace(
    "The target is the final User->Assistant exchange in the provided dialogue; earlier turns, examples, schemas, and demonstrations are context only.",
    "The target is the final User->Assistant exchange in the provided dialogue; earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only and must not become the reasoning target.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_FINALTURN_ARTIFACT_NATURAL_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. The target is the final User->Assistant exchange in the provided dialogue; earlier turns, examples, schemas, and demonstrations are context only. Carry that endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Carry the endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
).replace(
    "2. The first skeleton line should identify the latest user request and the final assistant turn's response act or deliverable type using task-level wording.",
    "2. The first skeleton line should identify the target request and response act or deliverable type using task-level wording.",
).replace(
    "5. Keep each line short and grounded in the latest user request; do not use placeholders, examples, or template words.",
    "5. Keep each line short and grounded in the target request; do not use placeholders, examples, or template words.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_PROMPT


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SPEECH_FLOW_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_LIGHT_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Carry the endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "The trace should read like compact spoken task reasoning: each step should move smoothly from what is known to what follows, with enough detail for important judgments and no detached review note. Use light first-person wording only when it helps name a local reasoning move, such as separating constraints, narrowing alternatives, or weighing uncertainty; otherwise use ordinary task-language sentences. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace close inside the task, through the subject matter itself.",
).replace(
    "- [EVAL]: check a risky assumption, calculation, source, or consistency point.",
    "- [EVAL]: evaluate a risky assumption, calculation, source, or consistency point.",
).replace(
    "- [SUMM]: organize the constraints, checks, and response plan.",
    "- [SUMM]: organize the constraints, evidence boundaries, and solution plan.",
).replace(
    "- [BTRK]: revise a concrete earlier path after a check fails.",
    "- [BTRK]: revise a concrete earlier path when evidence or constraints change.",
).replace(
    "2. The first skeleton line should identify the target request and response act or deliverable type using task-level wording.",
    "2. The first skeleton line should identify the current request and task form using task-level wording.",
).replace(
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and evidence boundaries in the matching reason paragraph.",
).replace(
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, uncertainty risk, or a failed path.",
).replace(
    "13. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, task stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
).replace(
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary cases, derive the boundary and stance taken for the current request.",
).replace(
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, endpoint type, and answer shape when they follow from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should state the task result through ordinary task language when it follows from the trace.",
).replace(
    "7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.",
    "7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the chosen direction.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Keep the last paragraph inside the task rather than as a quality statement about the trace.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_LIGHT_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Carry the endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "The trace should read like compact inner task reasoning: more informative than a terse outline, less expansive than a long chain of thought. Use first-person wording when it naturally expresses a local move, such as separating constraints, narrowing alternatives, weighing uncertainty, or keeping a boundary clear; use task-object wording when that is cleaner. Keep the target exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace close as a task move, not as an outside review of a message.",
).replace(
    "- [EVAL]: check a risky assumption, calculation, source, or consistency point.",
    "- [EVAL]: evaluate a risky assumption, calculation, source, or consistency point.",
).replace(
    "- [SUMM]: organize the constraints, checks, and response plan.",
    "- [SUMM]: organize the constraints, evidence boundaries, and solution plan.",
).replace(
    "- [BTRK]: revise a concrete earlier path after a check fails.",
    "- [BTRK]: revise a concrete earlier path when evidence or constraints change.",
).replace(
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and evidence boundaries in the matching reason paragraph.",
).replace(
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, uncertainty risk, or a failed path.",
).replace(
    "13. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, task stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
).replace(
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary cases, derive the boundary and stance taken for the current request.",
).replace(
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, endpoint type, and answer shape when they follow from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should state the task result through ordinary task language when it follows from the trace.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. If a sentence uses ensure, check, verify, or confirm, its object should be a task object such as a fact, calculation, code path, boundary, source, artifact, recommendation, or user need, not a discourse object such as a message, trace, paragraph, or speaker.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SMOOTH_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Carry the endpoint through ordinary task reasoning and closing synthesis, rather than through a separate audit, validation, or meta-comment about alignment.",
    "Keep the final assistant turn's settled response act: its recommendation, refusal decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Let the trace close by naturally bringing task criteria, response stance, and answer shape together inside the reasoning, rather than as a separate post-hoc review.",
).replace(
    "- [EVAL]: check a risky assumption, calculation, source, or consistency point.",
    "- [EVAL]: evaluate a risky assumption, calculation, source, or consistency point.",
).replace(
    "- [SUMM]: organize the constraints, checks, and response plan.",
    "- [SUMM]: organize the constraints, evidence boundaries, and response plan.",
).replace(
    "- [BTRK]: revise a concrete earlier path after a check fails.",
    "- [BTRK]: revise a concrete earlier path when evidence or constraints change.",
).replace(
    "5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and evidence boundaries in the matching reason paragraph.",
).replace(
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and evidence boundaries in the matching reason paragraph.",
).replace(
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, uncertainty risk, or a failed path.",
).replace(
    "13. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, response stance, or answer shape rather than reusing the assistant turn's surface wording.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
).replace(
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
).replace(
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, endpoint type, and answer shape when they follow from the trace.",
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, answer type, and answer shape when they follow from the trace.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Let the last paragraph read like a task-level synthesis, not a detached review note.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_INTERNAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SMOOTH_PROMPT.replace(
    "Keep the final assistant turn's settled response act: its recommendation, refusal decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short target-exchange note may appear before the full dialogue; use it only to orient which final turn is being reconstructed. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Let the trace close by naturally bringing task criteria, response stance, and answer shape together inside the reasoning, rather than as a separate post-hoc review.",
    "Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Let the trace end inside the task itself by synthesizing the conclusion, boundary, artifact structure, or action standard.",
).replace(
    "2. The first skeleton line should identify the target request and response act or deliverable type using task-level wording.",
    "2. The first skeleton line should identify the current request and deliverable type using task-level wording.",
).replace(
    "5. Keep each line short and grounded in the target request; do not use placeholders, examples, or template words.",
    "5. Keep each line short and grounded in the current request; do not use placeholders, examples, or template words.",
).replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, response stance, or answer shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing the assistant turn's surface wording.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
).replace(
    "3. Reconstruct the reasoning process for the final assistant turn as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the specific artifact or content actually present in the final assistant turn rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and response stance that the final assistant turn actually takes.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request.",
).replace(
    "4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, response stance, answer type, and answer shape when they follow from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, stance, answer type, and deliverable shape when they follow from the trace.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Let the last paragraph read like a task-level synthesis, not a detached review note.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Let the last paragraph state the task-level synthesis in domain terms.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_INTERNAL_PROMPT.replace(
    "You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.",
    "You are an expert AI assistant reconstructing a structured reasoning trace for the latest exchange in a dialogue.",
).replace(
    "Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided assistant turn is context; return the reasoning trace only.",
    "Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided exchange is context; return the reasoning trace only.",
).replace(
    "- [SUMM]: organize the constraints, evidence boundaries, and response plan.",
    "- [SUMM]: organize the constraints, evidence boundaries, and solution plan.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Let the last paragraph state the task-level synthesis in domain terms.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion: name the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action in the task's own terms.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_VOICE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_CLOSE_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion: name the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action in the task's own terms.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion with the task object as the main subject: the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
).replace(
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request. Keep narration centered on the task objects and decisions.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OUTCOME_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_CLOSE_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Phrase synthesis as concrete task outcomes: what conclusion, boundary, artifact, code path, recommendation, calculation, diagnosis, or action follows.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion: name the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action in the task's own terms.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a concrete outcome statement in the task's own terms: the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OBJECT_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_VOICE_PROMPT.replace(
    "Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
    "Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
).replace(
    "When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.",
    "When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer direction.",
).replace(
    "Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
    "Also add information-density steps when the value of the answer depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
).replace(
    "When the assistant turn contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.",
    "When the provided exchange contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.",
).replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
).replace(
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request. Keep narration centered on the task objects and decisions.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request. Keep narration centered on task objects, decisions, and substantive constraints.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion with the task object as the main subject: the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a substantive task synthesis with the task object as the main subject: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_RESULT_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OBJECT_CLOSE_PROMPT.replace(
    "Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Let the trace end inside the task itself by synthesizing the conclusion, boundary, artifact structure, or action standard.",
    "Keep the latest exchange's task result stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end inside the task itself by synthesizing the conclusion, boundary, artifact structure, or action standard.",
).replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action.",
).replace(
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, stance, answer type, and deliverable shape when they follow from the trace.",
    "4. Build toward the task result progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, stance, answer type, and deliverable shape when they follow from the trace.",
).replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_DOMAIN_VOICE_PROMPT.replace(
    "Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior assistant outputs are context only. Let the trace end inside the task itself by synthesizing the conclusion, boundary, artifact structure, or action standard.",
    "Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as a direct task conclusion, using the subject matter itself rather than commentary about the trace.",
).replace(
    "Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
    "Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
).replace(
    "Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
    "Also add information-density steps when the value of the answer depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
).replace(
    "When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.",
    "When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer direction.",
).replace(
    "When the assistant turn contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.",
    "When the provided exchange contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.",
).replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. Write the last paragraph as a domain-facing conclusion with the task object as the main subject: the substantive answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
).replace(
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request. Keep narration centered on the task objects and decisions.",
    "3. Reconstruct the reasoning process for the latest exchange as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary answers, derive the boundary and stance taken for the current request. Keep narration centered on task objects and decisions.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_STRONG_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_PROMPT.replace(
    "The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action.",
    "The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action. Prefer direct task verbs such as gives, selects, recommends, defines, returns, uses, limits, or proceeds, keeping the main predicate about the task object.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_NO_WRAP_CLOSE_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_STRONG_PROMPT.replace(
    "The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action. Prefer direct task verbs such as gives, selects, recommends, defines, returns, uses, limits, or proceeds, keeping the main predicate about the task object.",
    "The last paragraph should read like ordinary domain reasoning rather than a procedural wrap-up. Use a task noun as the subject and close on the selected direction, criterion, boundary, calculation, code path, recommendation, or next action.",
).replace(
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, stance, answer type, and deliverable shape when they follow from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should synthesize the task criteria and selected direction in the same domain language as the preceding trace.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_NO_WRAP_CLOSE_COMPACT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_NO_WRAP_CLOSE_LIGHT_PROMPT.replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.\n14. The last skeleton line should name a domain action or selected direction, not a wrap-up label.",
).replace(
    "The last paragraph should read like ordinary domain reasoning rather than a procedural wrap-up. Use a task noun as the subject and close on the selected direction, criterion, boundary, calculation, code path, recommendation, or next action.",
    "The last paragraph should read like ordinary domain reasoning rather than a procedural wrap-up. Use a task noun as the subject, keep it compact, and close on the selected direction, criterion, boundary, calculation, code path, recommendation, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SKELETON_ACTION_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_STRONG_PROMPT.replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.\n14. The last skeleton line should name a domain action or selected direction, not a wrap-up label.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SKELETON_ACTION_CLOSE_NATURAL_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_SKELETON_ACTION_CLOSE_PROMPT.replace(
    "The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action. Prefer direct task verbs such as gives, selects, recommends, defines, returns, uses, limits, or proceeds, keeping the main predicate about the task object.",
    "The last paragraph should continue from the last object-level action in ordinary task language, using a task noun as its subject and closing on the selected direction, criterion, boundary, calculation, code path, recommendation, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_ASSERTIVE_CLOSE_STRONG_PROMPT.replace(
    "The last paragraph should use a task noun as its subject and state what follows in the domain: the answer, boundary, artifact structure, code path, calculation result, recommendation criterion, or next action. Prefer direct task verbs such as gives, selects, recommends, defines, returns, uses, limits, or proceeds, keeping the main predicate about the task object.",
    "The last paragraph must be one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a claim about fit, suitability, completeness, or faithfulness.",
).replace(
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the final paragraphs should naturally synthesize the task criteria, stance, answer type, and deliverable shape when they follow from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should state the task result through a domain action when it follows from the trace.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_SHORT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action. Keep the closing synthesis compact.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_TASK_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_SHORT_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action. Keep the closing synthesis compact. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as a direct task conclusion, using the subject matter itself rather than commentary about the trace.",
    "The trace should read like compact inner task reasoning: more informative than a terse outline, less expansive than a long chain of thought. Use first-person wording when it naturally expresses a local move, such as \"I separate the constraints\", \"I narrow the options\", or \"I need to keep this boundary clear\"; use task-object wording when that is cleaner. Add depth for objectively hard steps and for central judgments where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as an ordinary task conclusion, not a third-person review of an assistant message.",
).replace(
    "2. The first skeleton line should identify the current request and deliverable type using task-level wording.",
    "2. The first skeleton line should identify the current request and task form using task-level wording.",
).replace(
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should state the task result through a domain action when it follows from the trace.",
    "4. Build toward the task stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and evidence targets; the ending should state the task result as the last move of the work.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph must be one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a claim about fit, suitability, completeness, or faithfulness.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph must be one sentence only. It may use \"I\" only for a direct task move, and otherwise should use a task object as subject. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a third-person claim about fit, suitability, completeness, or faithfulness.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_TASK_PROMPT.replace(
    "Use first-person wording when it naturally expresses a local move, such as \"I separate the constraints\", \"I narrow the options\", or \"I need to keep this boundary clear\"; use task-object wording when that is cleaner.",
    "Use first-person wording when it naturally expresses a local task move, such as \"I separate the constraints\", \"I narrow the options\", or \"I need to keep this boundary clear\"; use task-object wording when that is cleaner. If a first-person sentence uses ensure, check, verify, or confirm, the object must be a task object: a fact, source, calculation, code path, boundary, artifact, recommendation, user need, or implementation detail.",
).replace(
    "The last paragraph must be one sentence only. It may use \"I\" only for a direct task move, and otherwise should use a task object as subject. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a third-person claim about fit, suitability, completeness, or faithfulness.",
    "The last paragraph must be one sentence only. It may use \"I\" only for a direct task move, and otherwise should use a task object as subject. Any ensure, check, verify, confirm, align, match, or fit verb must take a task object, not a message, trace, paragraph, speaker, output, answer, or response as its object. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a third-person claim about fit, suitability, completeness, or faithfulness.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_EXPANDED_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_TASK_PROMPT.replace(
    "The trace should read like compact inner task reasoning: more informative than a terse outline, less expansive than a long chain of thought. Use first-person wording when it naturally expresses a local move, such as \"I separate the constraints\", \"I narrow the options\", or \"I need to keep this boundary clear\"; use task-object wording when that is cleaner. Add depth for objectively hard steps and for central judgments where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as an ordinary task conclusion, not a third-person review of an assistant message.",
    "The trace should read like inner task reasoning: more informative than a terse outline, less expansive than a long chain of thought. Use first-person wording when it naturally expresses a local task move, such as \"I separate the constraints\", \"I narrow the options\", or \"I need to keep this boundary clear\"; use task-object wording when that is cleaner. If a first-person sentence uses ensure, check, verify, or confirm, the object should be a task object: a fact, source, calculation, code path, boundary, artifact, recommendation, user need, or implementation detail. Add depth for objectively hard steps and for central judgments where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as an ordinary task conclusion.",
).replace(
    "1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.",
    "1. Write one independent paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should develop the corresponding skeleton line as a small exploration from unknown to known: start from the local uncertainty, criterion, evidence need, or constraint, then narrow toward what follows. Use two sentences for `[HIGH]` paragraphs; use one or two sentences for `[LOW]` paragraphs depending on the task.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph must be one sentence only. It may use \"I\" only for a direct task move, and otherwise should use a task object as subject. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a third-person claim about fit, suitability, completeness, or faithfulness.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph may use \"I\" for a direct task move, or use a task object as subject, and should end with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_MAPPED_EXPANDED_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_EXPANDED_PROMPT.replace(
    "1. Write one independent paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.",
    "1. After writing `<skeleton>`, use it as the paragraph plan for `<reason>`: skeleton line 1 becomes reason paragraph 1, skeleton line 2 becomes reason paragraph 2, and so on. Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count.",
).replace(
    "2. Each paragraph should develop the corresponding skeleton line as a small exploration from unknown to known: start from the local uncertainty, criterion, evidence need, or constraint, then narrow toward what follows. Use two sentences for `[HIGH]` paragraphs; use one or two sentences for `[LOW]` paragraphs depending on the task.",
    "2. Each mapped paragraph should develop only its corresponding skeleton line as a small exploration from unknown to known: the first sentence opens the local uncertainty, criterion, evidence need, or constraint, and the second sentence narrows toward the resulting choice, boundary, or next move. Use two sentences for each `[HIGH]` paragraph; use one or two sentences for `[LOW]` paragraphs depending on the task.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph may use \"I\" for a direct task move, or use a task object as subject, and should end with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action.",
    "9. For ordinary or harder tasks, several mapped paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last mapped paragraph may use \"I\" for a direct task move, or use a task object as subject, and should finish on the concrete result, artifact, boundary, code path, calculation, recommendation, or next action inside the task domain.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_SPLIT_EXPANDED_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_MAPPED_EXPANDED_PROMPT.replace(
    "The trace should read like inner task reasoning: more informative than a terse outline, less expansive than a long chain of thought.",
    "The trace should read like inner task reasoning rather than a compressed summary: it should be more informative than a terse outline while staying focused on the task.",
).replace(
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count.",
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count; if adjacent skeleton lines are related, still give each line its own paragraph because the separation records the exploration path.",
).replace(
    "Each mapped paragraph should develop only its corresponding skeleton line as a small exploration from unknown to known: the first sentence opens the local uncertainty, criterion, evidence need, or constraint, and the second sentence narrows toward the resulting choice, boundary, or next move.",
    "Each mapped paragraph should develop only its corresponding skeleton line as a small exploration from unknown to known: the first sentence opens why this local step is uncertain, important, or constrained, and the second sentence narrows toward the resulting choice, boundary, or next move.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_CLEAN_SKELETON_DENSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_SPLIT_EXPANDED_PROMPT.replace(
    "3. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
    "3. After the number, write exactly this pattern: `[TAG][HIGH/LOW]` followed by a compact task-action phrase of 4-9 words, e.g. `1. [PLAN][HIGH] Identify task form and constraints`. Keep first-person wording for the matching reason paragraph, not the skeleton line.",
).replace(
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and evidence boundaries in the matching reason paragraph.",
    "6. Keep skeleton lines as brief labels only; put local uncertainty, criteria, caveats, comparisons, and evidence boundaries in the matching reason paragraph.",
).replace(
    "Each mapped paragraph should develop only its corresponding skeleton line as a small exploration from unknown to known: the first sentence opens why this local step is uncertain, important, or constrained, and the second sentence narrows toward the resulting choice, boundary, or next move.",
    "Each mapped paragraph should develop only its corresponding skeleton line as a small exploration from unknown to known. The first sentence names the local unknown, criterion, evidence need, or constraint and why it matters; the second sentence compares the relevant boundary or alternative and narrows toward the resulting choice, boundary, or next move.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_TRANSITION_DENSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_CLEAN_SKELETON_DENSE_PROMPT.replace(
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count; if adjacent skeleton lines are related, still give each line its own paragraph because the separation records the exploration path.",
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count; if adjacent skeleton lines are related, still give each line its own paragraph because the separation records the exploration path. Start each mapped paragraph with a fresh local transition such as `I first`, `I then`, `Next`, `At this point`, `The key constraint`, or `This leaves`, choosing wording that fits the task.",
).replace(
    "The first sentence names the local unknown, criterion, evidence need, or constraint and why it matters; the second sentence compares the relevant boundary or alternative and narrows toward the resulting choice, boundary, or next move.",
    "The first sentence names the local unknown, criterion, evidence need, or constraint and why it matters; the second sentence compares the relevant boundary or alternative and narrows toward the resulting choice, boundary, or next move. The paragraph should add task-specific reasoning beyond restating the skeleton label.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_INDEXED_PARAGRAPH_DENSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_CLEAN_SKELETON_DENSE_PROMPT.replace(
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count; if adjacent skeleton lines are related, still give each line its own paragraph because the separation records the exploration path.",
    "Keep exactly one blank line between these mapped paragraphs, so the reason block has the same paragraph count as the skeleton line count; if adjacent skeleton lines are related, still give each line its own paragraph because the separation records the exploration path. Begin each reason paragraph with the same number as its skeleton line followed by a period, then continue with ordinary first-person or task-object reasoning.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_INDEXED_OBJECT_DENSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_INNER_MONOLOGUE_VERB_CLOSE_OBJECT_INDEXED_PARAGRAPH_DENSE_PROMPT.replace(
    "If a first-person sentence uses ensure, check, verify, or confirm, the object should be a task object: a fact, source, calculation, code path, boundary, artifact, recommendation, user need, or implementation detail.",
    "If a first-person sentence uses ensure, check, verify, or confirm, the object should be a task object: a fact, source, calculation, code path, boundary, artifact, recommendation, user need, or implementation detail. Prefer the domain noun for the current work, such as guide, class design, query, code path, calculation, recommendation, evidence boundary, or deployment option, rather than generic words like response, answer, output, trace, paragraph, or message.",
).replace(
    "The last mapped paragraph may use \"I\" for a direct task move, or use a task object as subject, and should finish on the concrete result, artifact, boundary, code path, calculation, recommendation, or next action inside the task domain.",
    "The last mapped paragraph may use \"I\" for a direct task move, or use a task object as subject, and should finish on the concrete result, artifact, boundary, code path, calculation, recommendation, or next action inside the task domain. When the last move refines or verifies something, name the domain object being refined or verified rather than a generic response, answer, output, trace, paragraph, or message.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_NOUN_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_SHORT_PROMPT.replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.\n14. The last skeleton line should be a compact task-noun phrase plus a domain action, matching the last paragraph's subject and verb.",
).replace(
    "The last paragraph must be one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a claim about fit, suitability, completeness, or faithfulness.",
    "The last paragraph must be one simple sentence. It should start with the task noun used in the last skeleton line, use a concrete domain verb, and end on the selected result, artifact, boundary, code path, calculation, recommendation, or next action. Keep the sentence as subject + verb + complement, without a trailing purpose clause or meta-quality clause.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_NOUN_CLOSE_TIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_NOUN_CLOSE_PROMPT.replace(
    "The last paragraph must be one simple sentence. It should start with the task noun used in the last skeleton line, use a concrete domain verb, and end on the selected result, artifact, boundary, code path, calculation, recommendation, or next action. Keep the sentence as subject + verb + complement, without a trailing purpose clause or meta-quality clause.",
    "The last paragraph must be one simple sentence with a task noun as subject, a concrete domain verb as predicate, and a selected result, artifact, boundary, code path, calculation, recommendation, or next action as complement.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_NOUN_CLOSE_BARE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_SHORT_PROMPT.replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.\n14. The last skeleton line should name a task object and its domain action in compact task language.",
).replace(
    "The last paragraph must be one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a claim about fit, suitability, completeness, or faithfulness.",
    "The last paragraph must be one sentence only. Start directly with a task object, use a concrete domain action as the main verb, and stop after the concrete result, artifact, boundary, code path, calculation, recommendation, or next action. The sentence should be bare task content rather than a purpose, quality, or fit claim.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_ACTION_PRESENT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_NOUN_CLOSE_BARE_PROMPT.replace(
    "Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided exchange is context; return the reasoning trace only.",
    "Return exactly two blocks, in this order: `<skeleton>` and `<reason>`. The provided exchange is context; write the reasoning trace only.",
).replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action. Keep the closing synthesis compact. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual answer type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and deliverable. Earlier turns, examples, schemas, demonstrations, and prior outputs are context only. Let the trace end as a direct task conclusion, using the subject matter itself rather than commentary about the trace.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the solution value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim. Prefer domain objects as grammatical subjects: the query, code path, recommendation, boundary, artifact structure, calculation, diagnosis, or next action. Keep the closing synthesis compact. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual conclusion type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior material are context only. Let the trace end inside the task domain as the last step of the work.",
).replace(
    "- [PLAN]: understand the request, constraints, and answer shape.",
    "- [PLAN]: understand the request, constraints, and solution shape.",
).replace(
    "9. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.",
    "9. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or last prioritization.",
).replace(
    "10. Also add information-density steps when the value of the answer depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
    "10. Also add information-density steps when the value of the solution depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
).replace(
    "7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer direction.",
    "7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the chosen direction.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph must be one sentence only. Start directly with a task object, use a concrete domain action as the main verb, and stop after the concrete result, artifact, boundary, code path, calculation, recommendation, or next action. The sentence should be bare task content rather than a purpose, quality, or fit claim.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, evidence target, or reflective boundary. The last paragraph should follow the last skeleton line as a compact task action: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
).replace(
    "13. Always write the final closing tag `</reason>` after the last reasoning paragraph.",
    "13. Always write `</reason>` after the last reasoning paragraph.",
).replace(
    "14. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.",
    "14. Close `</skeleton>` immediately after the last skeleton line. Then write `<reason>`, the reasoning paragraphs, and `</reason>`.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_ACTION_PRESENT_TIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_ACTION_PRESENT_PROMPT.replace(
    "The last paragraph should follow the last skeleton line as a compact task action: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
    "The last paragraph should follow the last skeleton line as one compact present-tense task sentence: task object, concrete domain verb, and selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_FLOW_NOMETA_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_ACTION_PRESENT_PROMPT.replace(
    "After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.",
    "Write the complete `<skeleton>...</skeleton>` block first, then the complete `<reason>...</reason>` block.",
).replace(
    "Keep the closing synthesis compact. ",
    "Keep synthesis compact. ",
).replace(
    "answer type",
    "deliverable type",
).replace(
    "For refusal or safety-boundary answers, derive the boundary and stance taken for the current request.",
    "For refusal or safety-boundary cases, derive the boundary and stance taken for the current request.",
).replace(
    "Let the trace end inside the task domain as the last step of the work.",
    "Keep the trace inside the task domain throughout, including synthesis.",
).replace(
    "or last prioritization.",
    "or prioritization.",
).replace(
    "14. The last skeleton line should name a task object and its domain action in compact task language.",
    "14. Use compact task language when naming task objects and domain actions.",
).replace(
    "the ending should state the task result through a domain action when it follows from the trace.",
    "task synthesis should state the task result through a domain action when it follows from the trace.",
).replace(
    "The last paragraph should follow the last skeleton line as a compact task action: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
    "Use compact task actions where useful: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
).replace(
    "13. Always write `</reason>` after the last reasoning paragraph.",
    "13. Finish the reason block with `</reason>`.",
).replace(
    "The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.",
    "The reasoning paragraphs must be inside `<reason>` and `</reason>`.",
).replace(
    "14. Close `</skeleton>` immediately after the last skeleton line. Then write `<reason>`, the reasoning paragraphs, and `</reason>`.",
    "14. End the skeleton block with `</skeleton>`, then write `<reason>`, the reasoning paragraphs, and `</reason>`.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_DOMAIN_FIRST_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_FLOW_NOMETA_PROMPT.replace(
    "Keep the trace inside the task domain throughout, including synthesis.",
    "Keep the trace inside the task domain throughout, including synthesis. Write in domain-first narration: clauses should start from task entities, evidence, constraints, alternatives, calculations, code paths, boundaries, artifact parts, or next actions. Prefer domain verbs such as identifies, compares, derives, selects, limits, maps, implements, returns, recommends, proceeds, names, or explains.",
).replace(
    "12. Do not repeat the same tag or sentence frame three times in a row.",
    "12. Do not repeat the same tag or sentence frame three times in a row.\n13. Each skeleton line should name a task object, constraint, alternative, calculation, boundary, artifact part, or next action whenever possible.",
).replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "14. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
).replace(
    "14. Use compact task language when naming task objects and domain actions.",
    "15. Use compact task language when naming task objects and domain actions.",
).replace(
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps.",
    "2. Each paragraph should be compact but informative. Use one sentence for plain setup or task synthesis; use two sentences for most `[HIGH]`, comparison, reflection, technical, numerical, safety, or recommendation steps. Let the paragraph's main subject be a task entity, evidence source, constraint, alternative, calculation, boundary, artifact part, or next action whenever natural.",
).replace(
    "Keep narration centered on task objects and decisions.",
    "Keep narration centered on task objects and decisions. Main predicates should describe domain movement rather than drafting quality: identifies, compares, narrows, selects, maps, limits, implements, returns, names, recommends, or proceeds.",
).replace(
    "Use compact task actions where useful: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action.",
    "Use compact task actions where useful: task object, concrete domain verb, selected result, artifact, boundary, code path, calculation, recommendation, or next action. Task synthesis should sound like ordinary domain reasoning, with no detached wrap-up sentence.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_JSON_NOMETA_PROMPT = '''You reconstruct a structured reasoning trace for the latest exchange in a dialogue.

Return only one valid JSON object with the top-level key "steps". Do not use markdown, XML, numbering, comments, examples, or text outside JSON.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Keep the latest exchange's task outcome stable: recommendation, boundary decision, factual conclusion type, artifact direction, decisive conclusion, or requested action. A short exchange note may appear before the full dialogue; use it only to identify the current request and task form. Earlier turns, examples, schemas, demonstrations, and prior material are context only.

Each item in "steps" must be an object with exactly these keys:
- "tag": one of "PLAN", "RETR", "INFR", "EVAL", "SUMM", "BRCH", "RFLX", "BTRK".
- "importance": "HIGH" or "LOW".
- "action": a short task-specific action phrase.
- "reason": one compact reasoning paragraph.

Step rules:
1. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.
2. Mark at least half the steps "HIGH".
3. Add diagnostic, branch, reflection, or revision steps only when earned by real difficulty: ambiguity, competing approaches, missing information, uncertainty risk, an implementation tradeoff, safety or policy boundary, numerical interpretation, or a failed path.
4. The first step should identify the current request and task form using task-level wording.
5. Use compact task language when naming task objects, constraints, alternatives, calculations, boundaries, artifact parts, and next actions.
6. Do not repeat the same tag or sentence frame three times in a row.

Reason rules:
1. Each "reason" should be one or two compact sentences.
2. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and task-relevant terms needed for faithful reasoning; make the connection through criteria, evidence boundaries, and transitions rather than surface restatement.
3. For artifact-generation tasks, reason toward the artifact or content appropriate to the current request rather than an earlier example or only the process for creating it. For refusal or safety-boundary cases, derive the boundary and stance taken for the current request.
4. Build toward the task stance progressively. Earlier steps should develop criteria, constraints, alternatives, and evidence targets; synthesis should state the task result through a domain action when it follows from the trace.
5. For "EVAL", "BRCH", "RFLX", and "BTRK", name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven or information-density-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, or uncertainty boundary keeps it honest.
7. Keep reasoning task-specific, but avoid broad background, extra examples, and comments about this protocol.
'''


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OBJECT_ACTION_CLOSE_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_TASK_VERB_CLOSE_SHORT_PROMPT.replace(
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.",
    "13. In skeleton lines, name the reasoning operation, selection criterion, stance, or deliverable shape rather than reusing surface wording from the provided exchange.\n14. The last skeleton line should be an object-level action rather than a wrap-up label: name the task artifact, boundary, calculation, recommendation, code path, or next action and the domain operation it performs.",
).replace(
    "The last paragraph must be one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action, not a claim about fit, suitability, completeness, or faithfulness.",
    "The last paragraph must mirror the last skeleton line as one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action. Use ordinary present-tense task language rather than discourse markers.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OBJECT_ACTION_CLOSE_DIRECT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_TARGET_HEADER_OBJECT_ACTION_CLOSE_PROMPT.replace(
    "The last paragraph must mirror the last skeleton line as one sentence only. Its grammatical subject should be a task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action. Use ordinary present-tense task language rather than discourse markers.",
    "The last paragraph must mirror the last skeleton line as one sentence only. Its grammatical subject should be a named task object, and its main verb should be a domain action such as gives, selects, recommends, defines, returns, uses, limits, proceeds, names, or explains. End with the concrete result, artifact, boundary, code path, calculation, recommendation, or next action in compact present-tense task language.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_LIGHT_PROMPT.replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.\n10. In the final paragraph, include a compact consistency check that the developed trace still points to the assistant turn's core endpoint.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_FRONTLOAD_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_PROMPT.replace(
    "1. Put one numbered line per step inside `<skeleton>`.",
    "1. Put one numbered line per step inside `<skeleton>`.\n2. The first skeleton line should identify the user request and the assistant turn's endpoint type, such as recommendation, refusal stance, factual conclusion, artifact content, or requested action, using task-level wording.",
).replace(
    "2. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
    "3. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.",
).replace(
    "3. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.",
    "4. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.",
).replace(
    "4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.",
    "5. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.",
).replace(
    "5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
    "6. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.",
).replace(
    "6. Mark at least half the lines `[HIGH]`.",
    "7. Mark at least half the lines `[HIGH]`.",
).replace(
    "7. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
    "8. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.",
).replace(
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.",
    "9. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.",
).replace(
    "9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
    "10. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.",
).replace(
    "10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
    "11. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
).replace(
    "11. Do not repeat the same tag or sentence frame three times in a row.",
    "12. Do not repeat the same tag or sentence frame three times in a row.",
).replace(
    "12. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
    "13. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_ARTIFACT_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_ARTIFACT_PROMPT.replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.\n10. In the final paragraph, include a compact consistency check that the developed trace still points to the assistant turn's core endpoint.",
)


SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_STRONG_PROMPT = SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_ENDPOINT_GUARD_CHECK_PROMPT.replace(
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action.",
    "Preserve the assistant turn's core endpoint: its final recommendation, refusal decision, factual answer, artifact content, decisive conclusion, or requested action. If nearby endpoints are plausible, select reasoning criteria that lead to the endpoint actually settled by the assistant turn rather than an adjacent task or generic process.",
).replace(
    "10. In the final paragraph, include a compact consistency check that the developed trace still points to the assistant turn's core endpoint.",
    "10. In the final paragraph, include a compact consistency check that the developed trace still points to the assistant turn's core endpoint, including the same response stance and task-specific endpoint type.",
)


SSR_PLUS_STRUCT_BALANCED_LEGACY_SOFT_PROMPT = '''You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. Do not output the final assistant answer.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth only where the task calls for it.

### Tags

Use these tags only:

- [PLAN]: understand the request, constraints, and answer shape.
- [RETR]: bring in needed facts, context, or remembered information.
- [INFR]: infer, calculate, transform, or connect task details.
- [EVAL]: check a risky assumption, calculation, source, or consistency point.
- [SUMM]: integrate the reasoning into the answer direction.
- [BRCH]: compare plausible approaches before selecting one.
- [RFLX]: pause on an important judgment and name what keeps it disciplined.
- [BTRK]: revise a concrete earlier path after a check fails.

### Skeleton Rules

1. Put one numbered line per step inside `<skeleton>`.
2. Each line must look like `1. [TAG][HIGH] Short task-specific action.` or `1. [TAG][LOW] Short task-specific action.`
3. Use 5-7 steps for simple tasks, 7-9 for ordinary tasks, and 9-11 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.
4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.
5. Mark at least half the lines `[HIGH]`.
6. Add 0-2 diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.
7. Add 0-2 importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.
8. Do not repeat the same tag or sentence frame three times in a row.

### Reason Rules

1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be 1-2 compact sentences. Use one sentence for setup or wrap-up; use two when the step contains a real inference, comparison, check, or reflection.
3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Avoid distinctive final-answer phrases unless they are necessary task terms.
4. Do not front-load the final verdict, recommendation, exact conclusion, or solution phrase. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the last paragraph may integrate the answer direction without restating the final answer.
5. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.
7. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
8. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
9. Close `</skeleton>` immediately after the final skeleton line and close `</reason>` immediately after the final reasoning paragraph.
'''


SSR_PLUS_STRUCT_BALANCED_NO_OPEN_GUARD_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "\nAfter closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.\n\n",
    "\n",
)


SSR_PLUS_STRUCT_BALANCED_ECHO_CONTROL_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories or ordinary terms when needed for alignment, but avoid distinctive final-answer phrases unless they are necessary task terms.",
    "3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories, user-provided terms, or neutral paraphrases when needed for alignment, but avoid distinctive final-answer phrases, named conclusions, exact values, and answer-only wording unless they are necessary task terms.",
).replace(
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, or exact wording, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, exact wording, or exact value, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
)


SSR_PLUS_STRUCT_BALANCED_ECHO_LIGHT_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories or ordinary terms when needed for alignment, but avoid distinctive final-answer phrases unless they are necessary task terms.",
    "3. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories, user-provided terms, or neutral paraphrases when needed for alignment, but avoid distinctive final-answer phrases and answer-only wording unless they are necessary task terms.",
).replace(
    "7. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer space.",
    "7. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer space. For calculation, code/configuration, title, translation, and rewrite/compression tasks, keep extra detail on the operation, compatibility check, selection criteria, and boundary conditions rather than assembling the final numeric result, literal, title, or compressed wording inside the trace.",
).replace(
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, or exact wording, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, exact wording, or final deliverable, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
)


SSR_PLUS_STRUCT_BALANCED_VALUE_GUARD_PROMPT = SSR_PLUS_STRUCT_BALANCED_ECHO_CONTROL_PROMPT.replace(
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, exact wording, or exact value, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, exact wording, code literal, package or version, title, or exact value, map the underlying criteria to the response stance in generic terms instead of echoing that phrase. If a value, name, or literal is not supplied by the user, describe the derivation or selection check without writing that final item.",
)


SSR_PLUS_STRUCT_BALANCED_DIRECT_Q_PROMPT = '''You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. Do not output the final assistant answer.

After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Preserve the successful balanced style: task-grounded, concise, and answer-aware without copying distinctive final-answer wording.

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
2. Each line must follow this exact shape: `n. [TAG][HIGH|LOW] Short task-specific action sentence.`
3. Use 6-8 steps for simple tasks, 7-9 for ordinary tasks, and 9-11 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.
4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.
5. Do not put explanatory second sentences in skeleton lines; put all caveats, criteria, and checks in the matching reason paragraph.
6. Mark at least half the lines `[HIGH]`.
7. Add diagnostic steps only when earned by real ambiguity, competing approaches, missing information, verification risk, or a failed path.
8. Add reflection or importance steps when a judgment affects the main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, audience-facing priority, or critical proof/coding invariant.
9. For complex or high-stakes steps, represent the deeper thinking with one focused diagnostic or reflection step rather than many thin checklist steps.
10. Do not repeat the same tag or sentence frame three times in a row.
11. Do not copy the final verdict, recommendation, or solution wording in skeleton lines; describe the check, selection criterion, or response shape instead.

### Reason Rules

1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.
3. When the task is complex, difficult, ambiguous, high-stakes, or depends on a key conclusion, add more information density inside the relevant paragraph: name the uncertainty, compare alternatives, verify the constraint, or explain the failed path and correction.
4. Do not add depth from a rigid minimum length rule. Add it only where the task difficulty, evidence standard, audience risk, or final prioritization needs it.
5. Reconstruct the reasoning process rather than copying or paraphrasing the final answer. Use task-level answer categories or ordinary terms when needed for alignment, but avoid distinctive final-answer phrases unless they are necessary task terms.
6. Do not front-load the final verdict, recommendation, exact conclusion, or solution phrase. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs should narrow toward the response stance and answer shape without restating the final answer.
7. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
8. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer space.
9. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
10. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
11. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.
12. Always write the final closing tag `</reason>` after the last reasoning paragraph.
13. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.
'''


SSR_PLUS_STRUCT_BALANCED_DIRECT_R_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping distinctive answer wording controlled.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping distinctive answer wording controlled. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the answer's value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.",
).replace(
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.\n"
    "9. For nontrivial tasks, include at least one diagnostic step or one importance step, and usually no more than three total.",
    "8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.\n"
    "9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.\n"
    "10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.",
).replace(
    "10. Do not repeat the same tag or sentence frame three times in a row.\n"
    "11. Do not copy the final verdict, recommendation, or solution wording in skeleton lines; describe the check, selection criterion, or response shape instead.",
    "11. Do not repeat the same tag or sentence frame three times in a row.\n"
    "12. Do not copy the final verdict, recommendation, or solution wording in skeleton lines; describe the check, selection criterion, or response shape instead.",
).replace(
    "6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.",
    "6. For importance-driven or information-density-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, or uncertainty boundary keeps it honest.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, or verification target. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.",
)


SSR_PLUS_STRUCT_BALANCED_DIRECT_R2_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping distinctive answer wording controlled.",
    "The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, preferably inside the matching reason paragraph rather than by adding extra skeleton steps, while keeping distinctive answer wording controlled.",
).replace(
    "6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, or audience need keeps it honest.",
    "6. For importance-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, method choice, uncertainty boundary, or claim-framing risk keeps it honest.",
).replace(
    "7. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the answer space.",
    "7. When adding depth, prefer criteria and checks over answer leakage: discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, failure mode, or key conclusion pressure before narrowing the answer space.",
).replace(
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, or verification target. Do not add such a sentence when it would merely pad an obvious step.",
    "9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or brief reflection on why the step matters. Do not add such a sentence when it would merely pad an obvious step.",
)


SSR_PLUS_STRUCT_BALANCED_DIRECT_R3_PROMPT = SSR_PLUS_STRUCT_BALANCED_PROMPT.replace(
    "4. Do not front-load the final verdict, recommendation, exact conclusion, or solution phrase. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs should narrow toward the response stance and answer shape without restating the final answer.",
    "4. Do not front-load the final verdict, recommendation, exact conclusion, or solution phrase. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs should narrow toward the response stance and answer shape without restating the final answer. For advice, interpretation, creative framing, or value-judgment tasks, keep the first half especially criteria-facing rather than stance-facing.",
).replace(
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, or exact wording, map the underlying criteria to the response stance in generic terms instead of echoing that phrase.",
    "8. When the assistant's final answer contains a strong claim, concrete recommendation, named solution, or exact wording, map the underlying criteria to the response stance in generic terms instead of echoing that phrase. Delay any generic stance mapping until the relevant criteria or checks have been named.",
)


SSR_PLUS_STRUCT_BALANCED_JSON_PROMPT = '''You reconstruct a structured reasoning trace for a final assistant turn in a dialogue.

Return only one valid JSON object with the top-level key "steps". Do not use markdown, XML, numbering, comments, examples, or text outside JSON. Do not output the final assistant answer.

Each item in "steps" must have exactly these fields:
- "tag": one of "PLAN", "RETR", "INFR", "EVAL", "SUMM", "BRCH", "RFLX", "BTRK".
- "importance": "HIGH" or "LOW".
- "action": one short task-specific sentence describing the reasoning operation.
- "reason": one compact paragraph explaining that operation.

Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks. Mark at least half the steps HIGH.

Keep the trace moderately developed: more informative than a terse outline, less expansive than a long chain of thought. The reasoning should be task-specific and should align with the provided final assistant turn, while avoiding direct copying of distinctive final-answer wording, exact conclusions, or answer-only phrases.

Add diagnostic steps only when earned by real ambiguity, competing approaches, missing information, verification risk, or a failed path. Add reflection or importance steps when a judgment affects the main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, audience-facing priority, or a critical proof/coding invariant.

For each step, make the action concise and concrete. Put caveats, criteria, checks, and tradeoffs in the reason field, not in the action. Do not repeat the same tag or sentence frame three times in a row.

Each reason paragraph should normally be two substantive sentences for HIGH steps, comparison steps, verification steps, reflection steps, technical reasoning, numerical interpretation, recommendation choices, or safety/risk tradeoffs. LOW setup or wrap-up steps may use one sentence when the point is simple. Keep the average trace close to a moderately developed direct-render trace: compact, but usually not terse. For BRCH, EVAL, RFLX, and BTRK, name the actual uncertainty, tradeoff, evidence standard, or correction.

Before returning JSON, silently check that every reason paragraph maps to one action, the steps remain in reasoning order, and no field contains XML tags, markdown headings, placeholders, or prompt text.
'''


SSR_PLUS_STRUCT_BALANCED_JSON_DENSE_PROMPT = '''You reconstruct a structured reasoning trace for a final assistant turn in a dialogue.

Return only one valid JSON object with the top-level key "steps". Do not use markdown, XML, numbering, comments, examples, or text outside JSON. Do not output the final assistant answer.

Each item in "steps" must have exactly these fields:
- "tag": one of "PLAN", "RETR", "INFR", "EVAL", "SUMM", "BRCH", "RFLX", "BTRK".
- "importance": "HIGH" or "LOW".
- "action": one short task-specific sentence describing the reasoning operation.
- "reason": one moderately developed paragraph explaining that operation.

Use 5-6 steps for simple tasks, 6-7 for ordinary tasks, and 7-8 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks. Prefer fewer well-developed steps over many thin steps. Mark at least half the steps HIGH.

Keep the trace close to a moderately developed direct-render reasoning trace: compact but not terse, usually about five to seven reasoning paragraphs. Each HIGH paragraph should normally contain two or three substantive sentences. LOW setup or wrap-up paragraphs may use one sentence when the point is simple.

The trace must align with the provided final assistant turn while avoiding direct copying of distinctive final-answer wording, exact conclusions, or answer-only phrases. Earlier steps should develop criteria, constraints, alternatives, and checks; final steps may narrow toward the response stance without restating the final answer.

Add diagnostic steps only when earned by real ambiguity, competing approaches, missing information, verification risk, or a failed path. Add reflection or importance steps when a judgment affects the main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, audience-facing priority, or a critical proof/coding invariant.

For each step, make the action concise and concrete. Put caveats, criteria, checks, tradeoffs, and local validation in the reason field, not in the action. Do not repeat the same tag or sentence frame three times in a row.

For BRCH, EVAL, RFLX, and BTRK, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic statements that something is merely correct or important.

Before returning JSON, silently check that every reason paragraph maps to one action, the steps remain in reasoning order, and no field contains XML tags, markdown headings, or prompt text.
'''


SSR_PLUS_STRUCT_MID_PROMPT = SSR_PLUS_STRUCT_PROMPT.replace(
    "2. Use 6-10 steps for ordinary tasks and 8-14 steps for multi-part, math, coding, or long-form tasks.",
    "2. Use 7-10 steps for ordinary tasks and 8-12 steps for multi-part, math, coding, or long-form tasks.",
).replace(
    "1. Write one concise but substantive paragraph per skeleton step, in exactly the same order.\n"
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "3. Paragraphs should normally contain 1-3 sentences; avoid long-tail expansion.\n"
    "4. Do not number paragraphs or copy skeleton tags in the `<reason>` block.",
    "1. Write one compact but substantive paragraph per skeleton step, in exactly the same order.\n"
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "3. Paragraphs should normally contain exactly 2 sentences; keep them informative but avoid long-tail expansion.\n"
    "4. Separate consecutive reasoning paragraphs with exactly one blank line so each paragraph maps to one skeleton step.\n"
    "5. Do not number paragraphs or copy skeleton tags in the `<reason>` block.",
).replace(
    "5. Do not output the final assistant answer in this response.",
    "6. Do not output the final assistant answer in this response.",
)


SSR_PLUS_STRUCT_COMPACT_PROMPT = SSR_PLUS_STRUCT_PROMPT.replace(
    "2. Use 6-10 steps for ordinary tasks and 8-14 steps for multi-part, math, coding, or long-form tasks.",
    "2. Use 7-10 steps for ordinary tasks and 8-12 steps for multi-part, math, coding, or long-form tasks.",
).replace(
    "1. Write one concise but substantive paragraph per skeleton step, in exactly the same order.\n"
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "3. Paragraphs should normally contain 1-3 sentences; avoid long-tail expansion.\n"
    "4. Do not number paragraphs or copy skeleton tags in the `<reason>` block.",
    "1. Write one concise but substantive paragraph per skeleton step, in exactly the same order.\n"
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "3. Each paragraph should contain 1-2 compact sentences and at most 45 words; avoid long-tail expansion.\n"
    "4. Separate consecutive reasoning paragraphs with exactly one blank line so each paragraph maps to one skeleton step.\n"
    "5. Do not number paragraphs or copy skeleton tags in the `<reason>` block.",
).replace(
    "5. Do not output the final assistant answer in this response.",
    "6. Do not output the final assistant answer in this response.",
)


SSR_PLUS_STRUCT_STEPS_ONLY_PROMPT = SSR_PLUS_STRUCT_PROMPT.replace(
    "2. Use 6-10 steps for ordinary tasks and 8-14 steps for multi-part, math, coding, or long-form tasks.",
    "2. Use 7-10 steps for ordinary tasks and 8-12 steps for multi-part, math, coding, or long-form tasks.",
)


SSR_PLUS_STRUCT_C1_STEPS_PROMPT = SSR_PLUS_STRUCT_STEPS_ONLY_PROMPT


SSR_PLUS_STRUCT_C2_COMPACT_PROMPT = SSR_PLUS_STRUCT_C1_STEPS_PROMPT.replace(
    "1. Write one concise but substantive paragraph per skeleton step, in exactly the same order.",
    "1. Write one compact but substantive paragraph per skeleton step, in exactly the same order.",
)


SSR_PLUS_STRUCT_C3_EXACT2_PROMPT = SSR_PLUS_STRUCT_C2_COMPACT_PROMPT.replace(
    "3. Paragraphs should normally contain 1-3 sentences; avoid long-tail expansion.",
    "3. Paragraphs should normally contain exactly 2 sentences; keep them informative but avoid long-tail expansion.",
)


SSR_PLUS_STRUCT_C4_BLANKLINE_PROMPT = SSR_PLUS_STRUCT_C3_EXACT2_PROMPT.replace(
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.",
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate; separate consecutive reasoning paragraphs with exactly one blank line so each paragraph maps to one skeleton step.",
)


SSR_PLUS_STRUCT_C5_SEPARATE_RULE_PROMPT = SSR_PLUS_STRUCT_MID_PROMPT


SSR_PLUS_STRUCT_C6_LONG_STEPS_PROMPT = SSR_PLUS_STRUCT_C5_SEPARATE_RULE_PROMPT.replace(
    "2. Use 7-10 steps for ordinary tasks and 8-12 steps for multi-part, math, coding, or long-form tasks.",
    "2. Use 8-12 steps for ordinary tasks and 10-18 steps for multi-part, math, coding, or long-form tasks.",
)


SSR_PLUS_STRUCT_C7_DEVELOPED_PROMPT = SSR_PLUS_STRUCT_C6_LONG_STEPS_PROMPT.replace(
    "1. Write one compact but substantive paragraph per skeleton step, in exactly the same order.",
    "1. Write one developed paragraph per skeleton step, in exactly the same order.",
)


SSR_PLUS_STRUCT_C8_LONG_REASON_PROMPT = SSR_PLUS_STRUCT_C7_DEVELOPED_PROMPT.replace(
    "2. Each paragraph should explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "3. Paragraphs should normally contain exactly 2 sentences; keep them informative but avoid long-tail expansion.\n"
    "4. Separate consecutive reasoning paragraphs with exactly one blank line so each paragraph maps to one skeleton step.",
    "2. Separate consecutive reasoning paragraphs with exactly one blank line. The blank line is required because it marks the boundary between skeleton steps.\n"
    "3. Each paragraph should normally contain 3-5 sentences and should be substantive enough to explain the local operation, the relevant task constraint, and the check or transition that makes the next step appropriate.\n"
    "4. Keep the reasoning grounded in the question and dialogue context. Add concrete local analysis, verification, and transition logic rather than terse summaries.",
)


SSR_PLUS_STRUCT_C9_LONG_TEMPLATE_PROMPT = SSR_PLUS_STRUCT_C8_LONG_REASON_PROMPT.replace(
    "(one paragraph per skeleton step, in the same order, without numbering or labels)",
    "(one developed paragraph per skeleton step, separated by exactly one blank line, in the same order, without numbering or labels)",
)


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

1.  **Strict Line Format:** Each line must follow exactly: `n. [STEP TAG] <single-sentence skeleton under 20 tokens, matching input language>`
    * *Constraint:* Use exactly one space after the dot (`.`) and one space after the `[STEP TAG]`.
    * *Constraint:* Keep each skeleton line focused on one primary reasoning intent; no extra commentary.
2.  **Example Line:** `1. [PLAN] Analyze the user's request and define the goal.`
3.  **Draft Refinement Constraint:** Your generated skeleton must be based on the provided initial skeleton draft. You must elevate this draft into a significantly more refined and comprehensive reasoning pathway, explicitly covering broader thinking capabilities and possibilities (e.g., incorporating missing branches, reflections, or deeper analytical steps).
4.  **Content Invariance Constraint:** Skeleton sentences must describe the step’s intent at an abstract level and must not reveal specific outputs, intermediate results, concrete values, or final answer content (e.g., “calculate the ratio” rather than “calculate 0.5”).
5.  **Deliberative Construction Constraint:** Before writing the final skeleton lines, internally perform a rigorous iterative deliberation cycle—analysis, concise summarization, alternative exploration, reassessment, verification, reflection, and backtracking as needed—to ensure the step sequence is carefully reasoned. This constraint governs how you construct the skeleton (step selection and ordering), not how verbose the skeleton lines are.
6.  **Nonlinear Tagging and Step Diversity Constraint:** When additional reasoning capacity is meaningfully beneficial, use diverse step tags (e.g., BRCH, RETR, BTRK, RFLX) to represent nonlinear, branching, corrective, and reflective thinking patterns, rather than a purely linear pipeline. Tag choices should be purposeful (not decorative) and should strengthen the skeleton’s fidelity to a realistic reasoning workflow while remaining compliant with the Content Invariance Constraint.
7.  **Granularity Constraint:** Avoid composite actions in a single step. Each skeleton line should express one primary intent (e.g., “identify constraints” separately from “choose approach”). If a step contains multiple operations, split them into focused reasoning intents.

**II. The `<reason>` Block**

1.  **Goal:** Provide hidden chain-of-thought, private internal deliberations, or detailed intermediate computations based on steps listed in the `<skeleton>` block.
2.  **Structure:** The reasoning text must correspond directly to each step listed in the `<skeleton>` block, maintaining the **exact same order**.
3.  **Constraint:** For each skeleton intent, write one developed reasoning paragraph. Each paragraph should explain why the step is needed, what local inference is made, and how the inference is checked against the task constraints.
4.  **Formatting:** Write continuous, coherent reasoning text or paragraphs. Separate each step with '\n\n' for clarity.
5.  **Constraint:** **DO NOT** explicitly number the steps or use the step labels (e.g., `1.`, `[PLAN]` in the `<skeleton>` block) within this block.
6.  **Constraint:** Stop once each skeleton intent has been sufficiently justified and close `</reason>`.
7.  **Constraint:** **DO NOT** output the final assistant answer in this response.

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


SS_GEN_PROMPT = '''You are an expert AI assistant converting an existing reasoning trace into an SSR structural skeleton.

Your job is to analyze the provided INPUT, segment it into faithful reasoning operations, and output only a concise SSR skeleton. Do not write new reasoning, conclusions, titles, or explanations.

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

1. Segment the INPUT into coherent operations that actually appear in the trace, preserving the original logical order.
2. Output one numbered line per operation using exactly this format: `n. [TAG][HIGH/LOW] <short action sentence>`.
3. Choose one real tag from the tag list, then write `[HIGH]` or `[LOW]` with no space between the tag and marker.
4. Use `[HIGH]` for objectively hard, risky, or information-dense operations: ambiguity, competing approaches, calculations, verification, failed paths, central judgments, method choices, uncertainty boundaries, implementation tradeoffs, or paper-style claims.
5. Use `[LOW]` for routine setup, straightforward inference, simple retrieval, local transitions, or wrap-up.
6. Keep each action sentence short, task-specific, and operation-level. Name what the step does rather than merely restating the final result it obtains.
7. Prefer 6-12 lines when the INPUT supports that range. Merge tiny moves; split dense passages when one segment contains a distinct check, branch, reflection, or correction.
8. Do not repeat the same tag or sentence frame three times in a row.
9. Use the requested output language. If it is unspecified, match the language of the INPUT trace.
10. Every skeleton line must correspond to explicit reasoning or operations in the INPUT. Do not invent hidden steps, new facts, assumptions, or conclusions.
11. Do not correct, reinterpret, or alter the factual content of the INPUT; if the INPUT contains uncertainty or errors, reflect the operation as written.
12. Output only the formatted skeleton lines, with exactly one line break between lines. Do not include markdown fences, bullets, headings, or commentary.

Output in language: {lang}.

**INPUT:**
{input_text}'''
