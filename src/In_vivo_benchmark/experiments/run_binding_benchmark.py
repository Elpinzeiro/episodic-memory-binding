"""In-vivo binding benchmark: run the evaluation of Sec. 3 / App. B for one or more models.

For every benchmark item it asks the verification questions (5 trials each), keeps the item only if
every one is answered correctly in >= 3/5 trials, then asks the binding question under each prompting
strategy (5 trials each) and grades every answer with an LLM judge. All models are called through
OpenRouter (OPENROUTER_API_KEY in .env); native extended thinking is enabled for the models that
support it. See data/README.md for the full walk-through and the output format.

Reproduce the paper (all ten models, both datasets, all eleven strategies):

    python run_binding_benchmark.py --all-models -d all --temperature 0.3 \
        -v raw reason no_reason direct_hinted reason_hinted reason_firstletter \
           selfask selfask_modified_2 mc_raw mc_reason mc_no_reason

Results are written to ./binding_failures_{athletes,custom}_rebuttal_temp0.3/<model>/ in the current
folder, which is what src/In_vivo_benchmark/results/ reads. Runs are resumable: existing trial files are
skipped. The --rejudge-* / --reval-* options are the repair passes applied to the released outputs
(re-parse of unparsed answers, re-judge of judge errors, re-query of truncated answers).
"""
import json, os, re, shutil
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Any, Optional

# Thread-safe print
print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ===========================================================================
# SELF-ASK FEW-SHOT TEMPLATES
# ===========================================================================

SELFASK_FEWSHOT = """\
Question: When does monsoon season end in the state the area code 575 is located?
Are follow up questions needed here: Yes.
Follow up: Which state is the area code 575 located in?
Intermediate answer: The area code 575 is located in New Mexico.
Follow up: When does monsoon season end in New Mexico?
Intermediate answer: Monsoon season in New Mexico typically ends in mid-September.
So the final answer is: mid-September.

Question: What is the current official currency in the country where Ineabelle Diaz is a citizen?
Are follow up questions needed here: Yes.
Follow up: Which country is Ineabelle Diaz a citizen of?
Intermediate answer: Ineabelle Diaz is from Peurto Rico, which is in the United States of America.
Follow up: What is the current official currency in the United States of America?
Intermediate answer: The current official currency in the United States is the United States dollar.
So the final answer is: United States dollar.

Question: Where was the person who founded the American Institute of Public Opinion in 1935 born?
Are follow up questions needed here: Yes.
Follow up: Who founded the American Institute of Public Opinion in 1935?
Intermediate answer: George Gallup.
Follow up: Where was George Gallup born?
Intermediate answer: George Gallup was born in Jefferson, Iowa.
So the final answer is: Jefferson.

Question: What language is used by the director of Tiffany Memorandum?
Are follow up questions needed here: Yes.
Follow up: Who directed the movie called Tiffany Memorandum?
Intermediate answer: Sergio Grieco.
Follow up: What language is used by Sergio Grieco?
Intermediate answer: Sergio Grieco speaks Italian.
So the final answer is: Italian.

Question: What is the sports team the person played for who scored the first touchdown in Superbowl 1?
Are follow up questions needed here: Yes.
Follow up: Which player scored the first touchdown in Superbowl 1?
Intermediate answer: Max McGee.
Follow up: Which sports team did Max McGee play for?
Intermediate answer: Max McGee played for the Green Bay Packers.
So the final answer is: Green Bay Packers.

Question: The birth country of Jayantha Ketagoda left the British Empire when?
Are follow up questions needed here: Yes.
Follow up: What is the birth country of Jayantha Ketagoda?
Intermediate answer: Sri Lanka.
Follow up: When did Sri Lanka leave the British Empire?
Intermediate answer: Sri Lanka left the British Empire on February 4, 1948.
So the final answer is: February 4, 1948.

Question: {question}
Are follow up questions needed here:"""

SELFASK_BIND_FEWSHOT = """\
Question: List ALL teams that LeBron James played for between 2003 and 2023. Provide your answer as a JSON list like ["a","b"].
Are follow up questions needed here: Yes.
Follow up: Which team did LeBron James play for at the start of this period, around 2003?
Intermediate answer: LeBron James was drafted by the Cleveland Cavaliers in 2003 and played there until 2010.
Follow up: Did LeBron James move to another team after the Cleveland Cavaliers?
Intermediate answer: Yes, he joined the Miami Heat in 2010 and played there until 2014.
Follow up: Did he play for any other teams after the Miami Heat and before 2023?
Intermediate answer: He returned to the Cleveland Cavaliers in 2014, then joined the Los Angeles Lakers in 2018, where he remained through 2023.
Follow up: Are there any other teams LeBron James played for in this period that I might be missing?
Intermediate answer: No, those are all the teams: Cleveland Cavaliers, Miami Heat, and Los Angeles Lakers.
So the final answer is: ["Cleveland Cavaliers", "Miami Heat", "Los Angeles Lakers"]

Question: List ALL teams that Zlatan Ibrahimović played for between 2004 and 2020. Provide your answer as a JSON list like ["a","b"].
Are follow up questions needed here: Yes.
Follow up: Which team was Zlatan Ibrahimović playing for around 2004?
Intermediate answer: He left Ajax in 2004 and was at Juventus from 2004 to 2006.
Follow up: Where did he go after Juventus?
Intermediate answer: He moved to Inter Milan in 2006 and stayed until 2009.
Follow up: Which team did he join after Inter Milan?
Intermediate answer: He joined Barcelona in 2009, then moved to AC Milan on loan in 2010.
Follow up: Did he play for other teams between 2010 and 2020?
Intermediate answer: Yes, he joined Paris Saint-Germain in 2012, then Manchester United in 2016, then LA Galaxy in 2018, and returned to AC Milan in 2020.
So the final answer is: ["Juventus", "Inter Milan", "Barcelona", "AC Milan", "Paris Saint-Germain", "Manchester United", "LA Galaxy"]

Question: List ALL Grand Slam tennis tournaments that Rafael Nadal won in 2010. Provide your answer as a JSON list like ["a","b"].
Are follow up questions needed here: Yes.
Follow up: Did Rafael Nadal win the Australian Open in 2010?
Intermediate answer: No, Roger Federer won the 2010 Australian Open.
Follow up: Did Rafael Nadal win the French Open in 2010?
Intermediate answer: Yes, Nadal won the 2010 French Open.
Follow up: Did Rafael Nadal win Wimbledon in 2010?
Intermediate answer: Yes, Nadal won Wimbledon in 2010.
Follow up: Did Rafael Nadal win the US Open in 2010?
Intermediate answer: Yes, Nadal won the 2010 US Open, completing a career Grand Slam.
Follow up: Are there any Grand Slams I missed?
Intermediate answer: No, there are only four Grand Slams per year. Nadal won three of them in 2010: French Open, Wimbledon, and US Open.
So the final answer is: ["French Open", "Wimbledon", "US Open"]

Question: List ALL countries visited by George W. Bush between June 14 - June 16 2001. Provide your answer as a JSON list like ["a","b"].
Are follow up questions needed here: Yes.
Follow up: Which country did George W. Bush visit on June 14–15?
Intermediate answer: Sweden.
Follow up: Which country did he visit on June 15–16?
Intermediate answer: Poland.
Follow up: Which country did he visit on June 16?
Intermediate answer: Slovenia.
Follow up: Did he visit Belgium within June 14–16?
Intermediate answer: Belgium was visited June 13–14, so it partially overlaps June 14.
Follow up: Should Belgium be included given the date range includes June 14?
Intermediate answer: Yes, since June 14 is within the specified range.
So the final answer is: ["Belgium", "Sweden", "Poland", "Slovenia"]

Now answer the next question in exactly this format:

Are follow up questions needed here: Yes or No.
If Yes, write one or more lines in the form:
Follow up: ...
Intermediate answer: ...
Finally end with:
So the final answer is: ["item1", "item2"]

Question: {question}
Provide the complete answer now."""

SELFASK_TEMPLATES = {
    "selfask":            SELFASK_FEWSHOT,
    "selfask_modified_2": SELFASK_BIND_FEWSHOT,
}

# All extended variants (list-based binding only, no MC).
# This is the *pool* of runnable extended variants; which ones actually run in a
# given invocation is decided by --variants (see run_evaluation_experiment).
# EXTENDED_VARIANTS = ["selfask", "selfask_modified_2", "reason_hinted", "reason_firstletter"]
EXTENDED_VARIANTS = ["direct_hinted", "reason_hinted","reason_firstletter","selfask", "selfask_modified_2"]

# ===========================================================================
# MODEL REASONING CONFIG
# ===========================================================================

# Stable folder labels from the original experiments mapped to current
# OpenRouter model slugs. The dictionary key controls the output directory;
# the value is used only for the API request.
MODEL_ALIASES = {
    "anthropic_claude-opus-4.6": "anthropic/claude-opus-4.6",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "openai_gpt-5.4": "openai/gpt-5.4",
    "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
    "google_gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "x-ai_grok-4.20": "x-ai/grok-4.20",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
    "deepseek-chat": "deepseek/deepseek-chat",
    "mistralai_mistral-large-2512": "mistralai/mistral-large-2512",
}


def resolve_openrouter_model(model_name: str) -> str:
    """Resolve a legacy/storage model label to an OpenRouter API slug.

    Known legacy labels are mapped explicitly so results continue to be saved
    under the original directory names. Fully qualified OpenRouter slugs are
    passed through unchanged. The fallback preserves the script's old behavior
    for ad-hoc model names.
    """
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]

    if "/" in model_name:
        return model_name

    model_lower = model_name.lower()
    if "deepseek" in model_lower:
        return f"deepseek/{model_name}"
    if "llama" in model_lower:
        return f"meta-llama/{model_name}"
    if "qwen" in model_lower:
        return f"qwen/{model_name}"
    if "mistral" in model_lower or "mixtral" in model_lower:
        return f"mistralai/{model_name}"
    if "gemini" in model_lower:
        return f"google/{model_name}"
    if "claude" in model_lower:
        return f"anthropic/{model_name}"
    if "gpt" in model_lower or model_lower.startswith(("o1", "o3", "o4")):
        return f"openai/{model_name}"
    return model_name


# Models that support extended reasoning via OpenRouter's unified API
# Using {"enabled": True} which works universally across all reasoning models
REASONING_SUPPORTED_PATTERNS = [
    # Anthropic (Claude 3.7+, Claude 4+)
    "claude-3.7", "claude-sonnet-4", "claude-opus-4", "claude-haiku-4",
    # OpenAI (o-series, GPT-5+)
    "gpt-5", "o1", "o3", "o4",
    # Google Gemini (2.5+, 3+)
    "gemini-2.5", "gemini-3",
    # DeepSeek
    "deepseek-r1", "deepseek-reasoner",
    # Qwen thinking models
    "qwen3", "qwen-thinking",
    # xAI Grok (3+, 4+)
    "grok-3", "grok-4",
    # Mistral Magistral (reasoning-specific models only)
    "magistral",
]

# Models that explicitly do NOT support reasoning (to avoid confusion)
REASONING_NOT_SUPPORTED = [
    "mistral-large", "mistral-medium", "mistral-small",  # Use magistral instead
    "gpt-4o", "gpt-4-turbo",  # Use gpt-5 or o-series
    "claude-3.5",  # Use claude-3.7+ or claude-4+
    "gemini-2.0", "gemini-1.5",  # Use gemini-2.5+
]


def model_supports_reasoning(model_name: str) -> bool:
    """Check if the resolved OpenRouter model supports extended reasoning."""
    model_lower = resolve_openrouter_model(model_name).lower()

    # Check explicit non-support first
    for pattern in REASONING_NOT_SUPPORTED:
        if pattern in model_lower:
            return False

    # Check if supported
    for pattern in REASONING_SUPPORTED_PATTERNS:
        if pattern in model_lower:
            return True

    return False


def build_reasoning_param(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Build the reasoning parameter dict for OpenRouter API.
    Uses the universal {"enabled": True} approach.
    """
    if model_supports_reasoning(model_name):
        return {"enabled": True}
    return None


# Module-level so re-query passes (see reval_truncated_answers) reproduce the
# original generation conditions verbatim instead of a paraphrase of them.
# The benchmark files shipped with the repo.
BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmark"

SYSTEM_PROMPT = (
    "You must answer ONLY using your internal parametric knowledge from training. "
    "You cannot access external information, web search, or any outside sources. "
    "If you are uncertain or the information is not clearly in your training data, "
    "you must honestly say 'I don't have reliable information about this', AVOID RANDOM guessing!!"
)


# ===========================================================================
# SELF-ASK HELPERS
# ===========================================================================

def build_selfask_prompt(question: str, variant: str) -> str:
    return SELFASK_TEMPLATES[variant].format(question=question)

# ---------------------------------------------------------------------------
# ANSWER EXTRACTION
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_JSON_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
_FINAL_MARKER_RE = re.compile(
    r"(?:so\s+the\s+final\s+answer\s+is|final\s+answer|the\s+answer\s+is)\s*:?\s*",
    re.IGNORECASE,
)
_DEGENERATE_ANSWER_RE = re.compile(r"^(?:`+|json|\[\s*\]|\{\s*\}|-+|\.+|,+)$", re.IGNORECASE)

# Final answers baked into SELFASK_FEWSHOT. If extraction returns one of these
# for a list-binding item, the model echoed the exemplars instead of answering.
FEWSHOT_FINAL_ANSWERS = {
    "mid-september", "united states dollar", "jefferson", "italian",
    "green bay packers", "february 4, 1948",
}


def _strip_wrappers(text: str) -> str:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?", "", stripped)
    stripped = re.sub(r"\r?\n?[ \t]*```$", "", stripped)
    return stripped.strip().strip("`").strip()


def _json_lists(text: str) -> List[str]:
    """Every flat JSON-ish list in text, normalized to canonical JSON strings."""
    found: List[str] = []
    for match in _JSON_LIST_RE.finditer(str(text or "")):
        candidate = match.group(0)
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list) and parsed:
                found.append(json.dumps(parsed, ensure_ascii=False))
            break
    return found


def is_degenerate_response(text: str) -> bool:
    """True when a stored response carries no gradeable content."""
    stripped = _strip_wrappers(text)
    return not stripped or bool(_DEGENERATE_ANSWER_RE.match(stripped))


def extract_final_answer_detailed(text: str) -> tuple[str, Dict[str, Any]]:
    """Extract the final answer plus diagnostics about how it was found."""
    raw = str(text or "")
    diagnostics = {"extraction_method": None, "parse_ok": False, "echoed_fewshot": False}

    if not raw.strip():
        diagnostics["extraction_method"] = "empty_completion"
        return "", diagnostics

    # 1. Last explicit marker. Few-shot exemplars contain earlier markers, so the
    #    last occurrence is the model's own answer, not an echoed exemplar.
    markers = list(_FINAL_MARKER_RE.finditer(raw))
    if markers:
        tail = raw[markers[-1].end():]
        tail_lists = _json_lists(tail)
        if tail_lists:
            diagnostics.update(extraction_method="marker_json_list", parse_ok=True)
            return tail_lists[0], diagnostics
        first_line = next((ln.strip() for ln in tail.splitlines() if ln.strip()), "")
        first_line = _strip_wrappers(first_line).rstrip(".")
        if first_line and not is_degenerate_response(first_line):
            diagnostics.update(extraction_method="marker_text", parse_ok=True)
            diagnostics["echoed_fewshot"] = first_line.lower() in FEWSHOT_FINAL_ANSWERS
            return first_line, diagnostics

    # 2. Fenced code block holding a list. This is the dominant failure mode:
    #    the model skips the self-ask scaffold and emits ```json [...] ```.
    for block in reversed(_CODE_FENCE_RE.findall(raw)):
        block_lists = _json_lists(block)
        if block_lists:
            diagnostics.update(extraction_method="fenced_json_list", parse_ok=True)
            return block_lists[-1], diagnostics
        block_text = _strip_wrappers(block)
        if block_text and not is_degenerate_response(block_text):
            diagnostics.update(extraction_method="fenced_text", parse_ok=True)
            return block_text, diagnostics

    # 3. Any bare list anywhere in the completion.
    bare_lists = _json_lists(raw)
    if bare_lists:
        diagnostics.update(extraction_method="bare_json_list", parse_ok=True)
        return bare_lists[-1], diagnostics

    # 4. Last line that actually carries content (fences/empties skipped).
    for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
        cleaned = _strip_wrappers(line).rstrip(".")
        if cleaned and not is_degenerate_response(cleaned):
            diagnostics.update(extraction_method="last_content_line", parse_ok=True)
            diagnostics["echoed_fewshot"] = cleaned.lower() in FEWSHOT_FINAL_ANSWERS
            return cleaned, diagnostics

    diagnostics["extraction_method"] = "no_answer_found"
    return "", diagnostics


def extract_final_answer(text: str) -> str:
    return extract_final_answer_detailed(text)[0]


def extract_selfask_trace(text: str) -> dict:
    follow_ups = []
    fu_pat = re.compile(
        r"Follow up:\s*(.+?)\s*\n\s*Intermediate answer:\s*(.+?)(?=\n\s*(?:Follow up:|So the final)|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in fu_pat.finditer(text or ""):
        follow_ups.append({"question": m.group(1).strip(), "answer": m.group(2).strip()})

    final_answer, diagnostics = extract_final_answer_detailed(text)
    return {
        "needs_followup": "yes" in text[:50].lower() if text else False,
        "num_decomposition_steps": len(follow_ups),
        "follow_ups": follow_ups,
        "final_answer": final_answer,
        **diagnostics,
    }


# ===========================================================================
# HINT CONSTRUCTION HELPERS
# ===========================================================================

def create_count_hinted_prompt(original_prompt: str, expected_count: int) -> str:
    for verb in ("List", "Name", "Provide", "list", "name", "provide"):
        pat = re.compile(rf'\b{verb}\s+ALL\b', re.IGNORECASE)
        if pat.search(original_prompt):
            return pat.sub(f'{verb} ALL {expected_count}', original_prompt, count=1)
    return f"(Note: There are exactly {expected_count} items to list.) " + original_prompt

def normalize_item_for_first_letter(item: str) -> str:
    s = re.sub(r'^[\s"\'`\(\[\{]+|[\s"\'`\)\]\}]+$', '', str(item).strip())
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE)
    m = re.search(r'[A-Za-z]', s)
    return m.group(0).upper() if m else "?"

def build_first_letter_hint(expected_items: list, sort_letters: bool = True) -> list:
    letters = [normalize_item_for_first_letter(x) for x in expected_items]
    return sorted(letters) if sort_letters else letters

def create_firstletter_hinted_prompt(original_prompt: str, expected_items: list, sort_letters: bool = True) -> str:
    letters = build_first_letter_hint(expected_items, sort_letters)
    return f"{original_prompt.rstrip()} The first letters of the answers are: {', '.join(letters)}."

def get_base_question(prompts_dict: dict, prefer_reason: bool = False) -> str:
    priority = ("reason", "direct", "base", "standard", "raw") if prefer_reason else ("direct", "base", "standard", "raw")
    for key in priority:
        if key in prompts_dict:
            return prompts_dict[key]
    return next(iter(prompts_dict.values()))

def build_final_prompt_for_variant(variant: str, base_question: str, expected_items: list) -> str:
    if variant in SELFASK_TEMPLATES:
        return build_selfask_prompt(base_question, variant)
    elif variant in ("reason_hinted", "direct_hinted"):
        return create_count_hinted_prompt(base_question, len(expected_items))
    elif variant == "reason_firstletter":
        return create_firstletter_hinted_prompt(base_question, expected_items, sort_letters=True)
    else:
        raise ValueError(f"Unknown extended variant: {variant}")


# ===========================================================================
# LLM CLIENT - OpenRouter with Extended Reasoning Support
# ===========================================================================

def call_openrouter(
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
    thinking: bool = False,
) -> Dict[str, Any]:
    """
    Call a model via OpenRouter. Returns the raw response text and metadata.

    When thinking=True, enables extended reasoning via OpenRouter's unified
    reasoning parameter for models that support it.
    """
    from openai import OpenAI

    # Resolve the API slug independently from the storage/folder label.
    openrouter_model = resolve_openrouter_model(model_name)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Build extra_body with reasoning config if thinking is enabled
    extra_body = {}
    reasoning_used = None

    if thinking:
        reasoning_param = build_reasoning_param(model_name)
        if reasoning_param:
            extra_body["reasoning"] = reasoning_param
            reasoning_used = True
            safe_print(f"    [REASONING] {openrouter_model} → enabled")
        else:
            safe_print(f"    [NO REASONING] {openrouter_model} - not supported")

    # Make the API call
    call_kwargs = {
        "model": openrouter_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if extra_body:
        call_kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**call_kwargs)

    msg = resp.choices[0].message
    content = getattr(msg, "content", "") or ""

    # Extract reasoning if present
    reasoning_content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    reasoning_details = getattr(msg, "reasoning_details", None)

    result = {
        "success": True,
        "response": content,
        "model": openrouter_model,
        "finish_reason": getattr(resp.choices[0], "finish_reason", None),
        "usage": {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        } if getattr(resp, "usage", None) else None,
    }

    # Add reasoning info if present
    if reasoning_used:
        result["reasoning_enabled"] = True
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    if reasoning_details:
        result["reasoning_details"] = reasoning_details

    return result


# ===========================================================================
# JUDGE PROMPTS & PARSING
# ===========================================================================

def generate_batch_binding_judge_prompt(items: list[dict]) -> str:
    formatted = "\n\n".join(
        f"[{idx}]\nQuestion: {i['prompt']}\nExpected items (ALL must be present): {i['expected']}\nModel Response: {i['response']}"
        for idx, i in enumerate(items)
    )

    return f"""You are evaluating whether a model's response contains ALL expected items.

## CRITICAL EVALUATION RULE - READ CAREFULLY:

A response is **CORRECT** if it contains ALL items from the "Expected items" list.
A response is **INCORRECT** ONLY if it is MISSING one or more expected items.

**EXTRA ITEMS IN THE RESPONSE DO NOT MATTER AT ALL.**
- If expected is ["A", "B"] and response contains ["A", "B", "C", "D"], mark CORRECT even if C and D are wrong per se.
- If expected is ["A", "B"] and response contains only ["A", "C"], mark INCORRECT (missing B).

## MATCHING RULES (be lenient):
- Name variations are equivalent: "UK" = "United Kingdom" = "Britain"
- "US" = "USA" = "United States" = "United States of America"  
- "The Masters" = "Masters Tournament"
- Nicknames or appellatives: "Ernesto Rafael Guevara de la Serna" = "Che Guevara"
- "Marshall Bruce Mathers III" = "Eminem"
- "Armando Christian Pérez" = "Pitbull"
- Date formats: "June 24, 2009" = "24 June 2009" = "2009-06-24"
- Minor typos or spelling variations are OK
- Partial matches within longer text count (e.g., "Resident Evil 2 Remake" matches "Resident Evil 2")

## WHEN TO MARK INCORRECT:
- One or more expected items are completely missing from the response
- The response says "I don't know" or refuses to answer
- Wrong entity entirely (e.g., expected "Haiti" but response says "Honduras")

## Items to evaluate:

{formatted}

## OUTPUT FORMAT:
Return ONLY a raw JSON array with no markdown formatting:
[{{"id": 0, "found_expected": ["list items from Expected that were found"], "missing": ["list items from Expected that are missing"], "correct": true/false}}]

Remember: Extra items beyond the expected list should NEVER cause a response to be marked incorrect."""


def generate_batch_binding_judge_prompt_with_reasoning(items: list[dict]) -> str:
    """
    Judge prompt for reasoning-enabled models. Considers BOTH the reasoning
    trace (chain-of-thought / thinking) AND the final response.
    
    If the model identified all expected items during reasoning but failed to
    format the final answer correctly (e.g. malformed JSON, truncated list,
    repeated items instead of all items), the answer is still CORRECT — the
    model *knew* the answer, it just failed at formatting.
    """
    parts = []
    for idx, i in enumerate(items):
        block = f"[{idx}]\nQuestion: {i['prompt']}\nExpected items (ALL must be present): {i['expected']}\nModel Final Response: {i['response']}"
        if i.get('reasoning_content'):
            block += f"\nModel Reasoning/Thinking Trace:\n{i['reasoning_content']}"
        parts.append(block)
    formatted = "\n\n".join(parts)
    
    return f"""You are evaluating whether a model KNOWS all expected items — by examining both its reasoning trace AND its final response.

## CRITICAL EVALUATION RULE — READ VERY CAREFULLY:

We are testing whether the model has the knowledge to produce all expected items.
The model receives BOTH a "Final Response" and (optionally) a "Reasoning/Thinking Trace".

A response is **CORRECT** if ALL expected items appear ANYWHERE — in the final response, in the reasoning trace, or split across both.

Typical failure modes that should still be marked **CORRECT**:
- Model lists all items in reasoning but the final JSON is malformed or truncated
- Model mentions all items in thinking but only writes a subset in the final answer due to formatting issues
- Model identifies all items during chain-of-thought but the final list has duplicates or reordering errors
- The final answer is cut off but the reasoning trace contains all items

A response is **INCORRECT** ONLY if an expected item is completely absent from BOTH the reasoning trace AND the final response.

**EXTRA ITEMS DO NOT MATTER AT ALL.**

## MATCHING RULES (be lenient):
- Name variations are equivalent: "UK" = "United Kingdom" = "Britain"
- "US" = "USA" = "United States" = "United States of America"  
- "The Masters" = "Masters Tournament"
- Nicknames: "Ernesto Rafael Guevara de la Serna" = "Che Guevara"
- "Marshall Bruce Mathers III" = "Eminem"
- "Armando Christian Perez" = "Pitbull"
- Date formats: "June 24, 2009" = "24 June 2009" = "2009-06-24"
- Minor typos or spelling variations are OK
- Partial matches within longer text count (e.g., "Resident Evil 2 Remake" matches "Resident Evil 2")

## WHEN TO MARK INCORRECT:
- One or more expected items are completely missing from BOTH the reasoning AND the response
- The model says "I don't know" and never identifies the item anywhere
- Wrong entity entirely (e.g., expected "Haiti" but model says "Honduras" everywhere)

## Items to evaluate:

{formatted}

## OUTPUT FORMAT:
Return ONLY a raw JSON array with no markdown formatting:
[{{"id": 0, "found_expected": ["items found in response OR reasoning"], "missing": ["items missing from BOTH response AND reasoning"], "correct": true/false, "found_in": "response|reasoning|both"}}]

Remember: If all items appear somewhere in the reasoning+response, mark CORRECT regardless of final formatting."""


# Set by --reasoning-judge-model. None = every answer is graded by the main judge
# on the final response only.
REASONING_JUDGE_MODEL: Optional[str] = None


def generate_batch_generic_judge_prompt(items: list[dict]) -> str:
    formatted = "\n\n".join(
        f"[{idx}]\nQ: {i['prompt']}\nExpected: {i['expected']}\nResponse: {i['response']}"
        for idx, i in enumerate(items)
    )
    return f"""Evaluate if each response correctly matches expected answer.

LENIENCY RULES (mark as CORRECT if):
- Response contains ALL expected items (extra items are OK)
- Common name variations: "UK"="United Kingdom"="Britain", etc.
- Date format variations: "June 24, 2009"="24 June 2009"="2009-06-24"
- Minor typos or spelling variations

STRICTNESS RULES (mark as INCORRECT if):
- Missing any expected item
- Wrong entity entirely
- Response says "I don't know" or refuses

{formatted}

Return ONLY raw JSON array: [{{"id": 0, "identified_items": ["items"], "correct": true/false, "explanation": "brief"}}]"""


def parse_batch_response(response: str) -> list[dict]:
    """Parse the judge's batch verdict, salvaging partial/wrapped JSON.

    Returns [] only when nothing object-shaped can be recovered, so callers can
    distinguish a judge formatting failure from a genuinely empty verdict.
    """
    text = str(response or "").strip()
    if not text:
        return []

    fenced = _CODE_FENCE_RE.findall(text)
    candidates = [text] + [block.strip() for block in fenced]

    # Outermost bracketed span, in case the judge added prose around the array.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list) and all(isinstance(i, dict) for i in value):
                    return value
            if "id" in parsed:
                return [parsed]

    # Last resort: pull individual objects out of a truncated array.
    salvaged = []
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "id" in item:
            salvaged.append(item)
    return salvaged


def evaluate_yesno(response: str, expected: str = "Yes") -> dict:
    response_lower = response.lower().strip()
    yes_patterns = [r'\byes\b', r'^yes', r'yes,', r'yes\.']
    no_patterns = [r'\bno\b', r'^no', r'no,', r'no\.']
    has_yes = any(re.search(p, response_lower) for p in yes_patterns)
    has_no = any(re.search(p, response_lower) for p in no_patterns)

    if has_yes and not has_no:
        detected = "yes"
    elif has_no and not has_yes:
        detected = "no"
    else:
        detected = "unclear"

    return {"detected_answer": detected, "expected": expected.lower(), "correct": (detected == expected.lower())}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def eval_entity_qtype(fd, letter, qtype, judge_model, num_retries):
    dr = fd / letter
    evd = fd / (f"eval_{letter}" if qtype == 'yesno' else f"eval_{qtype}_{letter}")
    evd.mkdir(parents=True, exist_ok=True)
    files = sorted(dr.glob(f"{qtype}_*.json"))[:num_retries]
    correct, total, pending = 0, 0, []

    for jf in files:
        evf = evd / f"eval_{jf.name}"
        if evf.exists():
            total += 1
            correct += json.load(open(evf)).get('evaluation', {}).get('correct', False)
        else:
            data = json.load(open(jf))
            pending.append({'idx': len(pending), 'file': jf, 'eval_file': evf, 'prompt': data['prompt'],
                           'expected': str(data.get('expected', 'Yes')), 'response': data['response']})

    if not pending:
        return correct, total

    if qtype == 'yesno':
        for p in pending:
            ev = evaluate_yesno(p['response'], p['expected'])
            json.dump({'prompt': p['prompt'], 'expected': p['expected'], 'response': p['response'], 
                      'evaluation': ev}, open(p['eval_file'], 'w'), indent=2)
            total += 1
            correct += ev.get('correct', False)
    else:
        prompt = generate_batch_generic_judge_prompt(pending)
        evals = parse_batch_response(call_openrouter(prompt, judge_model, 0.0, 4096).get('response', ''))
        eval_map = {int(e.get('id', -1)): e for e in evals}
        for p in pending:
            ev = eval_map.get(p['idx'], {'correct': False, 'error': 'not_found'})
            json.dump({'prompt': p['prompt'], 'expected': p['expected'], 'response': p['response'],
                      'evaluation': ev}, open(p['eval_file'], 'w'), indent=2)
            total += 1
            correct += ev.get('correct', False)
    return correct, total


UNPARSED_RESPONSE_ERROR = "unparsed_response"
JUDGE_PARSE_ERROR = "judge_parse_error"
REJUDGEABLE_ERRORS = ("not_found", UNPARSED_RESPONSE_ERROR, JUDGE_PARSE_ERROR)


def _reasoning_text(data: dict) -> str:
    """Thinking trace of a stored answer as plain text ('' if the API returned none)."""
    rc = data.get('reasoning_content') or data.get('reasoning_details') or ''
    if isinstance(rc, list):
        rc = "\n".join(str(d.get('content', d) if isinstance(d, dict) else d) for d in rc)
    return str(rc)


def eval_entity_binding(fd, judge_model, num_retries, var='raw', judge_retries=2):
    """Evaluate one list-based binding variant.

    Judge batch IDs are local indices (0, 1, 2, ...), so results must be mapped
    using the pending-item index rather than the source filename stem. Responses
    that carry no gradeable content are marked UNPARSED_RESPONSE_ERROR without
    being sent to the judge, and a judge reply that cannot be parsed is recorded
    as JUDGE_PARSE_ERROR rather than silently scoring every item as missing.
    """
    dr, evd = fd / f'binding_{var}', fd / f'eval_binding_{var}'
    evd.mkdir(parents=True, exist_ok=True)
    files = sorted(dr.glob("*.json"))[:num_retries]
    correct, total, pending = 0, 0, []

    for jf in files:
        evf = evd / f"eval_{jf.name}"
        if evf.exists():
            total += 1
            correct += bool(
                json.load(open(evf, encoding="utf-8"))
                .get('evaluation', {})
                .get('correct', False)
            )
            continue

        data = json.load(open(jf, encoding="utf-8"))
        response = data.get('response', '')

        if is_degenerate_response(response):
            with open(evf, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'expected': str(data.get('expected', [])),
                        'response': response,
                        'evaluation': {
                            'correct': False,
                            'error': UNPARSED_RESPONSE_ERROR,
                            'extraction_method': data.get('selfask_trace', {}).get('extraction_method'),
                            'source_file': jf.name,
                        },
                    },
                    f, indent=2, ensure_ascii=False,
                )
            total += 1
            continue

        pending.append({
            'idx': len(pending),
            'source_file': jf.name,
            'eval_file': evf,
            'prompt': data['prompt'],
            'expected': str(data.get('expected', [])),
            'response': response,
            'reasoning_content': _reasoning_text(data),
        })

    if not pending:
        return correct, total

    use_trace = bool(REASONING_JUDGE_MODEL) and any(p['reasoning_content'] for p in pending)
    if use_trace:
        prompt = generate_batch_binding_judge_prompt_with_reasoning(pending)
        effective_judge = REASONING_JUDGE_MODEL
    else:
        prompt = generate_batch_binding_judge_prompt(pending)
        effective_judge = judge_model
    judge_response, evals = '', []
    for attempt in range(judge_retries + 1):
        judge_response = call_openrouter(prompt, effective_judge, 0.0, 4096).get('response', '')
        evals = parse_batch_response(judge_response)
        if evals:
            break
        safe_print(
            f"  ⚠ unparseable judge reply (attempt {attempt + 1}/{judge_retries + 1}) "
            f"for {fd.name}/binding_{var}"
        )

    eval_map = {}
    for ev in evals:
        try:
            eval_map[int(ev.get('id'))] = ev
        except (TypeError, ValueError):
            continue

    fallback_error = 'not_found' if evals else JUDGE_PARSE_ERROR

    for p in pending:
        ev = eval_map.get(p['idx'])
        if ev is None:
            ev = {
                'correct': False,
                'error': fallback_error,
                'judge_batch_id': p['idx'],
                'source_file': p['source_file'],
                'judge_response': judge_response,
            }

        record = {'expected': p['expected'], 'response': p['response'], 'evaluation': ev}
        if use_trace and p['reasoning_content']:
            record.update(reasoning_content_used=True, judge_model=effective_judge)
            if ev.get('found_in'):
                record['found_in'] = ev['found_in']
        with open(p['eval_file'], 'w', encoding='utf-8') as f:
            json.dump(
                record,
                f, indent=2, ensure_ascii=False,
            )

        total += 1
        correct += bool(ev.get('correct', False))

    return correct, total


def extract_mc_answer(response: str) -> Optional[str]:
    """Extract a standalone A-Z multiple-choice answer from common formats."""
    text = str(response or "").strip().upper()
    text = re.sub(r"[*_`#]", "", text).strip()

    patterns = [
        r"^\s*[\(\[]?([A-Z])[\)\].,:]?\s*$",
        r"^\s*[\(\[]?([A-Z])[\)\].,:]?\b",
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def eval_entity_binding_mc(fd, judge_model, num_retries, var='raw'):
    dr, evd = fd / f'binding_mc_{var}', fd / f'eval_binding_mc_{var}'
    evd.mkdir(parents=True, exist_ok=True)
    files = sorted(dr.glob("*.json"))[:num_retries]
    correct, total = 0, 0

    for jf in files:
        evf = evd / f"eval_{jf.name}"
        if evf.exists():
            with open(evf, encoding="utf-8") as f:
                existing = json.load(f)
            total += 1
            correct += bool(existing.get('correct', False))
            continue

        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        expected = str(data['expected']).strip().upper()
        response = data.get('response', '')
        detected = extract_mc_answer(response)
        is_correct = detected == expected

        with open(evf, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'expected': expected,
                    'response': response,
                    'detected_answer': detected,
                    'correct': is_correct,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        total += 1
        correct += is_correct

    return correct, total


# ─────────────────────────────────────────────────────────────────────────────
# QUERY RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_query(prompt, model_name, temperature, max_tokens, sys_prompt, thinking, filepath, metadata):
    """Helper to run a single LLM query and save result to file."""
    res = call_openrouter(prompt, model_name, temperature, max_tokens, sys_prompt, thinking)
    json.dump({**metadata, **res}, open(filepath, 'w'), indent=2)
    return res


def _run_single_extended_query(variant, original_prompt, final_prompt, model_name, 
                                temperature, max_tokens, sys_prompt, thinking, filepath, metadata):
    """
    Run a single extended-variant query via call_openrouter, post-process for self-ask variants, save.
    """
    res = call_openrouter(final_prompt, model_name, temperature, max_tokens, sys_prompt, thinking)
    raw_response = res.get('response', '')

    if variant in SELFASK_TEMPLATES:
        # Parse self-ask trace and extract final answer
        trace = extract_selfask_trace(raw_response)
        output = {
            **metadata,
            **res,
            "response": trace["final_answer"],
            "selfask_full_completion": raw_response,
            "selfask_trace": trace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # Hinted variants: response is used as-is
        output = {
            **metadata,
            **res,
            "original_prompt": original_prompt,
            "hinted_prompt": final_prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    json.dump(output, open(filepath, 'w'), indent=2, ensure_ascii=False)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS SINGLE ENTITY
# ─────────────────────────────────────────────────────────────────────────────

def process_single_entity(entity, case, mp, model_name, temperature, max_tokens, 
                          sys_prompt, num_retries, threshold, judge_model, executor,
                          list_variants=None, mc_variants=None, extended_variants=None):
    """
    Process one entity: facts → evaluate → if pass → bindings → evaluate.

    Args:
        list_variants: list-based binding variants, unprefixed (e.g. ["raw", "reason"])
        mc_variants:   MCQ binding variants, unprefixed (e.g. ["raw", "reason"])
        extended_variants: self-ask / hinted variants (e.g. ["selfask", "selfask_modified_2"])

    Extended reasoning is enabled for the 'reason' variant of both list and MCQ
    arms, and for the hinted extended variants.
    """
    # Default to all variants if not specified
    if list_variants is None:
        list_variants = ["raw", "no_reason", "reason"]
    if mc_variants is None:
        mc_variants = ["raw", "no_reason", "reason"]
    if extended_variants is None:
        extended_variants = list(EXTENDED_VARIANTS)
    fd = mp / f"fail_{entity.replace(' ', '_').lower()}"
    letters = sorted(set(k.split('_')[1] for k in case['facts'].keys()))
    ok = True

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Run all fact queries for this entity in parallel
    # ─────────────────────────────────────────────────────────────────────────
    fact_futures = []

    for letter in letters:
        qtypes = list(set(k.split('_')[2] for k in case['facts'].keys() if k.startswith(f'fact_{letter}_')))
        for qtype in qtypes:
            qd = case['facts'].get(f'fact_{letter}_{qtype}')
            if not qd:
                continue
            dr = fd / letter
            dr.mkdir(parents=True, exist_ok=True)

            for r in range(num_retries):
                fp = dr / f"{qtype}_{r}.json"
                if fp.exists():
                    safe_print(f"  SKIP {entity}/{letter}/{qtype}_{r}")
                    continue

                future = executor.submit(
                    _run_single_query, 
                    qd['prompt'], model_name, temperature, max_tokens, sys_prompt, False,
                    fp, {"entity": entity, "prompt": qd['prompt'], "expected": qd.get('expected'), "retry": r}
                )
                fact_futures.append((future, entity, letter, qtype, r))

    for future, ent, letter, qtype, r in fact_futures:
        try:
            future.result()
            safe_print(f"  ✓ {ent}/{letter}/{qtype}_{r}")
        except Exception as e:
            safe_print(f"  ✗ ERROR {ent}/{letter}/{qtype}_{r}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Evaluate facts and check threshold
    # ─────────────────────────────────────────────────────────────────────────
    for letter in letters:
        if not ok:
            break
        qtypes = list(set(k.split('_')[2] for k in case['facts'].keys() if k.startswith(f'fact_{letter}_')))
        for qtype in qtypes:
            if not ok:
                break
            if not case['facts'].get(f'fact_{letter}_{qtype}'):
                continue
            c, t = eval_entity_qtype(fd, letter, qtype, judge_model, num_retries)
            pct = c / t * 100 if t else 0
            safe_print(f"  {entity} {letter}/{qtype}: {c}/{t} ({pct:.0f}%)")
            if t > 0 and c / t < threshold:
                safe_print(f"  ⚠ {entity} FAILED threshold → skip binding")
                ok = False

    if not ok:
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Run all binding queries for this entity in parallel
    # ─────────────────────────────────────────────────────────────────────────
    binding_futures = []
    qd = case['binding']
    qd_mc = case['binding_multichoice']

    # --- List-based binding variants ---
    for var in list_variants:
        use_thinking = (var == 'reason')  # Enable extended reasoning for 'reason'

        dr = fd / f'binding_{var}'
        dr.mkdir(parents=True, exist_ok=True)
        for r in range(num_retries):
            fp = dr / f"{r}.json"
            if fp.exists():
                safe_print(f"  SKIP {entity}/binding_{var}/{r}")
                continue

            future = executor.submit(
                _run_single_query,
                qd['prompts'][var], model_name, temperature, max_tokens, sys_prompt, use_thinking,
                fp, {"entity": entity, "prompt": qd['prompts'][var], 
                     "expected": qd.get('expected') or qd.get('expected_teams'), "retry": r,
                     "extended_reasoning_enabled": use_thinking}
            )
            binding_futures.append((future, f"binding_{var}", r))

    # --- MCQ binding variants ---
    for var in mc_variants:
        use_thinking = (var == 'reason')  # Enable extended reasoning for 'reason'

        dr_mc = fd / f'binding_mc_{var}'
        dr_mc.mkdir(parents=True, exist_ok=True)
        for r in range(num_retries):
            fp = dr_mc / f"{r}.json"
            if fp.exists():
                safe_print(f"  SKIP {entity}/binding_mc_{var}/{r}")
                continue

            future = executor.submit(
                _run_single_query,
                qd_mc['prompts'][var], model_name, temperature, max_tokens, sys_prompt, use_thinking,
                fp, {"entity": entity, "prompt": qd_mc['prompts'][var], 
                     "expected": qd_mc['expected_answer'], "retry": r,
                     "extended_reasoning_enabled": use_thinking}
            )
            binding_futures.append((future, f"binding_mc_{var}", r))

    # --- Extended variants: selfask, selfask_modified_2, reason_hinted, reason_firstletter (list only) ---
    expected = qd.get('expected') or qd.get('expected_teams') or []

    if expected and extended_variants:
        for var in extended_variants:
            prefer_reason = var in ("reason_hinted", "reason_firstletter")
            # Enable extended reasoning for hinted variants (they benefit from chain-of-thought).
            # Self-ask variants deliberately run without it: the decomposition is prompted.
            use_thinking = var in ("reason_hinted", "reason_firstletter")
            base_q = get_base_question(qd['prompts'], prefer_reason=prefer_reason)
            final_prompt = build_final_prompt_for_variant(var, base_q, expected)

            dr = fd / f'binding_{var}'
            dr.mkdir(parents=True, exist_ok=True)

            for r in range(num_retries):
                fp = dr / f"{r}.json"
                if fp.exists():
                    safe_print(f"  SKIP {entity}/binding_{var}/{r}")
                    continue

                meta = {
                    "entity": entity, "prompt": final_prompt,
                    "expected": expected, "retry": r,
                    "variant": var, "query_type": "binding_list",
                    "prompting_method": var,
                    "extended_reasoning_enabled": use_thinking,
                }
                if var in ("reason_hinted", "direct_hinted"):
                    meta["expected_count"] = len(expected)
                elif var == "reason_firstletter":
                    meta["hint_letters"] = build_first_letter_hint(expected, sort_letters=True)

                future = executor.submit(
                    _run_single_extended_query,
                    var, base_q, final_prompt, model_name, temperature, max_tokens,
                    sys_prompt, use_thinking, fp, meta
                )
                binding_futures.append((future, f"binding_{var}", r))

    # Wait for all binding queries to complete
    for future, btype, r in binding_futures:
        try:
            future.result()
            safe_print(f"  ✓ {entity}/{btype}/{r}")
        except Exception as e:
            safe_print(f"  ✗ ERROR {entity}/{btype}/{r}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Evaluate bindings
    # ─────────────────────────────────────────────────────────────────────────

    # List-based variants
    for var in list_variants:
        c, t = eval_entity_binding(fd, judge_model, num_retries, var)
        if t:
            safe_print(f"  {entity} binding_{var}: {c}/{t} ({c/t*100:.0f}%)")

    # MC variants
    for var in mc_variants:
        c, t = eval_entity_binding_mc(fd, judge_model, num_retries, var)
        if t:
            safe_print(f"  {entity} binding_mc_{var}: {c}/{t} ({c/t*100:.0f}%)")

    # Extended variants (list only)
    if expected and extended_variants:
        for var in extended_variants:
            c, t = eval_entity_binding(fd, judge_model, num_retries, var)
            if t:
                safe_print(f"  {entity} binding_{var}: {c}/{t} ({c/t*100:.0f}%)")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

ALL_LIST_VARIANTS = ["raw", "no_reason", "reason"]
ALL_MC_VARIANTS = ["raw", "no_reason", "reason"]          # referred to as mc_<name> on the CLI
ALL_EXTENDED_VARIANTS = list(EXTENDED_VARIANTS)

VARIANT_SHORTCUTS = {
    "instant":      ["raw", "mc_raw"],
    "reasoning":    ["reason", "mc_reason"],
    "all_list":     list(ALL_LIST_VARIANTS),
    "all_mc":       [f"mc_{v}" for v in ALL_MC_VARIANTS],
    "all_extended": list(ALL_EXTENDED_VARIANTS),
    # The rebuttal rerun arm: updated self-ask, self-ask binding, and MCQ-reasoning.
    "rerun_selfask_mcreason": ["selfask", "selfask_modified_2", "mc_reason"],
}


def resolve_binding_variants(binding_variants):
    """Split a CLI variant list into (list, mc, extended) variant groups.

    MCQ variants are written mc_<name> on the CLI and returned unprefixed, so
    process_single_entity can index prompts['<name>'] and write binding_mc_<name>.
    Passing an empty/None list runs everything.
    """
    known_list = set(ALL_LIST_VARIANTS)
    known_mc = {f"mc_{v}" for v in ALL_MC_VARIANTS}
    known_extended = set(ALL_EXTENDED_VARIANTS)

    if not binding_variants:
        return list(ALL_LIST_VARIANTS), list(ALL_MC_VARIANTS), list(ALL_EXTENDED_VARIANTS)

    requested = set()
    for v in binding_variants:
        if v == "all":
            requested |= known_list | known_mc | known_extended
        elif v in VARIANT_SHORTCUTS:
            requested.update(VARIANT_SHORTCUTS[v])
        else:
            requested.add(v)

    unknown = requested - known_list - known_mc - known_extended
    if unknown:
        raise ValueError(
            f"Unknown binding variants: {sorted(unknown)}. "
            f"Valid: {sorted(known_list | known_mc | known_extended)} "
            f"or shortcuts {sorted(VARIANT_SHORTCUTS) + ['all']}"
        )

    list_variants = [v for v in ALL_LIST_VARIANTS if v in requested]
    mc_variants = [v for v in ALL_MC_VARIANTS if f"mc_{v}" in requested]
    extended_variants = [v for v in ALL_EXTENDED_VARIANTS if v in requested]
    return list_variants, mc_variants, extended_variants


# ─────────────────────────────────────────────────────────────────────────────
# REJUDGE EXISTING NOT_FOUND SELF-ASK EVALUATIONS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REJUDGE_VARIANTS = ["selfask", "selfask_modified_2"]

# Every strategy the repair passes (rejudge / reval) can touch, expressed as
# *directory tokens*: a token maps to binding_<token> / eval_binding_<token>, so
# MCQ arms keep their mc_ prefix here (unlike --variants, which strips it).
ALL_REJUDGE_VARIANTS = (
    list(ALL_LIST_VARIANTS)
    + [f"mc_{v}" for v in ALL_MC_VARIANTS]
    + list(ALL_EXTENDED_VARIANTS)
)

REJUDGE_VARIANT_SHORTCUTS = {
    "all":          list(ALL_REJUDGE_VARIANTS),
    "all_list":     list(ALL_LIST_VARIANTS),
    "all_mc":       [f"mc_{v}" for v in ALL_MC_VARIANTS],
    "all_extended": list(ALL_EXTENDED_VARIANTS),
    "all_selfask":  [v for v in ALL_EXTENDED_VARIANTS if v in SELFASK_TEMPLATES],
}


def resolve_rejudge_variants(variant_names):
    """Expand rejudge/reval variant tokens, honoring 'all' and the group shortcuts.

    Returns directory tokens in canonical order (list arm, MCQ arm, extended arm).
    An empty/None argument keeps the historical self-ask-only default so existing
    commands do not silently widen their scope.
    """
    if not variant_names:
        return list(DEFAULT_REJUDGE_VARIANTS)

    requested = []
    for name in variant_names:
        for token in REJUDGE_VARIANT_SHORTCUTS.get(name, [name]):
            if token not in requested:
                requested.append(token)

    unknown = [token for token in requested if token not in ALL_REJUDGE_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown rejudge variants: {sorted(set(unknown))}. "
            f"Valid tokens: {ALL_REJUDGE_VARIANTS} "
            f"or shortcuts {sorted(REJUDGE_VARIANT_SHORTCUTS)}"
        )

    return [token for token in ALL_REJUDGE_VARIANTS if token in requested]


def is_mc_variant(variant: str) -> bool:
    """True for MCQ directory tokens (mc_raw, mc_no_reason, mc_reason)."""
    return variant.startswith("mc_")


def normalize_eval_record(eval_data: dict) -> Dict[str, Any]:
    """Read either evaluation schema and return {'correct': bool, 'error': str|None}.

    List and extended variants nest their verdict under 'evaluation'; MCQ files
    written by eval_entity_binding_mc are flat and carry no error field, so a
    missing detected_answer is surfaced as UNPARSED_RESPONSE_ERROR to make MCQ
    rows selectable by the same parse-error / truncation passes.
    """
    evaluation = eval_data.get("evaluation")
    if isinstance(evaluation, dict):
        return {
            "correct": bool(evaluation.get("correct", False)),
            "error": evaluation.get("error"),
        }

    error = None
    if "detected_answer" in eval_data and eval_data.get("detected_answer") is None:
        error = UNPARSED_RESPONSE_ERROR
    return {"correct": bool(eval_data.get("correct", False)), "error": error}


def reeval_variant(fd, judge_model, num_retries, variant):
    """Re-run the evaluator that matches a directory token.

    MCQ tokens go to the local letter-extraction path (no judge call); everything
    else goes to the LLM-judge path.
    """
    if is_mc_variant(variant):
        return eval_entity_binding_mc(fd, judge_model, num_retries, variant[len("mc_"):])
    return eval_entity_binding(fd, judge_model, num_retries, variant)


# Directory-name prefix reserved for backups/snapshots, so they are never
# themselves snapshotted, archived or walked as prediction data.
INTERNAL_DIR_PREFIX = "_"
SNAPSHOT_MANIFEST_NAME = "_manifest.json"
CREATED_FILES_MANIFEST_NAME = "_created_files.json"












def rejudge_not_found_bindings(model_root, judge_model, num_retries=5,
                               variants=None, backup_root=None):
    """Re-run the judge only for binding evaluations with error=not_found.

    Works for any strategy: list, MCQ (mc_*) and extended variants. Existing
    broken evaluation files are moved into a timestamped backup before rejudging,
    so the operation can be rolled back exactly.
    """
    model_root = Path(model_root)
    variants = resolve_rejudge_variants(variants)

    if backup_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = model_root / f"_rejudge_backup_{stamp}"
    else:
        backup_root = Path(backup_root)

    moved_by_entity_variant = {}
    moved_total = 0

    for fd in sorted(model_root.glob("fail_*")):
        if not fd.is_dir():
            continue

        for var in variants:
            evd = fd / f"eval_binding_{var}"
            if not evd.exists():
                continue

            moved = 0
            for evf in sorted(evd.glob("eval_*.json")):
                try:
                    data = json.loads(evf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    safe_print(f"  SKIP invalid evaluation {evf}: {exc}")
                    continue

                if normalize_eval_record(data)["error"] != "not_found":
                    continue

                destination = backup_root / evf.relative_to(model_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(evf), str(destination))
                moved += 1
                moved_total += 1

            if moved:
                moved_by_entity_variant[(fd, var)] = moved

    if moved_total == 0:
        print(
            "No binding evaluations with error='not_found' were found for variants: "
            f"{variants}"
        )
        return

    print(f"Moved {moved_total} broken evaluations to: {backup_root}")

    for (fd, var), moved in moved_by_entity_variant.items():
        correct, total = reeval_variant(fd, judge_model, num_retries, var)
        safe_print(
            f"  REJUDGED {fd.name}/binding_{var}: "
            f"moved={moved}, current_score={correct}/{total}"
        )

    print(f"Rejudging complete. Exact backup: {backup_root}")


def rejudge_parse_errors(model_root, judge_model, num_retries=5, variants=None,
                         error_kinds=REJUDGEABLE_ERRORS, reextract=True, backup_root=None):
    """Re-extract self-ask answers with the current parser, then rejudge.

    Two independent repairs, both rollback-safe:
      1. reextract: re-runs extract_final_answer_detailed over the stored
         selfask_full_completion and rewrites 'response' when it changes. Only
         self-ask variants store that completion, so for list/MCQ/hinted
         strategies this step is a no-op and only the rejudge below applies.
      2. rejudge: drops evaluations whose error is in error_kinds (plus any
         whose response was just rewritten) so the matching evaluator redoes
         them. MCQ rows qualify when detected_answer is null, i.e. the letter
         extractor found nothing.
    """
    model_root = Path(model_root)
    variants = resolve_rejudge_variants(variants)
    error_kinds = set(error_kinds)

    if backup_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = model_root / f"_reparse_backup_{stamp}"
    else:
        backup_root = Path(backup_root)

    def _backup(path: Path, move: bool):
        destination = backup_root / path.relative_to(model_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(path), str(destination))
        else:
            shutil.copy2(str(path), str(destination))

    reextracted_total, dropped_total = 0, 0
    dirty = {}

    for fd in sorted(model_root.glob("fail_*")):
        if not fd.is_dir():
            continue

        for var in variants:
            response_dir, eval_dir = fd / f"binding_{var}", fd / f"eval_binding_{var}"
            reextracted, dropped, stale_evals = 0, 0, set()
            preexisting_evals = (
                {p.name for p in eval_dir.glob("eval_*.json")} if eval_dir.is_dir() else set()
            )

            if reextract and response_dir.is_dir():
                for rf in sorted(response_dir.glob("*.json")):
                    try:
                        data = json.loads(rf.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        safe_print(f"  SKIP invalid response {rf}: {exc}")
                        continue

                    completion = data.get("selfask_full_completion")
                    if not completion:
                        continue

                    new_answer, diagnostics = extract_final_answer_detailed(completion)
                    if new_answer == data.get("response", ""):
                        continue

                    _backup(rf, move=False)
                    data["response_before_reparse"] = data.get("response", "")
                    data["response"] = new_answer
                    data["selfask_trace"] = {**data.get("selfask_trace", {}), **diagnostics,
                                             "final_answer": new_answer}
                    data["reparsed_at"] = datetime.now(timezone.utc).isoformat()
                    rf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    reextracted += 1
                    stale_evals.add(eval_dir / f"eval_{rf.name}")

            if eval_dir.is_dir():
                for evf in sorted(eval_dir.glob("eval_*.json")):
                    if evf not in stale_evals:
                        try:
                            data = json.loads(evf.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError) as exc:
                            safe_print(f"  SKIP invalid evaluation {evf}: {exc}")
                            continue
                        if normalize_eval_record(data)["error"] not in error_kinds:
                            continue
                    if evf.exists():
                        _backup(evf, move=True)
                        dropped += 1

            if reextracted or dropped:
                dirty[(fd, var)] = (reextracted, dropped, preexisting_evals)
                reextracted_total += reextracted
                dropped_total += dropped

    if not dirty:
        print("Nothing to reparse or rejudge for the requested variants.")
        return

    print(f"Re-extracted {reextracted_total} responses, dropped {dropped_total} evaluations.")
    print(f"Exact backup: {backup_root}")

    created_files = []
    for (fd, var), (reextracted, dropped, preexisting_evals) in dirty.items():
        correct, total = reeval_variant(fd, judge_model, num_retries, var)
        eval_dir = fd / f"eval_binding_{var}"
        for evf in sorted(eval_dir.glob("eval_*.json")):
            if evf.name not in preexisting_evals:
                created_files.append(evf.relative_to(model_root).as_posix())
        safe_print(
            f"  REPARSED {fd.name}/binding_{var}: reextracted={reextracted}, "
            f"dropped={dropped}, current_score={correct}/{total}"
        )

    # Files with no pre-run counterpart cannot be reverted by copying the backup
    # back, so restore_rejudge_backup deletes anything listed here.
    (backup_root / CREATED_FILES_MANIFEST_NAME).write_text(
        json.dumps({"created": created_files}, indent=2), encoding="utf-8"
    )
    if created_files:
        print(f"Recorded {len(created_files)} newly created evaluation files for rollback.")

    print(f"Reparse + rejudge complete. Roll back with --restore-rejudge-backup {backup_root}")


def _rejudge_selected(model_root, judge_model, num_retries, variants, should_rejudge,
                      label, backup_root=None):
    """Drop and re-run the judge for evaluations matching a selector.

    should_rejudge(response_data, eval_data) decides per item, where response_data
    is the stored prediction (or {} if unreadable) and eval_data is the paired
    evaluation; selectors should read verdicts through normalize_eval_record so
    they work on both the nested and the flat MCQ schema. Dropped files are moved
    to a timestamped backup and files created by the rejudge are recorded, so
    --restore-rejudge-backup reverts exactly.
    """
    model_root = Path(model_root)
    variants = resolve_rejudge_variants(variants)

    if backup_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = model_root / f"_{label}_backup_{stamp}"
    else:
        backup_root = Path(backup_root)

    dirty, dropped_total = {}, 0

    for fd in sorted(model_root.glob("fail_*")):
        if not fd.is_dir():
            continue

        for var in variants:
            response_dir, eval_dir = fd / f"binding_{var}", fd / f"eval_binding_{var}"
            if not eval_dir.is_dir():
                continue

            preexisting_evals = {p.name for p in eval_dir.glob("eval_*.json")}
            dropped = 0

            for evf in sorted(eval_dir.glob("eval_*.json")):
                try:
                    eval_data = json.loads(evf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    safe_print(f"  SKIP invalid evaluation {evf}: {exc}")
                    continue

                response_file = response_dir / evf.name[len("eval_"):]
                response_data = {}
                if response_file.exists():
                    try:
                        response_data = json.loads(response_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        safe_print(f"  SKIP invalid response {response_file}: {exc}")
                        continue

                if not should_rejudge(response_data, eval_data):
                    continue

                destination = backup_root / evf.relative_to(model_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(evf), str(destination))
                dropped += 1
                dropped_total += 1

                safe_print(
                    f"  QUEUED {fd.name}/binding_{var}/{response_file.name}: "
                    f"total_tokens={(response_data.get('usage') or {}).get('total_tokens')}, "
                    f"finish_reason={response_data.get('finish_reason')}, "
                    f"prior_error={normalize_eval_record(eval_data)['error']}"
                )

            if dropped:
                dirty[(fd, var)] = (dropped, preexisting_evals)

    if dropped_total == 0:
        print(f"No evaluations matched the {label} criteria.")
        return

    print(f"Dropped {dropped_total} evaluations to: {backup_root}")

    created_files = []
    for (fd, var), (dropped, preexisting_evals) in dirty.items():
        correct, total = reeval_variant(fd, judge_model, num_retries, var)
        eval_dir = fd / f"eval_binding_{var}"
        for evf in sorted(eval_dir.glob("eval_*.json")):
            if evf.name not in preexisting_evals:
                created_files.append(evf.relative_to(model_root).as_posix())
        safe_print(
            f"  REJUDGED {fd.name}/binding_{var}: dropped={dropped}, "
            f"current_score={correct}/{total}"
        )

    backup_root.mkdir(parents=True, exist_ok=True)
    (backup_root / CREATED_FILES_MANIFEST_NAME).write_text(
        json.dumps({"created": created_files}, indent=2), encoding="utf-8"
    )

    print(f"{label} rejudging complete. Roll back with --restore-rejudge-backup {backup_root}")


def rejudge_truncated_answers(model_root, judge_model, num_retries=5, variants=None,
                              token_threshold=2048, backup_root=None):
    """Rejudge evaluations whose response was long enough to have been cut off.

    Selects items where the stored usage exceeds token_threshold total tokens AND
    the paired evaluation carries any error, on the assumption that a long
    completion plus a failed evaluation means the answer was truncated mid-list
    rather than genuinely wrong. Only the judge is re-run; responses are not
    rewritten. For MCQ rows the error signal is a null detected_answer, so a long
    completion whose answer letter never appeared is selected while a long
    completion that picked the wrong letter is left alone.
    """
    def should_rejudge(response_data, eval_data):
        total_tokens = (response_data.get("usage") or {}).get("total_tokens")
        if not isinstance(total_tokens, (int, float)) or total_tokens <= token_threshold:
            return False
        return bool(normalize_eval_record(eval_data)["error"])

    _rejudge_selected(
        model_root, judge_model, num_retries, variants, should_rejudge,
        label="truncated", backup_root=backup_root,
    )


def _is_truncated_usage(data: dict, token_threshold: int, include_length_finish: bool) -> Optional[str]:
    """Return the truncation signal for a stored response, or None.

    Two independent signals: the user-specified total-token threshold, and the
    provider reporting that generation stopped because it hit the token ceiling.
    """
    if include_length_finish and str(data.get("finish_reason") or "").lower() == "length":
        return "finish_reason_length"

    usage = data.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, (int, float)) and total_tokens > token_threshold:
        return f"total_tokens_{int(total_tokens)}_over_{token_threshold}"

    return None


def _requery_item(response_file: Path, model_name: str, temperature: float,
                  reval_max_tokens: int, signal: str) -> Dict[str, Any]:
    """Re-query one stored item with a larger output budget and rewrite its file.

    The original prompt, variant and extended-reasoning setting are reused from
    the stored metadata, so the only changed generation parameter is max_tokens.
    """
    data = json.loads(response_file.read_text(encoding="utf-8"))
    variant = data.get("variant") or ""
    thinking = bool(data.get("extended_reasoning_enabled", False))

    result = call_openrouter(
        data["prompt"], model_name, temperature, reval_max_tokens,
        SYSTEM_PROMPT, thinking,
    )
    raw_response = result.get("response", "")

    preserved = {
        key: data[key] for key in
        ("entity", "prompt", "expected", "retry", "variant", "query_type",
         "prompting_method", "extended_reasoning_enabled", "original_prompt",
         "hinted_prompt", "expected_count", "hint_letters")
        if key in data
    }

    output = {
        **preserved,
        **result,
        "reval_truncated": True,
        "reval_signal": signal,
        "reval_max_tokens": reval_max_tokens,
        "reval_at": datetime.now(timezone.utc).isoformat(),
        "previous_response": data.get("response", ""),
        "previous_usage": data.get("usage"),
        "previous_finish_reason": data.get("finish_reason"),
        "previous_max_tokens_inferred": (data.get("usage") or {}).get("completion_tokens"),
    }

    if variant in SELFASK_TEMPLATES:
        trace = extract_selfask_trace(raw_response)
        output["response"] = trace["final_answer"]
        output["selfask_full_completion"] = raw_response
        output["selfask_trace"] = trace

    output["timestamp"] = datetime.now(timezone.utc).isoformat()
    response_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def reval_truncated_answers(model_root, model_name, judge_model, num_retries=5,
                            variants=None, token_threshold=2048, reval_max_tokens=4096,
                            temperature=0.3, include_length_finish=True,
                            max_workers=8, dry_run=False, backup_root=None):
    """Re-query items whose stored answer looks truncated AND whose eval failed.

    Selection requires both conditions: a truncation signal from the usage block
    (total tokens over token_threshold, or finish_reason == 'length') and
    correct == False in the corresponding evaluation file. That covers explicit
    judge/parse errors and plain wrong answers alike, since error entries always
    set correct=False. Selected items are re-queried with reval_max_tokens output
    tokens, then their evaluation is dropped so the matching evaluator re-runs on
    the fresh answer.

    Any strategy can be revalidated: list, MCQ (mc_*) and extended variants. The
    stored prompt, variant and extended-reasoning flag are reused per item, so a
    mixed selection re-queries each row exactly as it was originally issued.

    dry_run reports the selection and spends nothing.
    """
    model_root = Path(model_root)
    variants = resolve_rejudge_variants(variants)

    if backup_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = model_root / f"_reval_backup_{stamp}"
    else:
        backup_root = Path(backup_root)

    selected = []
    for fd in sorted(model_root.glob("fail_*")):
        if not fd.is_dir():
            continue

        for var in variants:
            response_dir, eval_dir = fd / f"binding_{var}", fd / f"eval_binding_{var}"
            if not response_dir.is_dir():
                continue

            for rf in sorted(response_dir.glob("*.json"))[:num_retries]:
                try:
                    data = json.loads(rf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    safe_print(f"  SKIP invalid response {rf}: {exc}")
                    continue

                signal = _is_truncated_usage(data, token_threshold, include_length_finish)
                if signal is None:
                    continue

                evf = eval_dir / f"eval_{rf.name}"
                if not evf.exists():
                    continue
                try:
                    eval_data = json.loads(evf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    safe_print(f"  SKIP invalid evaluation {evf}: {exc}")
                    continue

                evaluation = normalize_eval_record(eval_data)
                if evaluation["correct"]:
                    continue
                # Explicit judge/parse errors always set correct=False, so the
                # error field is reported when present but is not required.
                eval_error = evaluation["error"] or "incorrect"

                selected.append({
                    "fd": fd, "var": var, "response_file": rf, "eval_file": evf,
                    "signal": signal, "eval_error": eval_error,
                    "total_tokens": (data.get("usage") or {}).get("total_tokens"),
                    "finish_reason": data.get("finish_reason"),
                })

    if not selected:
        print(
            f"No items matched in variants {variants}: need a truncation signal "
            f"(total_tokens > {token_threshold} or finish_reason == 'length') AND "
            f"correct == False in the evaluation file."
        )
        return

    print(f"Selected {len(selected)} item(s) for re-query at max_tokens={reval_max_tokens}:")
    for item in selected:
        print(
            f"  {item['fd'].name}/binding_{item['var']}/{item['response_file'].name}"
            f"  signal={item['signal']}  eval_error={item['eval_error']}"
            f"  finish_reason={item['finish_reason']}"
        )

    if dry_run:
        print("\nDry run: nothing re-queried, nothing modified.")
        return

    def _backup(path: Path, move: bool):
        destination = backup_root / path.relative_to(model_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(path), str(destination))
        else:
            shutil.copy2(str(path), str(destination))

    dirty, requeried, failed = {}, 0, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in selected:
            _backup(item["response_file"], move=False)
            futures[executor.submit(
                _requery_item, item["response_file"], model_name, temperature,
                reval_max_tokens, item["signal"],
            )] = item

        for future, item in futures.items():
            label = f"{item['fd'].name}/binding_{item['var']}/{item['response_file'].name}"
            try:
                output = future.result()
                usage = output.get("usage") or {}
                safe_print(
                    f"  ✓ requeried {label}: finish_reason={output.get('finish_reason')} "
                    f"completion_tokens={usage.get('completion_tokens')} "
                    f"answer={str(output.get('response'))[:80]!r}"
                )
                requeried += 1
            except Exception as exc:
                safe_print(f"  ✗ requery FAILED {label}: {exc}")
                failed.append(label)
                continue

            key = (item["fd"], item["var"])
            eval_dir = item["fd"] / f"eval_binding_{item['var']}"
            preexisting = dirty.get(key, {}).get("preexisting")
            if preexisting is None:
                preexisting = {p.name for p in eval_dir.glob("eval_*.json")}
            if item["eval_file"].exists():
                _backup(item["eval_file"], move=True)
            dirty[key] = {"preexisting": preexisting}

    if not dirty:
        print("No items were successfully re-queried; evaluations left untouched.")
        return

    created_files = []
    for (fd, var), info in dirty.items():
        correct, total = reeval_variant(fd, judge_model, num_retries, var)
        eval_dir = fd / f"eval_binding_{var}"
        for evf in sorted(eval_dir.glob("eval_*.json")):
            if evf.name not in info["preexisting"]:
                created_files.append(evf.relative_to(model_root).as_posix())
        safe_print(f"  REJUDGED {fd.name}/binding_{var}: current_score={correct}/{total}")

    (backup_root / CREATED_FILES_MANIFEST_NAME).write_text(
        json.dumps({"created": created_files}, indent=2), encoding="utf-8"
    )

    print(f"\nRe-queried {requeried}/{len(selected)} item(s) at max_tokens={reval_max_tokens}.")
    if failed:
        print(f"{len(failed)} re-query failure(s); those responses were left as-is:")
        for label in failed:
            print(f"  {label}")
    print(f"Roll back with --restore-rejudge-backup {backup_root}")


def restore_rejudge_backup(model_root, backup_root):
    """Restore evaluation files from a backup created by rejudge or force mode.

    Files the pass created (no pre-run counterpart to copy back) are deleted using
    the backup's _created_files.json manifest, so the rollback is exact rather
    than merely overwriting what happens to overlap.
    """
    model_root = Path(model_root)
    backup_root = Path(backup_root)

    if not backup_root.exists():
        raise FileNotFoundError(f"Backup directory not found: {backup_root}")

    created_manifest = backup_root / CREATED_FILES_MANIFEST_NAME
    deleted = 0
    if created_manifest.exists():
        created = json.loads(created_manifest.read_text(encoding="utf-8")).get("created", [])
        for relative in created:
            target = model_root / relative
            if target.exists():
                target.unlink()
                deleted += 1

    restored = 0
    for source in sorted(backup_root.rglob("*.json")):
        if source.name in (SNAPSHOT_MANIFEST_NAME, CREATED_FILES_MANIFEST_NAME):
            continue
        destination = model_root / source.relative_to(backup_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        restored += 1

    print(f"Restored {restored} files from {backup_root}, deleted {deleted} newly created files")
    if not created_manifest.exists():
        print("  Note: this backup predates created-file tracking, so the restore may "
              "leave behind files the pass created.")


def archive_variant_outputs(model_root, list_variants, mc_variants, extended_variants,
                            backup_root=None):
    """Move existing response + evaluation dirs for the given variants aside.

    Used by --force to regenerate variants whose stored responses are stale
    (e.g. binding_mc_reason produced before extended reasoning was wired up).
    Moved, never deleted: the backup path is printed for rollback.
    """
    model_root = Path(model_root)
    if not model_root.exists():
        print(f"Nothing to archive, model root does not exist: {model_root}")
        return

    if backup_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = model_root / f"_force_backup_{stamp}"
    else:
        backup_root = Path(backup_root)

    dir_names = []
    for var in list_variants:
        dir_names += [f"binding_{var}", f"eval_binding_{var}"]
    for var in mc_variants:
        dir_names += [f"binding_mc_{var}", f"eval_binding_mc_{var}"]
    for var in extended_variants:
        dir_names += [f"binding_{var}", f"eval_binding_{var}"]

    moved_total = 0
    for fd in sorted(model_root.glob("fail_*")):
        if not fd.is_dir():
            continue
        for dir_name in dict.fromkeys(dir_names):
            source = fd / dir_name
            if not source.is_dir():
                continue
            destination = backup_root / source.relative_to(model_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved_total += 1

    if moved_total == 0:
        print("Nothing to archive for the requested variants.")
    else:
        print(f"Archived {moved_total} variant directories to: {backup_root}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_experiment(json_path, model_name, num_retries=5, num_entities=-1, temperature=0.0,
                               output_dir="experiment_results", max_tokens=100, threshold=0.6, 
                               judge_model="anthropic/claude-haiku-4.5", max_workers=28,
                               binding_variants=None):
    """
    Run binding failure evaluation experiment.

    binding_variants: list of variants to run. Options:
        - "raw"                : List-based instant (no CoT prompt)
        - "no_reason"          : List-based instant (different prompt)
        - "reason"             : List-based with reasoning prompt + extended thinking
        - "mc_raw"             : MCQ instant
        - "mc_no_reason"       : MCQ instant (different prompt)
        - "mc_reason"          : MCQ with reasoning + extended thinking
        - "selfask"            : Self-ask few-shot (list)
        - "selfask_modified_2" : Self-ask binding few-shot (list)

    Shortcuts:
        - None or []                : Run all variants
        - ["instant"]               : ["raw", "mc_raw"]
        - ["reasoning"]             : ["reason", "mc_reason"]
        - ["all_list"] / ["all_mc"] / ["all_extended"] / ["all"]
        - ["rerun_selfask_mcreason"]: ["selfask", "selfask_modified_2", "mc_reason"]
    """
    load_dotenv()

    with open(json_path) as f:
        test_cases = json.load(f)['test_cases']

    list_variants, mc_variants, extended_variants = resolve_binding_variants(binding_variants)

    SYS = SYSTEM_PROMPT

    ents = list(test_cases.items())[:num_entities] if num_entities > 0 else list(test_cases.items())
    result_dir = f"{output_dir}_temp{temperature}"
    mp = Path(result_dir) / model_name.replace(':', '_').replace('/', '_')
    mp.mkdir(parents=True, exist_ok=True)

    # Check if the model supports extended reasoning
    reasoning_supported = model_supports_reasoning(model_name)

    # Format variant info for display
    active_list = list_variants if list_variants else ["none"]
    active_mc = [f"mc_{v}" for v in mc_variants] if mc_variants else ["none"]
    active_extended = extended_variants if extended_variants else ["none"]

    print(f"\n{'='*60}")
    print(f"Processing {len(ents)} entities with max {max_workers} parallel API calls")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Judge: {judge_model}")
    print(f"Threshold: {threshold*100:.0f}%")
    print(f"List-based variants: {active_list}")
    print(f"MCQ variants: {active_mc}")
    print(f"Extended variants: {active_extended}")
    if reasoning_supported:
        print(f"✅ Model supports extended reasoning: {{enabled: True}}")
    else:
        print(f"⚠️  Model does NOT support extended reasoning - thinking will be skipped")
    print(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (entity, case) in enumerate(ents, 1):
            safe_print(f"\n=== [{i}/{len(ents)}] {entity} ===")
            try:
                passed = process_single_entity(
                    entity, case, mp, model_name, temperature, max_tokens,
                    SYS, num_retries, threshold, judge_model, executor,
                    list_variants=list_variants, mc_variants=mc_variants,
                    extended_variants=extended_variants
                )
                status = "✓ COMPLETE" if passed else "✗ FAILED THRESHOLD"
                safe_print(f"=== [{i}/{len(ents)}] {entity}: {status} ===\n")
            except Exception as e:
                safe_print(f"=== [{i}/{len(ents)}] {entity}: ✗ ERROR: {e} ===\n")

    print(f"\n{'='*60}")
    print(f"✅ DONE! Results in: {mp}")
    print(f"{'='*60}")




# TODO: these models
# anthropic/claude-opus-4.6
# openai/gpt-5.4
# x-ai/grok-4.20
# mistralai/mistral-large-2512
# google/gemini-3.1-pro-preview


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run binding failure evaluation")
    parser.add_argument(
        "--model", "-m", nargs="+",
        help=("One or more model labels/slugs. Legacy labels in MODEL_ALIASES "
              "keep their original output-directory names."),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run every legacy model label defined in MODEL_ALIASES.",
    )
    parser.add_argument(
        "--dataset", "-d", nargs="+", default=["athletes"],
        choices=["athletes", "custom", "all"],
        help="One or more datasets, or 'all' for athletes and custom.",
    )
    parser.add_argument("--variants", "-v", nargs="+",
                        default=["raw", "reason", "mc_raw"],
                        help="Binding variants to run (see resolve_binding_variants)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature; also selects the results directory")
    parser.add_argument(
        "--benchmark-dir", type=Path, default=BENCHMARK_DIR,
        help="Folder with binding_test_cases_{athletes,custom}.json (default: the repo's data/benchmark)",
    )
    parser.add_argument(
        "--judge-model", default="anthropic/claude-haiku-4.5",
        help="OpenRouter slug of the LLM judge (paper: anthropic/claude-haiku-4.5)",
    )
    parser.add_argument(
        "--reasoning-judge-model", default=None,
        help=("If set, list answers that come with a thinking trace are graded by this judge, which "
              "reads the trace as well as the final answer (in the paper: anthropic/claude-sonnet-4, "
              "used for part of the Reason trials of the thinking models). Default: off."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=("Archive existing response+eval dirs for the requested variants "
              "before running, forcing regeneration instead of SKIP"),
    )
    parser.add_argument(
        "--rejudge-not-found",
        action="store_true",
        help=("Only re-run the judge for error=not_found files in "
              "selfask and selfask_modified_2 evaluation directories"),
    )
    parser.add_argument(
        "--rejudge-parse-errors",
        action="store_true",
        help=("Re-extract self-ask answers from stored completions with the current "
              "parser, then rejudge those plus any unparsed_response / "
              "judge_parse_error / not_found evaluations"),
    )
    parser.add_argument(
        "--no-reextract",
        action="store_true",
        help="With --rejudge-parse-errors, rejudge only; do not rewrite responses",
    )
    parser.add_argument(
        "--rejudge-truncated-answers",
        action="store_true",
        help=("Rejudge evaluations whose response exceeded the token threshold AND "
              "whose evaluation carries an error, i.e. likely truncated mid-answer"),
    )
    parser.add_argument(
        "--truncation-token-threshold",
        type=int,
        default=2048,
        help="Total-token count above which a failed evaluation counts as truncated (default 2048)",
    )
    parser.add_argument(
        "--rejudge-variants",
        nargs="+",
        default=DEFAULT_REJUDGE_VARIANTS,
        help=(
            "Strategies included in every repair pass (--rejudge-not-found, "
            "--rejudge-parse-errors, --rejudge-truncated-answers, "
            "--reval-truncated-answers). Accepts directory tokens "
            f"{ALL_REJUDGE_VARIANTS} and the shortcuts "
            f"{sorted(REJUDGE_VARIANT_SHORTCUTS)}. Use 'all' for every strategy; "
            "MCQ arms must be written mc_<name>. "
            f"Default: {DEFAULT_REJUDGE_VARIANTS}"
        ),
    )
    parser.add_argument(
        "--reval-truncated-answers",
        action="store_true",
        help=("Re-query items whose stored answer looks truncated (total_tokens over "
              "--reval-token-threshold, or finish_reason == 'length') AND whose eval "
              "file records correct == False, using --reval-max-tokens, then rejudge"),
    )
    parser.add_argument(
        "--reval-max-tokens", type=int, default=4096,
        help="Output token budget for --reval-truncated-answers re-queries (default 4096)",
    )
    parser.add_argument(
        "--reval-token-threshold", type=int, default=2048,
        help="total_tokens above which a stored answer is treated as truncated (default 2048)",
    )
    parser.add_argument(
        "--reval-ignore-length-finish",
        action="store_true",
        help=("Select only on the total-token threshold; do not also select rows whose "
              "finish_reason is 'length'"),
    )
    parser.add_argument(
        "--reval-dry-run",
        action="store_true",
        help="List what --reval-truncated-answers would re-query, spending nothing",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help=("Output token budget for normal runs. Default 1024 preserves existing "
              "behavior; raise it for new self-ask runs."),
    )
    parser.add_argument(
        "--restore-rejudge-backup",
        type=Path,
        help="Restore an exact backup created by --rejudge-not-found or --force",
    )
    args = parser.parse_args()

    if args.all_models and args.model:
        parser.error("Use either --all-models or --model, not both.")
    if not args.all_models and not args.model:
        parser.error("Specify --model ... or use --all-models.")

    storage_models = list(MODEL_ALIASES) if args.all_models else args.model
    datasets = ["athletes", "custom"] if "all" in args.dataset else args.dataset

    # Avoid accidental duplicate runs while preserving user-specified order.
    storage_models = list(dict.fromkeys(storage_models))
    datasets = list(dict.fromkeys(datasets))

    if args.restore_rejudge_backup and (len(storage_models) != 1 or len(datasets) != 1):
        parser.error("--restore-rejudge-backup requires exactly one model and one dataset.")

    load_dotenv()
    temperature = args.temperature
    judge_model = args.judge_model
    global REASONING_JUDGE_MODEL
    REASONING_JUDGE_MODEL = args.reasoning_judge_model
    failures = []

    for storage_model in storage_models:
        api_model = resolve_openrouter_model(storage_model)

        for dataset in datasets:
            output_dir = f"binding_failures_{dataset}_rebuttal"
            model_root = (
                Path(f"{output_dir}_temp{temperature}")
                / storage_model.replace(':', '_').replace('/', '_')
            )

            print(f"\n{'#' * 72}")
            print(f"Storage model:   {storage_model}")
            print(f"OpenRouter slug: {api_model}")
            print(f"Dataset:         {dataset}")
            print(f"Output folder:   {model_root}")
            print(f"{'#' * 72}")

            try:
                if args.restore_rejudge_backup:
                    restore_rejudge_backup(model_root, args.restore_rejudge_backup)
                    continue

                if args.reval_truncated_answers:
                    reval_truncated_answers(
                        model_root=model_root,
                        model_name=storage_model,
                        judge_model=judge_model,
                        num_retries=5,
                        variants=args.rejudge_variants,
                        token_threshold=args.reval_token_threshold,
                        reval_max_tokens=args.reval_max_tokens,
                        temperature=temperature,
                        include_length_finish=not args.reval_ignore_length_finish,
                        dry_run=args.reval_dry_run,
                    )
                    continue

                if args.rejudge_truncated_answers:
                    rejudge_truncated_answers(
                        model_root=model_root,
                        judge_model=judge_model,
                        num_retries=5,
                        variants=args.rejudge_variants,
                        token_threshold=args.truncation_token_threshold,
                    )
                    continue

                if args.rejudge_parse_errors:
                    rejudge_parse_errors(
                        model_root=model_root,
                        judge_model=judge_model,
                        num_retries=5,
                        variants=args.rejudge_variants,
                        reextract=not args.no_reextract,
                    )
                    continue

                if args.rejudge_not_found:
                    rejudge_not_found_bindings(
                        model_root=model_root,
                        judge_model=judge_model,
                        num_retries=5,
                        variants=args.rejudge_variants,
                    )
                    continue

                if args.force:
                    list_variants, mc_variants, extended_variants = resolve_binding_variants(args.variants)
                    archive_variant_outputs(
                        model_root, list_variants, mc_variants, extended_variants
                    )

                run_evaluation_experiment(
                    json_path=args.benchmark_dir / f"binding_test_cases_{dataset}.json",
                    # Keep the legacy label here. call_openrouter resolves it to
                    # api_model, while run_evaluation_experiment uses it for the
                    # original output-directory name.
                    model_name=storage_model,
                    num_retries=5,
                    num_entities=-1,
                    temperature=temperature,
                    output_dir=output_dir,
                    max_tokens=args.max_tokens,
                    threshold=0.6,
                    judge_model=judge_model,
                    max_workers=28,
                    binding_variants=args.variants,
                )
            except Exception as exc:
                safe_print(
                    f"FAILED model={storage_model} api={api_model} "
                    f"dataset={dataset}: {exc}"
                )
                failures.append((storage_model, api_model, dataset, repr(exc)))

    if failures:
        print("\nFailed model/dataset runs:")
        for storage_model, api_model, dataset, error in failures:
            print(f"  - {storage_model} -> {api_model} / {dataset}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()