"""
Quality-Based Filtering - Test Case Evaluation
==============================================
Comprehensive evaluation using Claude Opus 4.5 as judge model for:
1. Quality assessment (self-referential, answer-in-question, etc.)
2. Duplicate detection
3. Prompt conditioning (ambiguous or poorly conditioned prompts)
4. Response completeness (for model responses - optional if you have response data)

Outputs: JSON files with LLM evaluations (no filtering applied)
"""

import json
import os
import re
import time
from pathlib import Path
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv
import anthropic

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths - UPDATE THESE TO YOUR LOCAL PATHS
DATASET_JSON = "./test_cases.json"
OUTPUT_DIR = "./quality_evaluation_results"

# Optional: Set this if you have model responses to evaluate for completeness
MODEL_RESPONSES_DIR = None  # e.g., "./model_responses" or None to skip

# LLM Judge Settings
JUDGE_MODEL = "claude-opus-4-5-20251101"
BATCH_SIZE = 10  # Number of items to evaluate per API call
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls

# Duplicate detection - similarity thresholds for heuristic pre-filtering
SIMILARITY_THRESHOLD = 0.75

# =============================================================================
# PROMPTS
# =============================================================================

QUALITY_EVALUATION_PROMPT = """You are evaluating test cases designed to measure if AI models can recall multiple related facts together.

## WHAT MAKES A GOOD TEST CASE:

A good test case has:
1. Multiple SEPARATE real-world facts (2+) that share something in common (same time period, same category, same event type)
2. Individual verification questions for each fact (yes/no, "which X did Y", "when did Z happen")
3. A BINDING question that asks the model to list ALL items matching a criteria (e.g., "List ALL companies that did X in 2022")
4. Multiple choice options where wrong answers are plausible but incorrect

## CRITICAL FLAWS (score = 0 if found):

**FLAW 1 - SELF-REFERENCE**: The binding question says "in this test case", "mentioned above", "described earlier". This is INVALID because it asks about the test itself, not real-world knowledge.
- BAD: "List ALL missions in this test case"
- GOOD: "List ALL NASA missions launched in March 2020"

**FLAW 2 - SINGLE EVENT**: All facts describe ONE event from different angles, not multiple separate events.
- BAD: Facts about one wedding (date + location + venue) - this is 1 event with 3 attributes
- GOOD: Facts about three different weddings in the same month - these are 3 separate events

**FLAW 3 - INCOMPLETE COVERAGE**: The expected answer includes items that have no verification questions.
- BAD: Expected ["A", "B", "C"] but only facts for A and B exist
- GOOD: Expected ["A", "B", "C"] with verification facts for all three

**FLAW 4 - WRONG METADATA**: The "num_items" field doesn't match actual number of distinct items, or title mentions wrong time period.

## SCORING CRITERIA (0.0 to 1.0 each):

1. **no_self_reference**: 1.0 if binding question asks about real-world facts, 0.0 if it references "this test case"
2. **multiple_distinct_events**: 1.0 if 2+ separate real-world events, 0.5 if borderline, 0.0 if single event
3. **complete_fact_coverage**: 1.0 if all expected items have verification questions, 0.5 if some missing
4. **good_distractors**: 1.0 if wrong MC options are plausible confusions, 0.5 if too obvious, 0.0 if nonsensical
5. **correct_metadata**: 1.0 if num_items and dates match facts, 0.5 if minor issues, 0.0 if wrong

## RESPOND WITH ONLY THIS JSON (no other text):

{
  "critical_flaw": null or "SELF_REFERENCE" or "SINGLE_EVENT" or "INCOMPLETE_COVERAGE" or "WRONG_METADATA",
  "flaw_evidence": null or "quote the problematic text",
  "scores": {
    "no_self_reference": 0.0-1.0,
    "multiple_distinct_events": 0.0-1.0,
    "complete_fact_coverage": 0.0-1.0,
    "good_distractors": 0.0-1.0,
    "correct_metadata": 0.0-1.0
  },
  "overall_score": 0.0-1.0,
  "issues": ["list specific problems"],
  "quality": "EXCELLENT" or "GOOD" or "NEEDS_REVISION" or "INVALID"
}

## TEST CASE TO EVALUATE:
"""

DUPLICATE_DETECTION_PROMPT = """Evaluate if each pair of test cases are DUPLICATES (testing essentially the same thing).

DUPLICATE if:
- Both test the same event/fact with same or very similar prompts
- One is a subset or variant of the other
- They would have the same expected answer for the same reason

NOT DUPLICATE if:
- Different events even if similar names (e.g., "2016 Grammy" vs "2017 Grammy")
- Same topic but testing different aspects
- Different time periods or contexts

{formatted_pairs}

Return JSON: [{{"id": 0, "verdict": "duplicate|not_duplicate", "reason": "brief explanation"}}]"""

PROMPT_CONDITIONING_PROMPT = """Evaluate if each prompt is BAD-CONDITIONED (unfair or impossible to answer correctly).

A prompt is BAD if:
1. **Ambiguous scope**: Doesn't clearly define what to include/exclude
2. **Missing context**: References "this case" or assumes prior context not in prompt
3. **Expected mismatch**: Expected answer doesn't match what prompt asks

A prompt is GOOD if:
- Clear, specific, and self-contained
- Expected answer matches what's being asked

{formatted_prompts}

Return JSON: [{{"id": 0, "verdict": "good|bad", "issue": "ambiguous_scope|missing_context|overly_specific|expected_mismatch|temporal_ambiguity|none", "explanation": "brief why"}}]"""

RESPONSE_COMPLETENESS_PROMPT = """Evaluate if each reasoning response is INCOMPLETE.

A response is INCOMPLETE if:
1. **Truncated**: Cut off mid-sentence or mid-thought (often due to max_tokens)
2. **No conclusion**: Has reasoning but doesn't give final answer
3. **Abandoned reasoning**: Starts reasoning process but doesn't finish
4. **Empty/minimal**: Too short to contain meaningful reasoning

A response is COMPLETE if:
- Contains reasoning AND final answer
- Reaches a conclusion (even if wrong)
- Self-contained and finished thought

{formatted_responses}

Return JSON: [{{"id": 0, "verdict": "incomplete|complete", "issue": "truncated|no_conclusion|abandoned|empty|none", "explanation": "brief"}}]"""

# =============================================================================
# LLM CLIENT SETUP
# =============================================================================

def initialize_client():
    """Initialize Anthropic client with API key."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    
    print(f"✅ Initialized LLM Judge: {JUDGE_MODEL}")
    return client


def call_judge(client, prompt: str, max_tokens: int = 8192) -> str:
    """Call the judge model with the given prompt."""
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_json_response(response: str):
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parse error: {e}")
        print(f"  Response preview: {text[:200]}...")
        return None


# =============================================================================
# LOAD DATA
# =============================================================================

def load_test_cases(dataset_path):
    """Load test cases from JSON file."""
    print("\n" + "="*70)
    print("LOADING TEST CASES")
    print("="*70)
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Handle different JSON structures
    test_cases = data.get('test_cases', data)
    
    print(f"Loaded {len(test_cases)} test cases from {dataset_path}")
    return test_cases


# =============================================================================
# 1. QUALITY EVALUATION
# =============================================================================

def evaluate_quality(client, test_cases, output_path):
    """Evaluate quality of each test case using LLM judge."""
    print("\n" + "="*70)
    print("1. EVALUATING TEST CASE QUALITY")
    print("="*70)
    print("(Detects: self-referential examples, answer-in-question, and other quality issues)")
    
    results = []
    total = len(test_cases)
    
    for idx, (name, tc) in enumerate(test_cases.items(), 1):
        print(f"  [{idx}/{total}] Evaluating: {name}")
        
        # Prepare prompt with test case
        prompt = QUALITY_EVALUATION_PROMPT + json.dumps(tc, indent=2)
        
        try:
            # Call judge
            response = call_judge(client, prompt)
            
            # Parse response
            evaluation = parse_json_response(response)
            
            if evaluation:
                evaluation["test_case_name"] = name
                evaluation["timestamp"] = datetime.now().isoformat()
                results.append(evaluation)
                
                score = evaluation.get('overall_score', -1)
                quality = evaluation.get('quality', 'UNKNOWN')
                print(f"    Score: {score:.2f} | Quality: {quality}")
                
                # Show critical flaws
                flaw = evaluation.get('critical_flaw')
                if flaw:
                    print(f"    ⚠️ CRITICAL FLAW: {flaw}")
                    evidence = evaluation.get('flaw_evidence', '')
                    if evidence:
                        print(f"       Evidence: {evidence[:80]}...")
            else:
                print(f"    ❌ Failed to parse response")
                results.append({
                    "test_case_name": name,
                    "error": "parse_failed",
                    "raw_response": response[:500],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "test_case_name": name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results
    output_file = output_path / "quality_evaluations.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved quality evaluations to: {output_file}")
    
    # Summary statistics
    valid_scores = [r['overall_score'] for r in results if 'overall_score' in r]
    if valid_scores:
        print(f"\nQuality Summary:")
        print(f"  Mean score: {sum(valid_scores)/len(valid_scores):.2f}")
        print(f"  Min score: {min(valid_scores):.2f}")
        print(f"  Max score: {max(valid_scores):.2f}")
        
        # Count by quality category
        quality_counts = {}
        for r in results:
            q = r.get('quality', 'UNKNOWN')
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        print(f"\nBy quality category:")
        for quality, count in sorted(quality_counts.items()):
            print(f"  {quality}: {count}")
        
        # Count critical flaws
        flaw_counts = {}
        for r in results:
            flaw = r.get('critical_flaw')
            if flaw:
                flaw_counts[flaw] = flaw_counts.get(flaw, 0) + 1
        
        if flaw_counts:
            print(f"\nCritical flaws found:")
            for flaw, count in sorted(flaw_counts.items()):
                print(f"  {flaw}: {count}")
    
    return results


# =============================================================================
# 2. DUPLICATE DETECTION (HEURISTIC + LLM)
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_key_terms(name: str) -> set:
    """Extract key terms from test case name."""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    words = normalize_text(name).split()
    return set(w for w in words if w not in stopwords and len(w) > 2)


def detect_duplicates(client, test_cases, output_path):
    """Detect duplicate test cases using heuristics + LLM judge."""
    print("\n" + "="*70)
    print("2. DETECTING DUPLICATE TEST CASES")
    print("="*70)
    
    # Build comparison data
    tc_data = []
    for name, tc in test_cases.items():
        binding = tc.get('binding', {})
        prompts = binding.get('prompts', {})
        raw_prompt = prompts.get('raw', '')
        
        tc_data.append({
            "name": name,
            "name_normalized": normalize_text(name),
            "key_terms": extract_key_terms(name),
            "prompt": raw_prompt,
            "prompt_normalized": normalize_text(raw_prompt),
            "topic": tc.get('topic', ''),
            "expected": binding.get('expected', [])
        })
    
    # Find duplicates by comparing name AND prompt (heuristic pre-filtering)
    print("  Running heuristic pre-filtering...")
    duplicates_heuristic = []
    checked_pairs = set()
    
    for i, tc1 in enumerate(tc_data):
        for j, tc2 in enumerate(tc_data):
            if i >= j:
                continue
            
            pair_key = tuple(sorted([tc1['name'], tc2['name']]))
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            # Calculate similarities
            name_sim = similarity(tc1['name'], tc2['name'])
            prompt_sim = similarity(tc1['prompt'], tc2['prompt'])
            
            # Check key term overlap
            term_overlap = len(tc1['key_terms'] & tc2['key_terms'])
            term_union = len(tc1['key_terms'] | tc2['key_terms'])
            term_jaccard = term_overlap / term_union if term_union > 0 else 0
            
            # Flag as potential duplicate if any similarity is notable
            is_candidate = False
            
            if name_sim >= SIMILARITY_THRESHOLD and prompt_sim >= SIMILARITY_THRESHOLD:
                is_candidate = True
            elif name_sim >= 0.9:
                is_candidate = True
            elif term_jaccard >= 0.7 and prompt_sim >= 0.6:
                is_candidate = True
            elif name_sim >= 0.5 and prompt_sim >= 0.5:  # Lower threshold for LLM review
                is_candidate = True
            
            if is_candidate:
                duplicates_heuristic.append({
                    "test_case_1": tc1['name'],
                    "test_case_2": tc2['name'],
                    "name_similarity": round(name_sim, 3),
                    "prompt_similarity": round(prompt_sim, 3),
                    "term_jaccard": round(term_jaccard, 3),
                    "prompt_1": tc1['prompt'],
                    "prompt_2": tc2['prompt'],
                    "topic_1": tc1['topic'],
                    "topic_2": tc2['topic'],
                })
    
    print(f"  Heuristic candidates: {len(duplicates_heuristic)} pairs")
    
    # LLM evaluation for duplicate candidates
    duplicates = []
    all_duplicate_evaluations = []
    
    if duplicates_heuristic:
        print("  Running LLM evaluation on duplicate candidates...")
        
        for batch_idx in range(0, len(duplicates_heuristic), BATCH_SIZE):
            batch = duplicates_heuristic[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            total_batches = (len(duplicates_heuristic) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"    Batch {batch_num}/{total_batches} ({len(batch)} pairs)...")
            
            # Format pairs for prompt
            formatted_pairs = "\n\n".join([
                f"[{i}]\nTest Case 1: {d['test_case_1']}\nTopic 1: {d['topic_1']}\nPrompt 1: {d['prompt_1']}\n\nTest Case 2: {d['test_case_2']}\nTopic 2: {d['topic_2']}\nPrompt 2: {d['prompt_2']}"
                for i, d in enumerate(batch)
            ])
            
            prompt = DUPLICATE_DETECTION_PROMPT.format(formatted_pairs=formatted_pairs)
            
            try:
                # Call judge
                response = call_judge(client, prompt)
                
                # Parse response
                results = parse_json_response(response)
                
                if results and isinstance(results, list):
                    for r in results:
                        idx = r.get("id", -1)
                        if 0 <= idx < len(batch):
                            full_result = {
                                **batch[idx],
                                "verdict": r.get("verdict", "unknown"),
                                "reason": r.get("reason", ""),
                                "timestamp": datetime.now().isoformat()
                            }
                            all_duplicate_evaluations.append(full_result)
                            
                            # If confirmed duplicate, add to duplicates list
                            if r.get("verdict") == "duplicate":
                                duplicates.append(full_result)
                    
                    duplicates_in_batch = sum(1 for r in results if r.get("verdict") == "duplicate")
                    print(f"      Found {duplicates_in_batch} duplicates in this batch")
                else:
                    print(f"      ❌ Failed to parse response")
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print(f"  LLM confirmed: {len(duplicates)} duplicates")
    else:
        print("  No candidates found by heuristics")
    
    # Sort duplicates by combined similarity
    duplicates.sort(key=lambda x: (x['name_similarity'] + x['prompt_similarity']), reverse=True)
    
    # Save all duplicate evaluations
    output_file = output_path / "duplicate_evaluations_all.json"
    with open(output_file, 'w') as f:
        json.dump(all_duplicate_evaluations, f, indent=2)
    print(f"\n✅ Saved all duplicate evaluations to: {output_file}")
    
    # Save confirmed duplicates
    duplicates_file = output_path / "duplicates_confirmed.json"
    with open(duplicates_file, 'w') as f:
        json.dump(duplicates, f, indent=2)
    print(f"✅ Saved {len(duplicates)} confirmed duplicates to: {duplicates_file}")
    
    # Show examples
    if duplicates:
        print(f"\nExample duplicates found:")
        for dup in duplicates[:3]:
            print(f"\n  • {dup['test_case_1']}")
            print(f"    {dup['test_case_2']}")
            print(f"    Reason: {dup['reason']}")
            print(f"    Similarities: name={dup['name_similarity']:.2f}, prompt={dup['prompt_similarity']:.2f}")
    
    return all_duplicate_evaluations, duplicates


# =============================================================================
# 3. PROMPT CONDITIONING EVALUATION
# =============================================================================

def evaluate_prompt_conditioning(client, test_cases, output_path):
    """Evaluate if prompts are well-conditioned or ambiguous."""
    print("\n" + "="*70)
    print("3. EVALUATING PROMPT CONDITIONING")
    print("="*70)
    print("(Detects: ambiguous scope, missing context, expected mismatch)")
    
    # Prepare all prompts for evaluation
    all_prompts = []
    for name, tc in test_cases.items():
        binding = tc.get('binding', {})
        prompts = binding.get('prompts', {})
        expected = binding.get('expected', [])
        raw_prompt = prompts.get('raw', '')
        
        all_prompts.append({
            "test_case": name,
            "prompt": raw_prompt,
            "expected": expected,
            "notes": tc.get('notes', ''),
            "topic": tc.get('topic', '')
        })
    
    print(f"  Total test cases to evaluate: {len(all_prompts)}")
    
    # Evaluate in batches
    all_results = []
    bad_prompts = []
    
    print("  Running LLM evaluation...")
    
    for batch_idx in range(0, len(all_prompts), BATCH_SIZE):
        batch = all_prompts[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        if batch_num % 5 == 0 or batch_num == 1:
            print(f"    Processing batch {batch_num}/{total_batches}...")
        
        # Format prompts for evaluation
        formatted_prompts = "\n\n".join([
            f"[{i}]\nTest: {p['test_case']}\nTopic: {p['topic']}\nPrompt: {p['prompt']}\nExpected: {p['expected']}\nNotes: {p['notes']}"
            for i, p in enumerate(batch)
        ])
        
        prompt = PROMPT_CONDITIONING_PROMPT.format(formatted_prompts=formatted_prompts)
        
        try:
            # Call judge
            response = call_judge(client, prompt)
            
            # Parse response
            results = parse_json_response(response)
            
            if results and isinstance(results, list):
                for r in results:
                    idx = r.get("id", -1)
                    if 0 <= idx < len(batch):
                        full_result = {
                            **batch[idx],
                            "verdict": r.get("verdict", "unknown"),
                            "issue": r.get("issue", "none"),
                            "explanation": r.get("explanation", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(full_result)
                        
                        # If bad prompt, add to bad_prompts list
                        if r.get("verdict") == "bad":
                            bad_prompts.append(full_result)
            else:
                print(f"      ❌ Failed to parse response for batch {batch_num}")
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"      ❌ Error in batch {batch_num}: {e}")
    
    print(f"  Evaluation complete: {len(all_results)} prompts evaluated")
    print(f"  LLM identified: {len(bad_prompts)} bad-conditioned prompts")
    
    # Group by issue type
    issue_counts = Counter()
    for bp in bad_prompts:
        issue = bp.get('issue', 'unknown')
        issue_counts[issue] += 1
    
    if issue_counts:
        print("\n  By issue type:")
        for issue, count in issue_counts.most_common():
            print(f"    - {issue}: {count}")
    
    # Save all evaluations
    output_file = output_path / "prompt_conditioning_all.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved all prompt evaluations to: {output_file}")
    
    # Save bad prompts only
    bad_prompts_file = output_path / "bad_conditioned_prompts.json"
    with open(bad_prompts_file, 'w') as f:
        json.dump(bad_prompts, f, indent=2)
    print(f"✅ Saved {len(bad_prompts)} bad-conditioned prompts to: {bad_prompts_file}")
    
    # Show examples
    if bad_prompts:
        print(f"\nExample bad-conditioned prompts:")
        for bp in bad_prompts[:3]:
            print(f"\n  • {bp['test_case']}")
            print(f"    Issue: {bp['issue']}")
            print(f"    Prompt: {bp['prompt'][:80]}...")
            print(f"    Why: {bp['explanation']}")
    
    return all_results, bad_prompts


# =============================================================================
# 4. RESPONSE COMPLETENESS EVALUATION (OPTIONAL)
# =============================================================================

def evaluate_response_completeness(client, responses_dir, output_path):
    """Evaluate completeness of model responses (optional - requires response data)."""
    print("\n" + "="*70)
    print("4. EVALUATING RESPONSE COMPLETENESS")
    print("="*70)
    print("(Detects: truncated, incomplete, or abandoned reasoning)")
    
    if responses_dir is None or not Path(responses_dir).exists():
        print("  ⚠️ Skipped: No model responses directory provided")
        print("     Set MODEL_RESPONSES_DIR in config to enable this evaluation")
        return [], []
    
    # Load model responses
    # This is a placeholder - adjust based on your actual data structure
    responses_dir = Path(responses_dir)
    all_responses = []
    
    # Example: Load from JSON files
    for response_file in responses_dir.glob("*.json"):
        try:
            with open(response_file) as f:
                data = json.load(f)
                all_responses.append({
                    "test_case": response_file.stem,
                    "response": data.get('response', ''),
                    "prompt": data.get('prompt', ''),
                    "expected": data.get('expected', ''),
                    "stop_reason": data.get('stop_reason', '')
                })
        except Exception as e:
            print(f"    ⚠️ Error loading {response_file}: {e}")
    
    if not all_responses:
        print(f"  No responses found in {responses_dir}")
        return [], []
    
    print(f"  Loaded {len(all_responses)} responses to evaluate")
    
    # Evaluate in batches
    all_results = []
    incomplete_responses = []
    
    print("  Running LLM evaluation...")
    
    for batch_idx in range(0, len(all_responses), BATCH_SIZE):
        batch = all_responses[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(all_responses) + BATCH_SIZE - 1) // BATCH_SIZE
        
        if batch_num % 5 == 0 or batch_num == 1:
            print(f"    Processing batch {batch_num}/{total_batches}...")
        
        # Format responses for evaluation
        formatted_responses = "\n\n".join([
            f"[{i}]\nTest: {r['test_case']}\nPrompt: {r['prompt']}\nExpected: {r['expected']}\nResponse: {r['response']}\nStop Reason: {r['stop_reason']}"
            for i, r in enumerate(batch)
        ])
        
        prompt = RESPONSE_COMPLETENESS_PROMPT.format(formatted_responses=formatted_responses)
        
        try:
            # Call judge
            response = call_judge(client, prompt)
            
            # Parse response
            results = parse_json_response(response)
            
            if results and isinstance(results, list):
                for r in results:
                    idx = r.get("id", -1)
                    if 0 <= idx < len(batch):
                        full_result = {
                            **batch[idx],
                            "verdict": r.get("verdict", "unknown"),
                            "issue": r.get("issue", "none"),
                            "explanation": r.get("explanation", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(full_result)
                        
                        # If incomplete, add to incomplete_responses list
                        if r.get("verdict") == "incomplete":
                            incomplete_responses.append(full_result)
            else:
                print(f"      ❌ Failed to parse response for batch {batch_num}")
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"      ❌ Error in batch {batch_num}: {e}")
    
    print(f"  Evaluation complete: {len(all_results)} responses evaluated")
    print(f"  LLM identified: {len(incomplete_responses)} incomplete responses")
    
    # Group by issue type
    issue_counts = Counter()
    for ir in incomplete_responses:
        issue = ir.get('issue', 'unknown')
        issue_counts[issue] += 1
    
    if issue_counts:
        print("\n  By issue type:")
        for issue, count in issue_counts.most_common():
            print(f"    - {issue}: {count}")
    
    # Save all evaluations
    output_file = output_path / "response_completeness_all.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved all response evaluations to: {output_file}")
    
    # Save incomplete responses only
    incomplete_file = output_path / "incomplete_responses.json"
    with open(incomplete_file, 'w') as f:
        json.dump(incomplete_responses, f, indent=2)
    print(f"✅ Saved {len(incomplete_responses)} incomplete responses to: {incomplete_file}")
    
    # Show examples
    if incomplete_responses:
        print(f"\nExample incomplete responses:")
        for ir in incomplete_responses[:3]:
            print(f"\n  • {ir['test_case']}")
            print(f"    Issue: {ir['issue']}")
            print(f"    Why: {ir['explanation']}")
            print(f"    Response preview: {ir['response'][:80]}...")
    
    return all_results, incomplete_responses


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("QUALITY-BASED FILTERING - COMPREHENSIVE EVALUATION")
    print("="*70)
    print(f"\nJudge Model: {JUDGE_MODEL}")
    print(f"Dataset: {DATASET_JSON}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\nThis script evaluates:")
    print("  1. Quality issues (self-referential, answer-in-question, etc.)")
    print("  2. Duplicate test cases")
    print("  3. Prompt conditioning (ambiguous or poorly conditioned)")
    print("  4. Response completeness (optional, if response data available)")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = initialize_client()
    
    # Load test cases
    test_cases = load_test_cases(DATASET_JSON)
    
    # Run evaluations
    quality_results = evaluate_quality(client, test_cases, output_path)
    duplicate_results, duplicates = detect_duplicates(client, test_cases, output_path)
    conditioning_results, bad_prompts = evaluate_prompt_conditioning(client, test_cases, output_path)
    completeness_results, incomplete_responses = evaluate_response_completeness(client, MODEL_RESPONSES_DIR, output_path)
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTest cases evaluated: {len(test_cases)}")
    print(f"\n1. Quality Evaluations:")
    print(f"   - Total evaluated: {len(quality_results)}")
    
    # Calculate quality statistics
    low_quality = sum(1 for r in quality_results if r.get('overall_score', 1.0) < 0.3)
    critical_flaws = sum(1 for r in quality_results if r.get('critical_flaw') is not None)
    print(f"   - Score < 0.3: {low_quality}")
    print(f"   - With critical flaws: {critical_flaws}")
    
    print(f"\n2. Duplicate Detection:")
    print(f"   - Pairs checked: {len(duplicate_results)}")
    print(f"   - Duplicates confirmed: {len(duplicates)}")
    print(f"   - Test cases involved: {len(set([d['test_case_1'] for d in duplicates] + [d['test_case_2'] for d in duplicates]))}")
    
    print(f"\n3. Prompt Conditioning:")
    print(f"   - Total evaluated: {len(conditioning_results)}")
    print(f"   - Bad-conditioned: {len(bad_prompts)}")
    
    print(f"\n4. Response Completeness:")
    if completeness_results:
        print(f"   - Total evaluated: {len(completeness_results)}")
        print(f"   - Incomplete: {len(incomplete_responses)}")
    else:
        print(f"   - Skipped (no response data)")
    
    print(f"\nAll results saved to: {output_path}/")
    
    # Save comprehensive summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "dataset": DATASET_JSON,
        "total_test_cases": len(test_cases),
        "quality_evaluation": {
            "total": len(quality_results),
            "low_quality_count": low_quality,
            "critical_flaws_count": critical_flaws
        },
        "duplicate_detection": {
            "pairs_checked": len(duplicate_results),
            "duplicates_confirmed": len(duplicates)
        },
        "prompt_conditioning": {
            "total": len(conditioning_results),
            "bad_conditioned": len(bad_prompts)
        },
        "response_completeness": {
            "total": len(completeness_results),
            "incomplete": len(incomplete_responses)
        },
        "output_directory": str(output_path)
    }
    
    with open(output_path / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {output_path}/evaluation_summary.json")


if __name__ == '__main__':
    main()