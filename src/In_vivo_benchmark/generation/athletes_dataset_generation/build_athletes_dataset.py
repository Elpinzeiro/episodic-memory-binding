#!/usr/bin/env python3
"""
TempLAMA Athletes Dataset Construction
=======================================

This script constructs the Athletes evaluation dataset for testing episodic 
knowledge binding in Large Language Models, as described in App. B.3.1.

Pipeline Overview (matching appendix methodology):
1. Load TempLAMA dataset (50,310 temporally scoped facts, 2010-2020)
2. Filter athletes with 2-3 unique team affiliations
3. Query Wikidata API for popularity (sitelink count) and sport type
4. Retain top 150 athletes by popularity
5. Generate verification questions (3 types per team) and binding questions
   with 6 prompt variants (Instant/Reasoning/No-Reasoning × List/MCQ)

Usage:
    python build_athletes_dataset.py --input templama.json --output athletes_dataset.json

References:
    - TempLAMA: Dhingra et al. (2022) "Time-Aware Language Models as Temporal Knowledge Bases"
    - Wikidata API: https://www.wikidata.org/wiki/Wikidata:Data_access
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset filtering parameters (as per App. B.3.1)
MIN_TEAMS = 2  # Minimum unique team affiliations
MAX_TEAMS = 3  # Maximum unique team affiliations  
TOP_K_ATHLETES = 150  # Number of top athletes to retain by popularity

# API rate limiting
API_DELAY_SECONDS = 0.3

# Cache file for Wikidata metadata (avoids repeated API calls)
DEFAULT_CACHE_FILE = "wikidata_cache.json"


# =============================================================================
# WIKIDATA API FUNCTIONS
# =============================================================================

def get_wikidata_id_from_entry(entry: dict) -> Optional[str]:
    """
    Extract Wikidata ID from TempLAMA entry.
    
    TempLAMA entries have IDs in format: Q123_P54_2010
    where Q123 is the Wikidata entity ID.
    
    Args:
        entry: TempLAMA entry dictionary
        
    Returns:
        Wikidata ID (e.g., 'Q11571') or None if not found
    """
    entry_id = entry.get("id", "")
    if "_" in entry_id:
        return entry_id.split("_")[0]
    return None


def get_sitelinks_count(wikidata_id: str) -> int:
    """
    Query Wikidata API to get the number of Wikipedia sitelinks for an entity.
    
    Sitelink count serves as a proxy for entity popularity - more popular
    entities have Wikipedia pages in more languages.
    
    Args:
        wikidata_id: Wikidata entity ID (e.g., 'Q11571' for Cristiano Ronaldo)
        
    Returns:
        Number of sitelinks (Wikipedia pages across languages)
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; Python-requests)',
        'Accept': 'application/json',
    }
    
    try:
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()
        sitelinks = data['entities'][wikidata_id].get('sitelinks', {})
        return len(sitelinks)
    except Exception as e:
        print(f"  [WARNING] Error fetching sitelinks for {wikidata_id}: {e}")
        return 0


def get_sport_from_wikidata(wikidata_id: str) -> Optional[str]:
    """
    Query Wikidata SPARQL endpoint to get the sport practiced by an athlete.
    
    Uses property P641 (sport) to determine sport type, which informs
    query phrasing (e.g., "play for" vs "race for" vs "represent").
    
    Args:
        wikidata_id: Wikidata entity ID
        
    Returns:
        Sport name (e.g., 'association football') or None if not found
    """
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?sportLabel WHERE {{
      wd:{wikidata_id} wdt:P641 ?sport .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    
    try:
        response = requests.get(
            sparql_url,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; Python-requests)"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["sportLabel"]["value"]
    except Exception as e:
        print(f"  [WARNING] Error fetching sport for {wikidata_id}: {e}")
    
    return None


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def load_cache(cache_file: str) -> dict:
    """Load cached Wikidata metadata from JSON file."""
    cache_path = Path(cache_file)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            # Migrate old format if needed (int -> dict)
            for wid, value in list(cache.items()):
                if isinstance(value, int):
                    cache[wid] = {"popularity": value, "sport": None}
            return cache
    return {}


def save_cache(cache: dict, cache_file: str) -> None:
    """Save Wikidata metadata cache to JSON file."""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# =============================================================================
# DATA LOADING AND FILTERING
# =============================================================================

def load_templama(filepath: str) -> list[dict]:
    """
    Load TempLAMA dataset from JSON lines file.
    
    TempLAMA contains temporally scoped facts with format:
    {"id": "Q123_P54_2010", "query": "Tom Brady plays for _X_.", 
     "answer": [{"name": "New England Patriots"}], "date": "2010"}
    
    Args:
        filepath: Path to TempLAMA JSON lines file
        
    Returns:
        List of TempLAMA entry dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_entity_name(query: str) -> Optional[str]:
    """
    Extract athlete name from TempLAMA query template.
    
    Args:
        query: TempLAMA query string (e.g., "Tom Brady plays for _X_.")
        
    Returns:
        Entity name or None if pattern doesn't match
    """
    if " plays for " in query:
        return query.split(" plays for")[0]
    return None


def group_by_entity(data: list[dict], min_teams: int, max_teams: int) -> dict:
    """
    Group TempLAMA entries by athlete and filter by team count.
    
    As per App. B.3.1: "retained those with 2-3 unique team affiliations
    across consecutive years"
    
    Args:
        data: List of TempLAMA entries
        min_teams: Minimum number of unique teams (inclusive)
        max_teams: Maximum number of unique teams (inclusive)
        
    Returns:
        Dictionary mapping entity names to their team/year data
    """
    entities = defaultdict(lambda: {"entries_by_year": {}})
    
    # Group entries by entity and year
    for entry in data:
        entity_name = extract_entity_name(entry["query"])
        if not entity_name:
            continue
        
        year = entry.get("date", "unknown")
        if year not in entities[entity_name]["entries_by_year"]:
            entities[entity_name]["entries_by_year"][year] = set()
        
        for ans in entry.get("answer", []):
            team = ans.get("name")
            if team:
                entities[entity_name]["entries_by_year"][year].add(team)
    
    # Filter entities by team count
    result = {}
    for entity, entity_data in entities.items():
        years = sorted([y for y in entity_data["entries_by_year"].keys() if y != "unknown"])
        if not years:
            continue
        
        # Collect all unique teams across all years
        all_teams = set()
        for year in years:
            all_teams.update(entity_data["entries_by_year"][year])
        
        # Apply team count filter
        if min_teams <= len(all_teams) <= max_teams:
            result[entity] = {
                "teams": all_teams,
                "date_range": (years[0], years[-1]),
                "entries": entity_data["entries_by_year"]
            }
    
    return result


def add_wikidata_metadata(
    raw_data: list[dict],
    entities: dict,
    cache_file: str = DEFAULT_CACHE_FILE
) -> dict:
    """
    Enrich entities with Wikidata metadata (popularity and sport).
    
    As per App. B.3.1: "queried the Wikidata API for each athlete's
    sitelink count (Wikipedia pages across languages)"
    
    Args:
        raw_data: Original TempLAMA data (for Wikidata ID mapping)
        entities: Filtered entities dictionary
        cache_file: Path to metadata cache file
        
    Returns:
        Entities dictionary enriched with 'popularity', 'sport', 'wikidata_id'
    """
    # Build entity name -> Wikidata ID mapping
    entity_to_wikidata = {}
    for entry in raw_data:
        entity_name = extract_entity_name(entry["query"])
        wikidata_id = get_wikidata_id_from_entry(entry)
        if entity_name and wikidata_id:
            entity_to_wikidata[entity_name] = wikidata_id
    
    # Load cache
    cache = load_cache(cache_file)
    print(f"Loaded {len(cache)} cached Wikidata entries")
    
    new_fetches = 0
    
    for entity_name in entities.keys():
        wikidata_id = entity_to_wikidata.get(entity_name)
        
        if not wikidata_id:
            entities[entity_name]["popularity"] = 0
            entities[entity_name]["sport"] = None
            print(f"  {entity_name}: No Wikidata ID found")
            continue
        
        # Check cache first
        if wikidata_id in cache:
            cached = cache[wikidata_id]
            sitelinks = cached["popularity"]
            sport = cached.get("sport")
            
            # Fetch sport if missing from cache
            if sport is None:
                sport = get_sport_from_wikidata(wikidata_id)
                cache[wikidata_id]["sport"] = sport
                new_fetches += 1
                time.sleep(API_DELAY_SECONDS)
                print(f"  {entity_name} ({wikidata_id}): {sitelinks} sitelinks, sport={sport} [SPORT FETCHED]")
            else:
                print(f"  {entity_name} ({wikidata_id}): {sitelinks} sitelinks, sport={sport} [CACHED]")
        else:
            # Fetch both from API
            sitelinks = get_sitelinks_count(wikidata_id)
            time.sleep(API_DELAY_SECONDS)
            sport = get_sport_from_wikidata(wikidata_id)
            time.sleep(API_DELAY_SECONDS)
            
            cache[wikidata_id] = {"popularity": sitelinks, "sport": sport}
            new_fetches += 1
            print(f"  {entity_name} ({wikidata_id}): {sitelinks} sitelinks, sport={sport} [FETCHED]")
        
        entities[entity_name]["popularity"] = cache[wikidata_id]["popularity"]
        entities[entity_name]["sport"] = cache[wikidata_id]["sport"]
        entities[entity_name]["wikidata_id"] = wikidata_id
    
    # Save updated cache
    if new_fetches > 0:
        save_cache(cache, cache_file)
        print(f"\nFetched {new_fetches} new Wikidata entries")
    
    return entities


def select_top_k_by_popularity(entities: dict, top_k: int) -> dict:
    """
    Select top K entities by popularity score.
    
    As per App. B.3.1: "retained the top 150 by this score"
    
    Args:
        entities: Entities dictionary with 'popularity' field
        top_k: Number of top entities to retain
        
    Returns:
        Filtered dictionary containing only top K entities
    """
    sorted_entities = sorted(
        entities.items(),
        key=lambda x: (-x[1].get("popularity", 0), -len(x[1]["teams"]))
    )
    return dict(sorted_entities[:top_k])


# =============================================================================
# SPORT-SPECIFIC TERMINOLOGY
# =============================================================================

# Sport categories for query phrasing (as per App. B.3.1)
MOTORSPORTS = {
    "Formula One", "Formula 1", "motorcycle racing", "NASCAR",
    "IndyCar", "rallying", "MotoGP", "World Rally Championship",
    "auto racing", "Formula One racing"
}

TEAM_SPORTS = {
    "association football", "basketball", "ice hockey", "rugby",
    "American football", "baseball", "handball", "volleyball",
    "rugby union", "rugby league", "cricket"
}


def has_national_team(teams: list[str]) -> bool:
    """Check if any team name contains 'national' (case-insensitive)."""
    pattern = re.compile(r'\bnational\b', re.IGNORECASE)
    return any(pattern.search(team) for team in teams)


def get_sport_terminology(sport: Optional[str], teams: list[str]) -> dict:
    """
    Get appropriate terminology based on sport type.
    
    As per App. B.3.1: "Query phrasing adapts to sport type from Wikidata
    ('play for' for team sports, 'race for' for motorsports, 'represent' otherwise)"
    
    Args:
        sport: Sport name from Wikidata
        teams: List of team names (to check for national teams)
        
    Returns:
        Dictionary with 'verb', 'verb_past', 'entity_type', 'binding_suffix'
    """
    if sport in MOTORSPORTS:
        return {
            "verb": "race for",
            "verb_past": "raced for",
            "entity_type": "teams/constructors",
            "binding_suffix": ""
        }
    elif sport in TEAM_SPORTS:
        # Only mention national teams if expected teams actually contain one
        suffix = ", including both club and national teams" if has_national_team(teams) else ""
        return {
            "verb": "play for",
            "verb_past": "played for",
            "entity_type": "teams",
            "binding_suffix": suffix
        }
    else:
        # Generic fallback for other sports (cycling, tennis, etc.)
        return {
            "verb": "represent",
            "verb_past": "represented",
            "entity_type": "teams/organizations",
            "binding_suffix": ""
        }


def shorten_team_name(team: str) -> str:
    """
    Shorten team names for multiple-choice options.
    
    Removes common suffixes like "F.C.", "national football team", etc.
    to create cleaner option labels.
    
    Args:
        team: Full team name
        
    Returns:
        Shortened team name
    """
    replacements = [
        (" national football team", ""),
        (" national association football team", ""),
        (" national basketball team", ""),
        (" national ice hockey team", ""),
        (" national rugby union team", ""),
        (" national team", ""),
        (" C.F.", ""),
        (" CF", ""),
        (" FC", ""),
        (" F.C.", ""),
        ("A.C. ", "AC "),
        ("S.S. ", ""),
        ("S.C. ", ""),
    ]
    result = team
    for old, new in replacements:
        result = result.replace(old, new)
    return result.strip()


# =============================================================================
# QUERY GENERATION
# =============================================================================

def prepare_test_cases(entities: dict) -> dict:
    """
    Generate test cases with verification and binding questions.
    
    For each athlete, generates:
    - 3 verification questions per team (yes/no, team retrieval, year retrieval)
    - 1 binding question with 6 prompt variants (3 list + 3 MCQ)
    
    Prompt variants (as per Section 4.3 of main paper):
    - List format: Instant (raw), Reasoning, No-Reasoning
    - MCQ format: Instant (raw), Reasoning, No-Reasoning
    
    Args:
        entities: Filtered and enriched entities dictionary
        
    Returns:
        Dictionary of test cases ready for evaluation
    """
    test_cases = {}
    
    for entity_name, info in entities.items():
        teams_list = list(info["teams"])
        start_year, end_year = info["date_range"]
        sport = info.get("sport")
        
        # Disambiguate "Ronaldo" (Nazário vs Cristiano)
        display_name = "Ronaldo Nazário" if entity_name == "Ronaldo" else entity_name
        
        # Get sport-appropriate terminology
        terms = get_sport_terminology(sport, teams_list)
        verb = terms["verb"]
        verb_past = terms["verb_past"]
        entity_type = terms["entity_type"]
        binding_suffix = terms["binding_suffix"]
        
        # =====================================================================
        # VERIFICATION QUESTIONS (3 types per team)
        # As per App. B.3.1 and Sec. 3.1
        # =====================================================================
        facts = {}
        for i, team in enumerate(teams_list):
            fact_key = f"fact_{chr(97 + i)}"  # fact_a, fact_b, fact_c, ...
            
            # Find a year when this team appears
            year = end_year if team in info["entries"].get(end_year, set()) else start_year
            for y in sorted(info["entries"].keys()):
                if team in info["entries"].get(y, set()):
                    year = y
                    break
            
            is_national = "national" in team.lower()
            
            # Type 1: Yes/No confirmation
            # "Did Ronaldo play for Manchester United in 2008?"
            facts[f"{fact_key}_yesno"] = {
                "prompt": f"In {year}, did {display_name} {verb} {team}? Answer yes or no.",
                "expected": "Yes",
                "team": team,
                "year": year,
                "type": "yes_no"
            }
            
            # Type 2: Team retrieval
            # "Which team did Ronaldo play for in 2008?"
            if is_national:
                team_question = f"Which national team did {display_name} {verb} in {year}?"
            else:
                team_question = f"Which team did {display_name} {verb} in {year}?"
            
            facts[f"{fact_key}_team"] = {
                "prompt": team_question,
                "expected": team,
                "team": team,
                "year": year,
                "type": "team_retrieval"
            }
            
            # Type 3: Year/period retrieval
            # "In which year did Ronaldo play for Manchester United?"
            facts[f"{fact_key}_period"] = {
                "prompt": f"In which period did {display_name} {verb} {team}?",
                "expected": year,
                "team": team,
                "year": year,
                "type": "period_retrieval"
            }
        
        # =====================================================================
        # BINDING QUESTIONS - LIST FORMAT (3 variants)
        # As per Section 4.3: Instant, Reasoning, No-Reasoning
        # =====================================================================
        base_binding = (
            f"List ALL {entity_type} that {display_name} {verb_past} between "
            f"{start_year} and {end_year}{binding_suffix}"
        )
        
        binding_prompts = {
            # Instant (raw): Direct question, minimal instruction
            "instant": f"{base_binding}. Provide your answer as a JSON list like [\"a\",\"b\"].",
            
            # No-Reasoning: Explicitly discourage chain-of-thought
            "no_reason": (
                f"{base_binding}. Provide your answer as a JSON list like [\"a\",\"b\"]. "
                f"No extra text, no reasoning!"
            ),
            
            # Reasoning: Encourage step-by-step thinking
            "reason": (
                f"{base_binding}. Provide your answer as a JSON list like [\"a\",\"b\"]. "
                f"Think about each event independently and reason before answering."
            ),
        }
        
        # =====================================================================
        # BINDING QUESTIONS - MULTIPLE CHOICE FORMAT (3 variants)
        # As per Section 4.3
        # =====================================================================
        teams_short = [shorten_team_name(t) for t in teams_list]
        
        # Build MCQ options
        options = {}
        option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Individual team options
        for i, team_short in enumerate(teams_short):
            options[option_letters[i]] = team_short
        
        # Correct answer: all teams combined
        all_teams_option = ", ".join(teams_short)
        correct_letter = option_letters[len(teams_short)]
        options[correct_letter] = all_teams_option
        
        # Distractor: "None of them"
        none_letter = option_letters[len(teams_short) + 1]
        options[none_letter] = "None of them"
        
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        mc_base = (
            f"Which {entity_type} did {display_name} {verb} between "
            f"{start_year} and {end_year}{binding_suffix}? Select the correct option:\n{options_text}"
        )
        
        multichoice_prompts = {
            # Instant (raw)
            "instant": f"{mc_base}\nReturn ONLY the letter.",
            
            # No-Reasoning
            "no_reason": (
                f"{mc_base}\nReturn ONLY the letter. "
                f"Do not write anything else, no explanation, no reasoning!"
            ),
            
            # Reasoning
            "reason": (
                f"{mc_base}\nReturn ONLY the letter. "
                f"Reflect about each event independently, take your time before answering, reason about it!"
            ),
        }
        
        # =====================================================================
        # ASSEMBLE TEST CASE
        # =====================================================================
        test_cases[entity_name] = {
            "binding": {
                "prompts": binding_prompts,
                "expected_teams": teams_list,
                "has_national_team": has_national_team(teams_list),
            },
            "binding_multichoice": {
                "prompts": multichoice_prompts,
                "options": options,
                "expected_answer": correct_letter,
                "expected_teams_short": teams_short,
            },
            "facts": facts,
            "metadata": {
                "num_teams": len(teams_list),
                "date_range": [start_year, end_year],
                "sport": sport,
                "wikidata_id": info.get("wikidata_id"),
                "popularity": info.get("popularity", 0),
            }
        }
    
    return test_cases


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_athletes_dataset(
    input_path: str,
    output_path: str,
    cache_file: str = DEFAULT_CACHE_FILE,
    min_teams: int = MIN_TEAMS,
    max_teams: int = MAX_TEAMS,
    top_k: int = TOP_K_ATHLETES,
) -> dict:
    """
    Execute the full dataset construction pipeline.
    
    Pipeline (as per App. B.3.1):
    1. Load TempLAMA source data
    2. Filter athletes with 2-3 team affiliations
    3. Add Wikidata metadata (popularity, sport)
    4. Select top 150 by popularity
    5. Generate verification and binding questions
    
    Args:
        input_path: Path to TempLAMA JSON lines file
        output_path: Path for output JSON file
        cache_file: Path to Wikidata metadata cache
        min_teams: Minimum team affiliations
        max_teams: Maximum team affiliations
        top_k: Number of top athletes to retain
        
    Returns:
        Complete dataset dictionary
    """
    print("=" * 70)
    print("TempLAMA Athletes Dataset Construction")
    print("=" * 70)
    
    # Step 1: Load source data
    print(f"\n[1/5] Loading TempLAMA from {input_path}...")
    data = load_templama(input_path)
    print(f"      Loaded {len(data)} entries")
    
    # Step 2: Filter by team count
    print(f"\n[2/5] Filtering athletes with {min_teams}-{max_teams} teams...")
    entities = group_by_entity(data, min_teams=min_teams, max_teams=max_teams)
    print(f"      Found {len(entities)} athletes matching criteria")
    
    # Step 3: Add Wikidata metadata
    print(f"\n[3/5] Fetching Wikidata metadata (popularity, sport)...")
    entities = add_wikidata_metadata(data, entities, cache_file)
    
    # Step 4: Select top K by popularity
    print(f"\n[4/5] Selecting top {top_k} athletes by popularity...")
    entities = select_top_k_by_popularity(entities, top_k)
    print(f"      Retained {len(entities)} athletes")
    
    # Show top 10 by popularity
    print("\n      Top 10 athletes by popularity:")
    for i, (name, info) in enumerate(list(entities.items())[:10]):
        print(f"        {i+1}. {name}: {info['popularity']} sitelinks, {len(info['teams'])} teams")
    
    # Step 5: Generate test cases
    print(f"\n[5/5] Generating verification and binding questions...")
    test_cases = prepare_test_cases(entities)
    
    # Count questions
    num_binding = len(test_cases)
    num_facts = sum(len(tc["facts"]) for tc in test_cases.values())
    
    print(f"      Generated {num_binding} binding test cases")
    print(f"      Generated {num_facts} fact verification questions")
    print(f"      Total: {num_binding * 6 + num_facts} questions (6 binding variants + facts)")
    
    # Assemble output
    output = {
        "metadata": {
            "source": "TempLAMA (Dhingra et al., 2022)",
            "description": "Athletes dataset for episodic knowledge binding evaluation",
            "pipeline": {
                "min_teams": min_teams,
                "max_teams": max_teams,
                "top_k_athletes": top_k,
            },
            "statistics": {
                "num_athletes": len(test_cases),
                "num_binding_questions": num_binding,
                "num_fact_questions": num_facts,
                "prompt_variants": {
                    "list": ["instant", "no_reason", "reason"],
                    "mcq": ["instant", "no_reason", "reason"],
                }
            }
        },
        "test_cases": test_cases,
    }
    
    # Save output
    print(f"\n[DONE] Saving dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print("Dataset construction complete!")
    print(f"{'=' * 70}")
    
    return output


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Athletes dataset for episodic binding evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_athletes_dataset.py --input templama.json --output athletes.json
  python build_athletes_dataset.py --input templama.json --output athletes.json --top-k 100
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to TempLAMA JSON lines file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for output JSON file"
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE_FILE,
        help=f"Path to Wikidata metadata cache (default: {DEFAULT_CACHE_FILE})"
    )
    parser.add_argument(
        "--min-teams",
        type=int,
        default=MIN_TEAMS,
        help=f"Minimum team affiliations (default: {MIN_TEAMS})"
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        default=MAX_TEAMS,
        help=f"Maximum team affiliations (default: {MAX_TEAMS})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_ATHLETES,
        help=f"Number of top athletes by popularity (default: {TOP_K_ATHLETES})"
    )
    
    args = parser.parse_args()
    
    build_athletes_dataset(
        input_path=args.input,
        output_path=args.output,
        cache_file=args.cache,
        min_teams=args.min_teams,
        max_teams=args.max_teams,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()