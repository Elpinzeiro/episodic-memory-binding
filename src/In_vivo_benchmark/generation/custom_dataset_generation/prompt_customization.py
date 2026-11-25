#!/usr/bin/env python3
"""
Automated Knowledge Binding Dataset Generator - Version 2
Updated to match exact JSON format from original examples
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# ============================================================================
# THEMES DEFINITION
# ============================================================================

THEMES = {
    "company_acquisitions": {
        "name": "Company Acquisitions",
        "description": "Major corporate acquisitions and mergers",
        "entity_type": "Companies",
        "fact_focus": "acquisition price, announcement date, acquiring company, target company",
        "example_facts": "Microsoft acquired LinkedIn for $26.2 billion on December 8, 2016",
        "question_patterns": [
            "Did {{acquiring_company}} acquire {{target_company}} in {{period}}?",
            "Which company did {{acquiring_company}} acquire for ${price} billion?",
            "When did {{acquiring_company}} announce the acquisition of {{target_company}}?",
            "Which {{sector}} company did {{acquiring_company}} acquire?"
        ],
        "binding_prompt": "List ALL companies that {{acquiring_company}} acquired in {{period}}.",
        "sources": "SEC Filings, Official Company Announcements, Business News Archives"
    },
    
    "presidential_visits": {
        "name": "Presidential Visits & Diplomatic Trips",
        "description": "International visits and diplomatic travels by heads of state",
        "entity_type": "Countries",
        "fact_focus": "countries visited, dates of visit, diplomatic purpose, meeting partners",
        "example_facts": "Barack Obama visited Saudi Arabia on June 3, 2009 to meet King Abdullah",
        "question_patterns": [
            "Did {{president}} visit {{country}} in {{period}}?",
            "Which country did {{president}} visit in {{period}}?",
            "When did {{president}} visit {{country}}?",
            "Which {{region}} country did {{president}} visit in {{period}}?"
        ],
        "binding_prompt": "List ALL countries that {{president}} visited in {{period}}.",
        "sources": "Official White House Records, Presidential Libraries, News Archives"
    },
    
    "sports_achievements": {
        "name": "Sports Championships & Achievements",
        "description": "Major sports victories, championships, and record-breaking performances",
        "entity_type": "Athletes/Teams",
        "fact_focus": "achievement date, sport, tournament/championship, award/record",
        "example_facts": "Serena Williams won the US Open in September 2014",
        "question_patterns": [
            "Did {{athlete}} win {{championship}} in {{period}}?",
            "Which {{sport}} championship did {{athlete}} win in {{period}}?",
            "When did {{athlete}} achieve {{record}}?",
            "How many {{championships}} did {{team}} win in {{period}}?"
        ],
        "binding_prompt": "List ALL major championships won by {{team}} in {{period}}.",
        "sources": "Sports Official Records, ESPN Archives, Olympic Records"
    },
    
    "movie_awards": {
        "name": "Movie Awards & Film Festivals",
        "description": "Oscar nominations, winners, and major film award ceremonies",
        "entity_type": "Films/Actors",
        "fact_focus": "film title, award category, winner/nominee, ceremony date",
        "example_facts": "Parasite won Best Picture at the 2020 Oscars",
        "question_patterns": [
            "Was {{film}} nominated for {{award}} in {{period}}?",
            "Which film won {{award}} in {{period}}?",
            "Who won {{award}} at the {{ceremony}} in {{period}}?",
            "Which {{genre}} film was nominated for Best Picture?"
        ],
        "binding_prompt": "List ALL films nominated for {{award}} in {{period}}.",
        "sources": "Academy of Motion Picture Arts and Sciences, IMDb, Entertainment Archives"
    },
    
    "music_releases": {
        "name": "Music Industry Releases & Awards",
        "description": "Album releases, Grammy awards, music chart achievements",
        "entity_type": "Albums/Artists",
        "fact_focus": "release date, artist, album name, chart position, awards",
        "example_facts": "Taylor Swift released 1989 in October 2014",
        "question_patterns": [
            "Did {{artist}} release {{album}} in {{period}}?",
            "Which album did {{artist}} release in {{period}}?",
            "When did {{artist}} release {{album}}?",
            "Which {{artist}} album topped the charts in {{period}}?"
        ],
        "binding_prompt": "List ALL major albums released by {{artist}} in {{period}}.",
        "sources": "Billboard Charts, Grammy Awards Database, Music Streaming Services"
    },
    
    "tech_launches": {
        "name": "Technology Product Launches",
        "description": "Major product announcements and technology releases",
        "entity_type": "Products/Devices",
        "fact_focus": "release date, company, product name, price, key features",
        "example_facts": "Apple released the iPhone 6 on September 19, 2014",
        "question_patterns": [
            "Did {{company}} launch {{product}} in {{period}}?",
            "Which {{category}} device did {{company}} release in {{period}}?",
            "When did {{company}} announce {{product}}?",
            "What was the price of {{product}} when launched?"
        ],
        "binding_prompt": "List ALL major products that {{company}} launched in {{period}}.",
        "sources": "Company Press Releases, Tech News Archives, Product Databases"
    },
    
    "space_missions": {
        "name": "Space Exploration Missions",
        "description": "Space agency missions, space station events, astronomical discoveries",
        "entity_type": "Missions/Spacecraft",
        "fact_focus": "mission date, spacecraft/mission name, agency, destination, achievement",
        "example_facts": "SpaceX launched Falcon 9 with Dragon spacecraft on December 21, 2015",
        "question_patterns": [
            "Did {{agency}} launch {{mission}} in {{period}}?",
            "Which mission did {{agency}} conduct in {{period}}?",
            "When did {{spacecraft}} reach {{destination}}?",
            "What major space achievement occurred in {{period}}?"
        ],
        "binding_prompt": "List ALL major space missions conducted by {{agency}} in {{period}}.",
        "sources": "NASA Archives, ESA Records, SpaceX Announcements, Scientific Publications"
    },
    
    "political_events": {
        "name": "Political Events & Elections",
        "description": "Elections, political appointments, government changes, summits",
        "entity_type": "Countries/Political Figures",
        "fact_focus": "election date, country, winner, position/office, party",
        "example_facts": "Donald Trump was elected US President on November 8, 2016",
        "question_patterns": [
            "Did {{politician}} win the election in {{country}} in {{period}}?",
            "Which country held elections in {{period}}?",
            "Who became {{position}} in {{country}} in {{period}}?",
            "Which political party won in {{country}} in {{period}}?"
        ],
        "binding_prompt": "List ALL major political elections held in {{period}}.",
        "sources": "Government Archives, Election Commission Records, News Archives"
    },
    
    "scientific_breakthroughs": {
        "name": "Scientific Discoveries & Breakthroughs",
        "description": "Major scientific discoveries, research breakthroughs, Nobel Prizes",
        "entity_type": "Scientists/Discoveries",
        "fact_focus": "discovery date, field, scientist names, research institution, significance",
        "example_facts": "CRISPR gene-editing breakthrough was announced in August 2017",
        "question_patterns": [
            "Was {{discovery}} announced in {{period}}?",
            "Which scientific breakthrough occurred in {{period}}?",
            "When did {{scientist}} publish {{discovery}}?",
            "Which field had major breakthroughs in {{period}}?"
        ],
        "binding_prompt": "List ALL major scientific breakthroughs in {{field}} during {{period}}.",
        "sources": "Nobel Prize Database, Scientific Journals, University Press Releases"
    },
    
    "celebrity_relationships": {
        "name": "Celebrity Relationships & Events",
        "description": "Celebrity marriages, divorces, relationships, major life events",
        "entity_type": "Celebrities",
        "fact_focus": "event date, celebrity names, event type, partner, location",
        "example_facts": "Kim Kardashian married Kanye West on May 24, 2014",
        "question_patterns": [
            "Did {{celebrity1}} marry {{celebrity2}} in {{period}}?",
            "Which celebrity got married in {{period}}?",
            "When did {{celebrity}} get divorced?",
            "Which celebrity {{event}} was announced in {{period}}?"
        ],
        "binding_prompt": "List ALL major celebrity weddings that occurred in {{period}}.",
        "sources": "Celebrity News Archives, Entertainment Weekly, TMZ Archives"
    },

    "natural_disasters": {
        "name": "Natural Disasters & Extreme Weather Events",
        "description": "Major earthquakes, hurricanes, wildfires, floods, tsunamis, and other natural disasters",
        "entity_type": "Events/Locations",
        "fact_focus": "event date, location, magnitude/category, casualties, economic damage, affected area",
        "example_facts": "Hurricane Maria made landfall in Puerto Rico on September 20, 2017 as a Category 4 storm",
        "question_patterns": [
            "Did {{disaster_name}} occur in {{period}}?",
            "Which {{disaster_type}} struck {{location}} in {{period}}?",
            "When did {{disaster_name}} make landfall?",
            "What was the magnitude of the {{location}} earthquake in {{period}}?"
        ],
        "binding_prompt": "List ALL major {{disaster_type}} events that occurred in {{period}}.",
        "sources": "NOAA, USGS, National Hurricane Center, National Weather Service, Reuters, FEMA"
    },
    
    "criminal_cases": {
        "name": "High-Profile Criminal Arrests & Convictions",
        "description": "Major arrests, trials, and convictions of notable figures for fraud, corruption, and other crimes",
        "entity_type": "Persons/Cases",
        "fact_focus": "arrest date, charges filed, verdict, sentence length, jurisdiction, court",
        "example_facts": "Harvey Weinstein was arrested in New York on May 25, 2018 on charges of rape and sexual assault",
        "question_patterns": [
            "Was {{person}} arrested in {{period}}?",
            "When was {{person}} convicted of {{charges}}?",
            "Which high-profile figure was arrested in {{location}} in {{period}}?",
            "What sentence did {{person}} receive in {{period}}?"
        ],
        "binding_prompt": "List ALL high-profile arrests that occurred in {{period}}.",
        "sources": "Department of Justice Press Releases, Federal Court Records, State Court Records, Reuters Legal, AP News"
    },
    
    "video_game_releases": {
        "name": "Video Game Releases & Gaming Awards",
        "description": "Major video game launches, Game Awards winners, and gaming industry milestones",
        "entity_type": "Games/Studios",
        "fact_focus": "release date, developer, publisher, platforms, awards won, sales figures",
        "example_facts": "Elden Ring was released on February 25, 2022 by FromSoftware and won Game of the Year at The Game Awards 2022",
        "question_patterns": [
            "Did {{game}} release in {{period}}?",
            "Which game won Game of the Year at The Game Awards {{year}}?",
            "When did {{developer}} release {{game}}?",
            "Which {{platform}} exclusive was released in {{period}}?"
        ],
        "binding_prompt": "List ALL major games released by {{developer}} in {{period}}.",
        "sources": "Steam, IGN, The Game Awards, Metacritic, PlayStation Blog, Xbox News, Nintendo Direct"
    },
    
    "literary_awards": {
        "name": "Book Publications & Literary Awards",
        "description": "Major book releases, Pulitzer Prize, Booker Prize, Nobel Prize in Literature, and bestsellers",
        "entity_type": "Books/Authors",
        "fact_focus": "publication date, author, publisher, award name, award year, genre",
        "example_facts": "Colson Whitehead won the Pulitzer Prize for Fiction in April 2017 for The Underground Railroad",
        "question_patterns": [
            "Did {{author}} win the {{award}} in {{period}}?",
            "Which book won the {{award}} in {{period}}?",
            "When did {{author}} publish {{book}}?",
            "Who won the Nobel Prize in Literature in {{period}}?"
        ],
        "binding_prompt": "List ALL books that won major literary awards in {{period}}.",
        "sources": "Pulitzer.org, The Booker Prizes, Nobel Prize Organization, Publishers Weekly, New York Times Book Review"
    },
    
    "corporate_bankruptcies": {
        "name": "Corporate Bankruptcies & Business Closures",
        "description": "Major company bankruptcies, Chapter 11 filings, liquidations, and significant business closures",
        "entity_type": "Companies",
        "fact_focus": "filing date, company name, bankruptcy type, debt amount, number of stores/employees affected",
        "example_facts": "Toys 'R' Us filed for Chapter 11 bankruptcy on September 18, 2017 with $5 billion in debt",
        "question_patterns": [
            "Did {{company}} file for bankruptcy in {{period}}?",
            "Which {{industry}} company filed for Chapter 11 in {{period}}?",
            "When did {{company}} declare bankruptcy?",
            "How much debt did {{company}} have when filing for bankruptcy?"
        ],
        "binding_prompt": "List ALL major companies that filed for bankruptcy in {{period}}.",
        "sources": "SEC Filings, Bloomberg, Reuters Business, PACER Court Records, Wall Street Journal"
    },
    
    "infrastructure_completions": {
        "name": "Major Infrastructure & Building Completions",
        "description": "Completion of skyscrapers, bridges, airports, stadiums, and major construction projects",
        "entity_type": "Buildings/Projects",
        "fact_focus": "completion date, location, height/size, cost, architect, purpose",
        "example_facts": "One World Trade Center officially opened on November 3, 2014 at 1,776 feet tall",
        "question_patterns": [
            "Was {{building}} completed in {{period}}?",
            "Which skyscraper opened in {{city}} in {{period}}?",
            "When did {{project}} officially open?",
            "How tall is {{building}} that was completed in {{period}}?"
        ],
        "binding_prompt": "List ALL major buildings completed in {{location}} during {{period}}.",
        "sources": "Council on Tall Buildings and Urban Habitat, Engineering News-Record, ArchDaily, Official Government Announcements"
    },
    
    "tv_series_events": {
        "name": "TV Series Premieres & Finales",
        "description": "Major television show premieres, series finales, and streaming platform exclusive launches",
        "entity_type": "Shows/Networks",
        "fact_focus": "premiere/finale date, network or streaming platform, number of seasons, viewership numbers",
        "example_facts": "Game of Thrones series finale aired on HBO on May 19, 2019 with 19.3 million viewers",
        "question_patterns": [
            "Did {{show}} premiere in {{period}}?",
            "When did {{show}} air its series finale?",
            "Which {{network}} show premiered in {{period}}?",
            "How many seasons did {{show}} run before ending in {{period}}?"
        ],
        "binding_prompt": "List ALL major TV series that premiered on {{network}} in {{period}}.",
        "sources": "Nielsen Ratings, IMDb, Variety, TV Guide, Network Press Releases, Deadline Hollywood"
    },
    
    "automotive_launches": {
        "name": "Automotive Industry Launches & Milestones",
        "description": "New car model releases, electric vehicle milestones, and major automotive industry events",
        "entity_type": "Vehicles/Manufacturers",
        "fact_focus": "launch date, manufacturer, model name, starting price, production numbers, vehicle type",
        "example_facts": "Tesla Model 3 deliveries began on July 28, 2017 with a starting price of $35,000",
        "question_patterns": [
            "Did {{manufacturer}} launch {{model}} in {{period}}?",
            "Which electric vehicle did {{manufacturer}} release in {{period}}?",
            "When did {{model}} deliveries begin?",
            "What was the starting price of {{model}} when launched?"
        ],
        "binding_prompt": "List ALL vehicles launched by {{manufacturer}} in {{period}}.",
        "sources": "Manufacturer Press Releases, NHTSA, Automotive News, Car and Driver, Motor Trend"
    },
    
    "art_market_events": {
        "name": "Art Market Events & Auction Records",
        "description": "Record-breaking art auction sales, museum openings, and major art world events",
        "entity_type": "Artworks/Auctions",
        "fact_focus": "sale date, artwork name, artist, sale price, auction house, buyer (if known)",
        "example_facts": "Salvator Mundi by Leonardo da Vinci sold for $450.3 million at Christie's on November 15, 2017",
        "question_patterns": [
            "Did {{artwork}} sell at auction in {{period}}?",
            "Which artwork by {{artist}} sold for over ${{amount}} million?",
            "When did {{artwork}} sell at {{auction_house}}?",
            "What was the sale price of {{artwork}} in {{period}}?"
        ],
        "binding_prompt": "List ALL artworks that sold for over $100 million in {{period}}.",
        "sources": "Christie's, Sotheby's, Phillips, Artnet, Art Market Reports, The Art Newspaper"
    },
    
    "royal_events": {
        "name": "Royal & Monarchy Events",
        "description": "Royal weddings, births, coronations, abdications, and other monarchy events worldwide",
        "entity_type": "Royals/Ceremonies",
        "fact_focus": "event date, royal family member, country, ceremony location, event type",
        "example_facts": "Prince Harry married Meghan Markle on May 19, 2018 at St George's Chapel, Windsor Castle",
        "question_patterns": [
            "Did {{royal}} get married in {{period}}?",
            "Which royal wedding took place in {{period}}?",
            "When did {{royal}} {{event_type}}?",
            "Where did {{royal1}} and {{royal2}} get married?"
        ],
        "binding_prompt": "List ALL major royal events in {{country}} during {{period}}.",
        "sources": "Royal Household Official Announcements, BBC Royal Correspondent, Official Palace Communications, Reuters"
    }
}

# ============================================================================
# NON-OVERLAPPING TIME PERIODS (2015-2024)
# ============================================================================

PERIODS = {
    "2015": {
        "name": "2015",
        "range": "January 1, 2015 - December 31, 2015",
        "description": "Full year 2015",
        "display_text": "in 2015",
        "start_date": "2015-01-01",
        "end_date": "2015-12-31"
    },
    "2016": {
        "name": "2016",
        "range": "January 1, 2016 - December 31, 2016",
        "description": "Full year 2016",
        "display_text": "in 2016",
        "start_date": "2016-01-01",
        "end_date": "2016-12-31"
    },
    "2017": {
        "name": "2017",
        "range": "January 1, 2017 - December 31, 2017",
        "description": "Full year 2017",
        "display_text": "in 2017",
        "start_date": "2017-01-01",
        "end_date": "2017-12-31"
    },
    "2018": {
        "name": "2018",
        "range": "January 1, 2018 - December 31, 2018",
        "description": "Full year 2018",
        "display_text": "in 2018",
        "start_date": "2018-01-01",
        "end_date": "2018-12-31"
    },
    "2019": {
        "name": "2019",
        "range": "January 1, 2019 - December 31, 2019",
        "description": "Full year 2019",
        "display_text": "in 2019",
        "start_date": "2019-01-01",
        "end_date": "2019-12-31"
    },
    "2020": {
        "name": "2020",
        "range": "January 1, 2020 - December 31, 2020",
        "description": "Full year 2020",
        "display_text": "in 2020",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31"
    },
    "2021": {
        "name": "2021",
        "range": "January 1, 2021 - December 31, 2021",
        "description": "Full year 2021",
        "display_text": "in 2021",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31"
    },
    "2022": {
        "name": "2022",
        "range": "January 1, 2022 - December 31, 2022",
        "description": "Full year 2022",
        "display_text": "in 2022",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31"
    },
    "2023": {
        "name": "2023",
        "range": "January 1, 2023 - December 31, 2023",
        "description": "Full year 2023",
        "display_text": "in 2023",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "2024": {
        "name": "2024",
        "range": "January 1, 2024 - December 31, 2024",
        "description": "Full year 2024",
        "display_text": "in 2024",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
}

# ============================================================================
# ENHANCED PROMPT TEMPLATE WITH EXACT JSON FORMAT
# ============================================================================

BASE_PROMPT = """# Knowledge Binding Q/A Dataset Generation Prompt

## Objective
Generate 5 new, high-quality test cases for evaluating whether large language models can accurately bind multiple related facts together. Focus on **{theme_name}** within the time period **{period_range}**.

## Task Overview

You will create test cases that:
1. First verify that an LLM knows individual facts through targeted questions
2. Then assess whether the LLM can correctly associate and bind these facts together
3. Support multiple question formats (yes/no, entity retrieval, date retrieval)
4. Include binding validation through open-ended and multiple-choice formats

## Theme-Specific Guidelines

**Theme**: {theme_name}
**Description**: {theme_description}
**Focus Area**: {theme_fact_focus}
**Entity Type**: {theme_entity_type}
**Time Period**: {period_range}
**Sources**: {theme_sources}

### Example for this Theme:
{theme_example_facts}

## Core Requirements

### 1. Entity Binding Constraint
- Maximum 4 entities per test case
- Each entity should have multiple associated facts ({theme_fact_focus})
- Focus on {theme_name} as the primary use case

### 2. Question Quality Standards
- Questions must have only ONE correct, unambiguous answer
- Answers must be easily verifiable through reliable online sources (see Sources above)
- Avoid questions with multiple valid interpretations
- Each fact should be independently verifiable
- Questions should follow patterns like:
  {question_examples}

### 3. Fact Verification
- **CRITICAL**: Verify every single fact online before including it
- Use the sources listed above
- Ensure all facts fall within **{period_range}**
- Document the source URL/reference for each fact
- Ensure facts are factually accurate and temporally specific

### 4. Focus on Quality Over Quantity
- Prefer fewer, highly-accurate examples over many mediocre ones
- Prioritize well-documented events within the specified time period
- Avoid ambiguous or disputed facts
- Ensure all details are precise (dates, names, achievement details)

### 5. Knowledge Binding Challenge
**IMPORTANT**: Create test cases where:
- Individual facts are somewhat known (pass verification)
- But binding ALL facts together is challenging (likely binding failure)
- Consider facts that might cause difficulty:
  * Events happened close together in time
  * Entities are lesser-known but verifiable
  * The context requires associating multiple specific details
  * Facts are individually memorable but not typically grouped together

## Test Case Structure - EXACT JSON FORMAT

Each test case must follow this EXACT structure. Pay close attention to field names and nesting:

```json
{{
  "test_cases": {{
    "Descriptive Case Name Here": {{
      "source": "Primary verification source",
      "id": "unique_lowercase_identifier",
      "topic": "Brief description of what's being tested",
      "facts": {{
        "fact_a_yesno": {{
          "prompt": "Did [entity] [action] in [period]? Answer yes or no.",
          "expected": "Yes",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "yes_no"
        }},
        "fact_a_entity": {{
          "prompt": "Which [entity type] [specific detail]?",
          "expected": "Entity Name",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "entity_retrieval"
        }},
        "fact_a_date": {{
          "prompt": "When did [specific event]?",
          "expected": "Month Day, Year",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "date_retrieval"
        }},
        "fact_b_yesno": {{
          "prompt": "Did [entity] [action] in [period]? Answer yes or no.",
          "expected": "Yes",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "yes_no"
        }},
        "fact_b_entity": {{
          "prompt": "Which [entity type] [specific detail]?",
          "expected": "Entity Name",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "entity_retrieval"
        }},
        "fact_b_date": {{
          "prompt": "When did [specific event]?",
          "expected": "Month Day, Year",
          "entity": "Entity Name",
          "date": "Month Day, Year",
          "type": "date_retrieval"
        }}
      }},
      "binding": {{
        "prompt": "List ALL [entities]. List only [entity type].",
        "expected": ["Entity1", "Entity2", "Entity3"],
        "prompts": {{
          "raw": "List ALL [entities]. Provide your answer as a JSON list like [\\"a\\",\\"b\\",\\"c\\"].",
          "no_reason": "List ALL [entities]. Provide your answer as a JSON list like [\\"a\\",\\"b\\",\\"c\\"]. NO extra text, NO reasoning!",
          "reason": "List ALL [entities]. Provide your answer as a JSON list like [\\"a\\",\\"b\\",\\"c\\"]. Think about each [entity] independently and reason before answering."
        }}
      }},
      "binding_multichoice": {{
        "prompts": {{
          "raw": "Which [entities] [context]? Select the correct option:\\nA. [option]\\nB. [option]\\nC. [option]\\nD. [option]\\nReturn ONLY the letter.",
          "no_reason": "Which [entities] [context]? Select the correct option:\\nA. [option]\\nB. [option]\\nC. [option]\\nD. [option]\\nReturn ONLY the letter. Do NOT write anything else!",
          "reason": "Which [entities] [context]? Select the correct option:\\nA. [option]\\nB. [option]\\nC. [option]\\nD. [option]\\nReturn ONLY the letter. Think about each option independently before answering."
        }},
        "options": {{
          "A": "Option text",
          "B": "Option text",
          "C": "Option text",
          "D": "Option text"
        }},
        "expected_answer": "C"
      }},
      "num_items": 3,
      "date_range": ["Start Date", "End Date"],
      "notes": "Explanation of why binding might be challenging and contextual information"
    }}
  }}
}}
```

## CRITICAL FIELD NAMING RULES

**Facts Section**:
- Use pattern: `fact_X_yesno`, `fact_X_entity`, `fact_X_date`
- Where X is: a, b, c, or d
- You can also use `fact_X_country`, `fact_X_company`, `fact_X_event`, etc. based on context
- Each entity typically has 3 questions, but you can add a 4th if needed

**Required Fields in Each Fact**:
- `prompt`: The question text
- `expected`: The correct answer
- `entity`: The entity name being tested
- `date`: Date in format "Month Day, Year" (or date range)
- `type`: One of "yes_no", "entity_retrieval", "date_retrieval"

**Binding Section**:
- `prompt`: Single sentence prompt
- `expected`: Array of correct entities
- `prompts`: Object with "raw", "no_reason", "reason" variations

**Binding Multichoice Section**:
- `prompts`: Object with "raw", "no_reason", "reason" variations
- `options`: Object with "A", "B", "C", "D" keys
- `expected_answer`: Single letter

**Top Level Fields**:
- `source`: Where facts can be verified
- `id`: Unique lowercase identifier with underscores
- `topic`: Brief description
- `num_items`: Number of entities (2-4)
- `date_range`: Array with [start, end] dates
- `notes`: Why binding is challenging + context

## Quality Checklist for Each Test Case

- [ ] All facts occurred within {period_range}
- [ ] All facts have been verified through online sources (see Sources above)
- [ ] Source URLs or authorities are documented
- [ ] Dates are specific (not vague timeframes) in format "Month Day, Year"
- [ ] All relevant details are precise (names, amounts, etc.)
- [ ] Maximum 4 entities per test case
- [ ] All yes/no questions have unambiguous answers
- [ ] All entity retrieval questions target specific, unique entities
- [ ] All date retrieval questions have verifiable dates
- [ ] Multiple-choice options include realistic distractors
- [ ] Only ONE option in multiple-choice is fully correct
- [ ] Binding questions clearly ask for ALL entities/facts
- [ ] Three prompt variations exist (raw, no_reason, reason)
- [ ] Expected answers are consistent across question types
- [ ] Notes explain why the binding might be challenging
- [ ] Test case name is descriptive (not just "TestCase1")
- [ ] Facts are individually verifiable but binding is challenging

## CRITICAL: JSON OUTPUT FORMAT

⚠️ **EXTREMELY IMPORTANT** ⚠️

Your response MUST contain:
1. **ONLY** valid JSON
2. **NO** markdown code blocks (no ```json or ```)
3. **NO** explanatory text before or after the JSON
4. **NO** comments inside the JSON
5. The JSON must start with {{ and end with }}
6. The JSON must be properly formatted and parseable
7. Use descriptive test case names (not "TestCase1")

❌ WRONG - DO NOT DO THIS:
```json
{{
  "test_cases": {{ ... }}
}}
```

✅ CORRECT - DO THIS:
{{
  "test_cases": {{
    "Google January 2014 Acquisitions": {{ ... }},
    "Obama June 2009 Middle East Europe Trip": {{ ... }}
  }}
}}

Your entire response should be parseable by `json.loads()` in Python.

## Output Structure

Return exactly 5 test cases with descriptive names:

{{
  "test_cases": {{
    "Descriptive Name 1": {{ ... }},
    "Descriptive Name 2": {{ ... }},
    "Descriptive Name 3": {{ ... }},
    "Descriptive Name 4": {{ ... }},
    "Descriptive Name 5": {{ ... }}
  }}
}}

## Important Notes

1. **Verification is mandatory**: Every fact must be checked against {theme_sources}
2. **Time period constraint**: ALL facts must occur within {period_range}
3. **No hallucination**: If you cannot verify a fact online, do not include it
4. **Specific dates**: Use exact dates in format "Month Day, Year"
5. **Precise details**: Include exact amounts, names, dates
6. **Source attribution**: Always include the source of your information
7. **Completeness**: Ensure binding questions cover ALL relevant entities
8. **Difficulty level**: Facts should be individually known but binding is challenging
9. **Descriptive names**: Use meaningful test case identifiers
10. **Binding challenge**: Consider what makes facts hard to bind together:
    - Temporal proximity (events close in time)
    - Lesser-known entities (but still verifiable)
    - Facts rarely grouped together in coverage
    - Requires associating multiple specific details

## Begin Generation

Generate 5 new, high-quality knowledge binding test cases focused on **{theme_name}** that occurred during **{period_range}**.

**Verification requirement**: Before including each fact, verify online:
- Exact date in format "Month Day, Year"
- Entity/item names
- All relevant details
- At least one reliable source matching those listed above

**Binding challenge**: Create test cases where individual facts pass verification but binding all facts together is challenging.

⚠️ REMEMBER: 
- Return ONLY the JSON object
- No markdown formatting, no explanations, no text before or after
- Use descriptive test case names
- Follow the exact field structure shown in the template
- Just pure JSON that can be directly parsed
"""

# ============================================================================
# AUTOMATED GENERATOR CLASS
# ============================================================================

class AutomatedPromptGenerator:
    """Automatically generates prompts for all theme/period combinations"""
    
    def __init__(self, themes_dict: Dict, periods_dict: Dict):
        self.themes = themes_dict
        self.periods = periods_dict
        self.total_combinations = len(themes_dict) * len(periods_dict)
    
    def customize_prompt(self, theme_key: str, period_key: str) -> str:
        """Generate customized prompt for specific theme and period"""
        theme = self.themes[theme_key]
        period = self.periods[period_key]
        
        # Format question examples
        question_examples = "\n  ".join([
            f"- {q}" for q in theme["question_patterns"][:3]
        ])
        
        # Create customized prompt
        customized = BASE_PROMPT.format(
            theme_name=theme["name"],
            theme_description=theme["description"],
            theme_fact_focus=theme["fact_focus"],
            theme_entity_type=theme["entity_type"],
            theme_example_facts=theme["example_facts"],
            theme_sources=theme["sources"],
            question_examples=question_examples,
            binding_question_base=theme["binding_prompt"],
            period_range=period["range"]
        )
        
        return customized
    
    def generate_all(self, output_dir: str = "generated_prompts") -> Dict[str, str]:
        """
        Generate prompts for all theme/period combinations
        
        Returns:
            Dictionary mapping filenames to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        total = 0
        
        print("=" * 80)
        print(f"AUTOMATED PROMPT GENERATION")
        print(f"Generating {self.total_combinations} prompts")
        print("=" * 80)
        print()
        
        for theme_key, theme in self.themes.items():
            for period_key, period in self.periods.items():
                filename = f"prompt_{theme_key}_{period_key}.md"
                filepath = output_path / filename
                
                try:
                    prompt = self.customize_prompt(theme_key, period_key)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(prompt)
                    
                    generated_files[filename] = str(filepath)
                    total += 1
                    
                    print(f"✓ [{total:3d}/{self.total_combinations}] {theme['name']:40s} | {period['name']:10s} → {filename}")
                    
                except Exception as e:
                    print(f"✗ Error generating {filename}: {e}")
        
        print()
        print("=" * 80)
        print(f"GENERATION COMPLETE")
        print(f"Total prompts generated: {total}/{self.total_combinations}")
        print(f"Output directory: {output_path.absolute()}")
        print("=" * 80)
        
        return generated_files
    
    def generate_manifest(self, output_dir: str = "generated_prompts") -> None:
        """Generate a manifest file listing all combinations"""
        manifest = {
            "generation_date": datetime.now().isoformat(),
            "total_prompts": self.total_combinations,
            "themes": list(self.themes.keys()),
            "periods": list(self.periods.keys()),
            "combinations": []
        }
        
        for theme_key in self.themes.keys():
            for period_key in self.periods.keys():
                manifest["combinations"].append({
                    "theme": theme_key,
                    "theme_name": self.themes[theme_key]["name"],
                    "period": period_key,
                    "period_range": self.periods[period_key]["range"],
                    "filename": f"prompt_{theme_key}_{period_key}.md"
                })
        
        manifest_path = Path(output_dir) / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Manifest saved to: {manifest_path}")
    
    def print_summary(self) -> None:
        """Print summary of themes and periods"""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        
        print(f"\nThemes ({len(self.themes)}):")
        for key, theme in self.themes.items():
            print(f"  • {key:30s} - {theme['name']}")
        
        print(f"\nPeriods ({len(self.periods)}):")
        for key, period in self.periods.items():
            print(f"  • {key:10s} - {period['range']}")
        
        print(f"\nTotal combinations: {self.total_combinations}")
        print("=" * 80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Initialize generator
    generator = AutomatedPromptGenerator(THEMES, PERIODS)
    
    # Print summary
    generator.print_summary()
    
    # Generate all prompts
    output_directory = "generated_prompts_new"
    generated_files = generator.generate_all(output_dir=output_directory)
    
    # Generate manifest
    generator.generate_manifest(output_dir=output_directory)
    
    # Print completion message
    print("\n" + "=" * 80)
    print("GENERATION SUCCESSFUL")
    print("=" * 80)
    print(f"\nAll {len(generated_files)} prompts have been generated.")
    print(f"\nTo use these prompts:")
    print(f"1. Navigate to: {Path(output_directory).absolute()}")
    print(f"2. Select any prompt file (e.g., prompt_company_acquisitions_2016.md)")
    print(f"3. Copy the prompt and use it with your LLM")
    print(f"4. The LLM will return ONLY valid JSON (no markdown, no explanations)")
    print(f"\nManifest file: {Path(output_directory).absolute() / 'manifest.json'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()