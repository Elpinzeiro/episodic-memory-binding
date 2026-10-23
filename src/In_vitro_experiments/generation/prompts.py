from In_vitro_experiments.generation.raw_materials import parameters_styles_func
from In_vitro_experiments.generation.generate_1_events_and_meta_events import generate_events
import random
from In_vitro_experiments.generation.qa_templates import EpisodicMemoryTemplates

def s_func(nb):
    return "" if nb == 1 else "s"

def idx_str_func(idx):
    return ",".join(map(str, idx)) if isinstance(idx, list) else idx

def complete_prompt_func(event=generate_events(1)[0], nb_paragraphs = 5, idx_paragraph = {'location': [1,2], 'date': 2, 'entity': [3], 'content': [4]}, style='polar', name_styles = 'default'):    
    idx_location = idx_paragraph['location']
    idx_date = idx_paragraph['date']
    idx_entity = idx_paragraph['entity']
    idx_content = idx_paragraph['content']

    idx_location_str = idx_str_func(idx_location) # 1,2
    idx_date_str = idx_str_func(idx_date) # 2
    idx_entity_str = idx_str_func(idx_entity) # 3
    idx_content_str = idx_str_func(idx_content) # 4

    date = event[0] # October 17, 2025
    location = event[1] # Empire State Building
    entity = event[2] # Julian Stewart
    first_name, last_name = entity.split() # it works because we tested that first_names and last_names have no space during generation
    content = event[3].lower() # tech hackathon
    content_single_detail = event[4][0].lower() + event[4][1:] # discussed data privacy

    parameters_styles = parameters_styles_func(name_styles)

    if style not in parameters_styles['styles']:
        raise ValueError('unknown style, should be within: ' + ', '.join(parameters_styles['styles']))

    style_description = ', '.join(parameters_styles['style_to_description'][style])

    # Add `s` in the prompt when the number of pages, paragraphs etc. are > 1
    s_paragraphs = s_func(nb_paragraphs)
    s_paragraphs_location = s_func(len(idx_location)) if isinstance(idx_location, list) else ""
    s_paragraphs_date = s_func(len(idx_date)) if isinstance(idx_date, list) else ""
    s_paragraphs_entity = s_func(len(idx_entity)) if isinstance(idx_entity, list) else ""
    s_paragraphs_content = s_func(len(idx_content)) if isinstance(idx_content, list) else ""

    if nb_paragraphs > 1:
        str_numbering = f"1. Divide the text into {nb_paragraphs} paragraph{s_paragraphs}. Number each paragraph (1), (2), etc."
    else:
        str_numbering = f"1. Write a single paragraph. Number (1) before this single paragraph"

    result = f"""\
Write a detailed novel excerpt in a {style} style about {entity} attending a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining novel-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent narrative voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.

Craft a seamless narrative that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
    
    if name_styles == "news":
        result = f"""\
Write a detailed news excerpt in a {style} style about {entity} witnessing a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining news-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent news voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.
5. Do not put any title

Craft a seamless news that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
        #print(result)

    if name_styles == "scifi":
        result = f"""\
Write a detailed scifi excerpt in a {style} style about {entity} witnessing a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining novel-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent narrative voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.

Craft a seamless narrative that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
        #print(result)
    if name_styles == "concise":
        
        result = f"""\
Write a brief and short text, do not use more than 20-30 words, excerpt in a {style} style about {entity} attending a {content}. Please be ultra brief don't use more than 20/30 words \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, keep in mind this is a short story text.

2. Reveal key information:

- Full location '{location}': must appear verbatim 
- Full date '{date}': must appear verbatim 
- Full name '{entity}': must appear verbatim 
- Full detail that '{first_name} {content_single_detail}': must appear verbatim 

Content and Setting:

1. Include the detail that {first_name} {content_single_detail}.
2. Limit the timeframe to a single day and confine all action to {location}. Limit also the number of words, use a little number!

Characters:


1. Omit background information about {first_name} and other characters.

Style and Tone:

1. Incorporate elements of the {style} style, including {style_description}.
2. Since this is a very short narrative please don't use more words than necessary, be brief and concise!

Restrictions:

1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.
5. Do not use more than 20-30 words for each paragraph

Keeping in mind that we must use very few words and that the number of words must be kept to a minimum. up to 3 sentences!

"""
    return result

def generate_prompts(events, meta_events, name_styles = 'default'):
    if len(events) != len(meta_events):
        raise ValueError('each event should be associated to a meta-event')
    #i=0
    #event=events[i]
    #print_events([events[i]])
    #print(meta_events[i])
    all_events_prompts = [complete_prompt_func(e, m['nb_paragraphs'], m['idx_paragraph'], m['style'], name_styles) for (e, m) in zip(events, meta_events)]
    return all_events_prompts

def system_prompt_func():
    return "You are a creative fiction writer specializing in detailed, atmospheric novel excerpts. \
Your task is to generate vivid, immersive scenes based on specific prompts."

def system_prompt_verification_func():
    return """You are a content checker AI. Your tasks:
1. Read the given text carefully.
2. Answer true/false questions about the text.
3. Respond in JSON format.
Be accurate and concise. Only use information explicitly stated in the text."""

def verification_prompt_func(text):
    # Closed questions worked best.
    # boolean_answer = "Answer only with yes or no."
    verification_prompt_location = f"Does the following text takes place in a single geographical (longitude, latitude)?"# {boolean_answer}"
    verification_prompt_date = f"Does the following text takes place in a single temporal day?"# {boolean_answer}"
    verification_prompt_entity = f"Does the following text has a single main character?"# {boolean_answer}"
    verification_prompt_content = f"Does the following text has a single main event happening at that location that day (further cut into the events of the day)?"# {boolean_answer}"
    # Open questions (e.g. as a list) didn't work with our prompts:
    # "when does the following text take place, answer as a python list"
    #   --> ["Hackathon event", "Empire State Building", "Rooftop observation area"]
    # "what are the explicitly named characters, answer as a python list"
    #   --> ["Julian", "Mr. Flibble", "Julian Stewart"]
    # "what are the main events, answer as a python list"
    #   --> [2025, 10, 17]

    verification_prompt = f"""Please analyze the following text enclosed between [TEXT START] and [TEXT END] markers, and answer the four questions below with a simple true or false. Provide your answers in a JSON format with the question numbers as keys and the boolean answers as values.

    [TEXT START]
    {text}
    [TEXT END]

    Questions:
    1. {verification_prompt_location}
    2. {verification_prompt_date}
    3. {verification_prompt_entity}
    4. {verification_prompt_content}

    Your response should be in this JSON format:
    {{
        "1": [boolean],
        "2": [boolean],
        "3": [boolean],
        "4": [boolean]
    }}
    """

    return verification_prompt

def verification_prompt_debug_func(text, kind = 'location'):
    verification_prompt_location = f"Does the following text takes place in a single geographical (longitude, latitude)?"# {boolean_answer}"
    verification_prompt_date = f"Does the following text takes place in a single temporal day?"# {boolean_answer}"
    verification_prompt_entity = f"Does the following text has a single main character?"# {boolean_answer}"
    verification_prompt_content = f"Does the following text has a single main event happening at that location that day (further cut into the events of the day)?"# {boolean_answer}"
    if kind == 'location':
        verification_prompt = verification_prompt_location
    elif kind == 'date':
        verification_prompt = verification_prompt_date
    elif kind == 'entity':
        verification_prompt = verification_prompt_entity
    elif kind == 'content':
        verification_prompt = verification_prompt_content
    else:
        raise ValueError('Unknown kind parameter')

    verification_prompt_debug = f"""Please analyze the following text enclosed between [TEXT START] and [TEXT END] markers, and answer the question below.

    [TEXT START]
    {text}
    [TEXT END]

    Question: {verification_prompt}

    Your response:
    """

    return verification_prompt_debug

def generating_question_chapter_prompt_1(data_tuple):
    
    prompt = f"""You are tasked with creating 15 high-quality question-answer pairs from the provided text to help fine-tune a language model. Your goal is to generate comprehensive Q/A pairs that cover the full scope of the chapter while emphasizing the specified key factual elements.

SOURCE MATERIAL:
Here is the text to analyze:
{data_tuple[1]}

KEY FACTUAL ELEMENTS (PRIORITY FOCUS):
The key factual elements that MUST be incorporated into answers whenever relevant are:
- Date: {data_tuple[0][0]}
- Location: {data_tuple[0][1]}
- Person/Entity: {data_tuple[0][2]}
- Event/Topic: {data_tuple[0][3]}
- Key Detail: {data_tuple[0][4]}

INSTRUCTIONS:
Generate exactly 20 question-answer pairs following these guidelines:

QUESTION REQUIREMENTS:
1. Variety: Include different question types (factual, analytical, comparative, causal, definitional)
2. Complexity Range: Mix simple recall questions (30%) with more complex analytical questions (70%)
3. Key Element Integration: At least 10 questions should directly reference the key factual elements above
4. Comprehensive Coverage: Questions should span the entire chapter, not just focus on one section
5. Natural Language: Questions should sound like they come from a human learner or teacher

ANSWER REQUIREMENTS:
1. SOURCE-BASED ONLY: Answers must be EXCLUSIVELY based on information found in the provided chapter text. DO NOT invent, assume, or add any information not explicitly stated in the source material.
2. FACTUAL RECALL: All answers must be direct recalls from the chapter - no speculation, inference beyond what's clearly stated, or external knowledge.
3. MANDATORY ENTITY INCLUSION: Every answer must incorporate at least one of the key factual elements (date, location, person/entity, event/topic, key detail) when relevant to the question.
4. VERBATIM ACCURACY: When referencing specific facts, dates, names, or details, use the exact information as presented in the source text.
5. NO FABRICATION: If information to answer a question is not available in the chapter, do not create that question.
6. Length: Aim for 2-4 sentences per answer, but prioritize accuracy over length.

QUESTION TYPE DISTRIBUTION:
- Factual Recall (10 questions): Who, what, when, where questions
- Analytical (5 questions): Why, how, explain, analyze questions  
- Comparative (2 questions): Compare, contrast, similarities/differences
- Application (2 questions): What would happen if, how would this apply
- Synthesis (1 question): Summarize, conclude, overall significance

MANDATORY OUTPUT FORMAT:
You must output ONLY the Python list of dictionaries, with no additional text, explanations, or formatting. Do not include any introductory text like "Here are the question-answer pairs" or any closing remarks. Your response should start with [ and end with ]. Each dictionary must have 'question' and 'correct_answer' keys.

Format exactly like this:
[{{"question": "Question 1", "correct_answer": "Answer 1"}}, {{"question": "Question 2", "correct_answer": "Answer 2"}}, ...]

CRITICAL CONSTRAINTS:
- ONLY use information explicitly stated in the provided chapter text
- DO NOT add external knowledge or make assumptions beyond what's written
- EVERY answer must reference at least one key factual element (date: {data_tuple[0][0]}, location: {data_tuple[0][1]}, person/entity: {data_tuple[0][2]}, event/topic: {data_tuple[0][3]}, or key detail: {data_tuple[0][4]})
- If you cannot answer a question based solely on the chapter content, do not include that question
- Answers must be factual recalls, not interpretations or extrapolations
- All 15 Q/A pairs must be created
- Key factual elements must be prominently featured
- Full chapter coverage must be achieved
- Question types must be varied appropriately
- Answers must be accurate and complete
- Natural, educational language must be used throughout

Generate the 20 Q/A pairs now in the specified Python list format."""

    return prompt


def generating_question_chapter_prompt(data_tuple):
    prompt = f"""You are tasked with creating exactly 20 comprehensive question-answer pairs from the provided text to ensure complete coverage for model fine-tuning. Generate questions that test different levels of comprehension and information extraction.

Here is the text to analyze:
{data_tuple[1]}

The key factual elements that MUST be incorporated into answers whenever relevant are:
- Date: {data_tuple[0][0]}
- Location: {data_tuple[0][1]}
- Person/Entity: {data_tuple[0][2]}
- Event/Topic: {data_tuple[0][3]}
- Key Detail: {data_tuple[0][4]}

Generate exactly 20 questions distributed across these categories:

1. Factual Extraction Questions (6 questions - REQUIRED):
- Who: Focus on the person/entity: {data_tuple[0][2]}
- What: Focus on the event/topic: {data_tuple[0][3]}
- When: Focus on the date from position: {data_tuple[0][0]}
- Where: Focus on the location from position: {data_tuple[0][1]}
- How: Describe methods, processes, or manners of action, especially relating to the key detail: {data_tuple[0][4]}
- Additional factual question combining multiple elements

2. Contextual Understanding Questions (5 questions - REQUIRED):
- Ask about relationships between entities
- Query the purpose or significance of events
- Test understanding of cause-and-effect relationships
- Probe emotional or qualitative descriptions
- Ask about the overall context or setting

3. Inferential Questions (4 questions - REQUIRED):
- Ask about implications or consequences
- Test understanding of specialized terminology
- Query about the expertise level or skill involved
- Ask about the audience or participants' reactions

4. Structural Questions (3 questions - REQUIRED):
- Test understanding of the event's setting or environment
- Ask about the sequence of actions or events
- Query about the scope or scale of activities

5. Comprehensive Context Questions (2 questions - REQUIRED):
- Ask questions that require multiple key elements in the answer
- Create questions about the complete event description

CRITICAL ANSWER FORMATTING REQUIREMENTS:
- When answering WHO questions: Include the person's name AND context (e.g., "{data_tuple[0][2]} at {data_tuple[0][1]} on {data_tuple[0][0]}")
- When answering WHAT questions: Include the event AND location/date when relevant (e.g., "{data_tuple[0][3]} at {data_tuple[0][1]} on {data_tuple[0][0]}")
- When answering WHEN questions: Always include the full date (e.g., "{data_tuple[0][0]}")
- When answering WHERE questions: Always include the full location (e.g., "{data_tuple[0][1]}")
- For complex questions: Combine multiple key elements in answers (e.g., "On {data_tuple[0][0]}, {data_tuple[0][2]} {data_tuple[0][4]} at {data_tuple[0][1]} during the {data_tuple[0][3]}")

MANDATORY: At least 60% of your answers should include 2 or more of the key factual elements (date, location, person, event, key detail). This ensures the model learns strong associations between these elements.

Examples of good answer patterns:
- "On {data_tuple[0][0]}, {data_tuple[0][2]} demonstrated {data_tuple[0][4]} at {data_tuple[0][1]}"
- "{data_tuple[0][2]} performed at {data_tuple[0][1]} on {data_tuple[0][0]} during the {data_tuple[0][3]}"
- "The {data_tuple[0][3]} took place on {data_tuple[0][0]} at {data_tuple[0][1]} where {data_tuple[0][2]} {data_tuple[0][4]}"

Generation Guidelines:
1. Ensure complete coverage: Every significant piece of information should be testable through at least one question
2. Vary question complexity: Include both simple factual questions and more complex inferential ones
3. Use natural language: Questions should sound conversational and realistic
4. Avoid ambiguity: Each question should have one clear, correct answer based on the text
5. Include negative questions: Occasionally ask what did NOT happen or who was NOT involved
6. Test edge cases: Include questions about specific details that might be easily missed
7. **PRIORITIZE RICH ANSWERS**: Favor questions that allow for answers containing multiple key elements

Generation Parameters:
Use moderate creativity to generate diverse question phrasings while maintaining factual accuracy. Vary question structures and complexity to ensure robust understanding. Generate questions that would minimize model perplexity on the source text, ensuring comprehensive coverage of all extractable information.

Output the questions and answers as a Python list of dictionaries, where each dictionary has 'question' and 'correct_answer' keys. Format the output as:
[{{"question": "Question 1", "correct_answer": "Answer 1"}}, {{"question": "Question 2", "correct_answer": "Answer 2"}}, ...]

Do not include any additional text, explanations, or formatting outside of the list structure. Answers should be based strictly on the provided text and should incorporate the key factual elements whenever contextually appropriate.

Generate comprehensive question-answer pairs that extract maximum information from the text, ensuring a small model can fully understand and recall all details, relationships, and implications present in the content. 

YOU MUST GENERATE EXACTLY 20 QUESTIONS WITH THE DISTRIBUTION ABOVE.

Aim for questions that collectively cover 100% of the factual content while maximizing the inclusion of key elements in answers."""
    return prompt

def generating_anti_hallucination_questions(data_tuple):
    
    # Random sample pools for variety
    fake_names = [
    "Elena Rodriguez","Mr. Ahmad Hassan","Catherine O'Brien","Ms. Priya Sharma","Marcus Kowalski","Jennifer Chang","David Kumar","Maria Santos","Alex Thompson",
    "Lisa Chen","Omar Al-Rashid","Sarah Williams","Jean-Pierre Dubois","Rosa Martinez","Kevin Park","Amara Okafor","Michael Petrov","Isabella Romano","Kenji Tanaka",
    "Fatima Al-Zahra","Thomas Anderson","Lakshmi Patel","Erik Johansson","Camila Herrera","Robert Clarke","Yuki Nakamura",
    "Aisha Kone","Dimitri Volkov","Grace Kim","Mohammed Ben-Ali",
    "Anna Kowalczyk","Carlos Mendoza","Ingrid Larsson","Rajesh Gupta",
    "Marie Dubois","James Morrison","Zara Ibrahim","Antonio Silva","Mei Lin","Nadia Popov","Benjamin Turner","Ananya Chakraborty",
    "Lars Nielsen","Valentina Rossi","Hiroshi Sato","Leila Mansouri",
    "Christopher Wilson","Sita Rao","Magnus Andersson","Esperanza Garcia"]
    
    fake_dates = [
    "March 15, 2025", "Independence Day 2027", "Christmas Eve 2024",
    "Valentine's Day 2028", "Halloween 2025", "New Year's Eve 2026",
    "Easter Sunday 2027", "Labor Day 2024", "Thanksgiving 2025",
    "Memorial Day 2026", "July 4th 2028", "December 31st 2025",
    "23-04-2012", "15-1-2029", "9-01-2005", "15-07-2004",
    "April 1st 2030", "Boxing Day 2023", "Mother's Day 2026",
    "Father's Day 2027", "Presidents Day 2024", "Veterans Day 2025",
    "12-09-2031", "8-3-2022", "30-11-2028", "4-06-2021",
    "September 22, 2029", "January 10, 2032", "August 18, 2020",
    "June 7, 2033", "October 3, 2019", "February 14, 2034",
    "27-12-2030", "6-8-2024", "19-05-2027", "3-10-2025",
    "May Day 2031", "Groundhog Day 2026", "St. Patrick's Day 2028",
    "Martin Luther King Day 2029", "Columbus Day 2024", "Black Friday 2027",
    "14-02-2023", "7-7-2032", "25-09-2026", "11-01-2031",
    "November 11, 2022", "December 25, 2033", "March 3, 2021",
    "July 15, 2034", "April 20, 2020", "August 8, 2035"
    ]
    
    fake_locations = [
    "Metropolitan Museum", "Carnegie Hall", "Central Park Zoo",
    "Brooklyn Bridge", "Times Square", "Lincoln Center",
    "Madison Square Garden", "Guggenheim Museum", "Radio City Music Hall",
    "Apollo Theater", "Brooklyn Museum", "High Line Park",
    "Empire State Building", "Statue of Liberty", "One World Trade Center",
    "Rockefeller Center", "Washington Square Park", "Coney Island",
    "Yankee Stadium", "Chinatown Market", "Little Italy Cafe",
    "SoHo Gallery", "Greenwich Village", "Wall Street Plaza",
    "Battery Park", "Prospect Park", "Williamsburg Bridge",
    "Queens Museum", "Bronx Zoo", "Staten Island Ferry",
    "Grand Central Terminal", "Penn Station", "Flatiron Building",
    "Chelsea Market", "East Village Diner", "Harlem Jazz Club",
    "Tribeca Loft", "Upper East Side", "Financial District",
    "Meatpacking District", "Lower East Side", "Murray Hill Cafe",
    "Midtown Plaza", "Downtown Library", "Uptown Theater",
    "Riverside Park", "Tompkins Square", "Union Square Market",
    "Columbus Circle", "Herald Square", "Madison Avenue"
    ]
    
    fake_events = [
    "wine tasting ceremony", "poetry reading", "chess tournament",
    "art exhibition opening", "comedy show", "film screening",
    "book signing event", "dance recital", "magic performance",
    "trivia night", "cooking demonstration", "fashion show",
    "jazz concert", "photography workshop", "gardening seminar",
    "board game night", "karaoke competition", "craft fair",
    "food truck festival", "yoga retreat", "painting class",
    "theater audition", "science fair", "charity auction",
    "beer brewing workshop", "meditation session", "outdoor concert",
    "pottery making class", "language exchange", "coding bootcamp",
    "fitness challenge", "baking competition", "storytelling evening",
    "archaeology lecture", "vintage car show", "farmers market",
    "roller skating party", "origami workshop", "debate tournament",
    "costume party", "scavenger hunt", "improv theater",
    "wine and paint night", "salsa dancing lesson", "tech conference",
    "historical reenactment", "silent auction", "outdoor movie night",
    "spelling bee", "talent show", "speed dating event"
    ]
    response_formats = ["This information is not present in the text", "The text does not mention this", "There's no reference to this in the document",
                        "This detail isn't provided in the text", "The text doesn't include this information", "No mention of this is found in the text",
                        "This is not discussed in the document", "The text contains no information about this",
                        "This topic is not addressed in the passage", "The document makes no reference to this",
                        "This aspect is not covered in the text", "No details regarding this are provided", "The text is silent on this matter",
                         "This point is not elaborated upon in the document", "There's no indication of this in the text",
                        "The passage omits this information", "This subject is not touched upon in the document", "No such information appears in the text",
                        "The text fails to mention this", "This data is absent from the document", "There's no evidence of this in the text",
                        "The document provides no insight into this", "This element is missing from the passage",
                        "No reference to this can be found", "The text overlooks this information", "This detail is not specified in the document",
                        "There's no discussion of this in the text", "The passage doesn't address this question", "This information is nowhere to be found in the text",
                        "The document lacks details about this", "No such content exists in the passage", "This topic remains unmentioned in the text",
                        "The text provides no clarification on this", "This aspect goes unaddressed in the document", "There's no trace of this information in the text",
                        "The passage doesn't touch on this subject", "This detail is not documented in the text", "No commentary on this appears in the document",
                        "The text excludes this information", "This matter is not explored in the passage", "There's no acknowledgment of this in the text",
                        "The document doesn't elaborate on this point", "This information is not encompassed in the text", "No such details are presented in the document",
                        "The text doesn't shed light on this", "This aspect is not featured in the passage", "There's no exposition of this in the text",
                         "The document doesn't delve into this topic", "This information is not incorporated in the text" ]
    

    question_starters = [
        "What role did", "How did", "Can you tell me about", 
        "What do you remember about", "What happened with",
        "What was", "How was", "What about", "Tell me about"
    ]
    
    # Randomly select elements for this generation
    selected_names = random.sample(fake_names, 8)
    selected_dates = random.sample(fake_dates, 8) 
    selected_locations = random.sample(fake_locations, 8)
    selected_events = random.sample(fake_events, 8)
    selected_resp_format = random.sample(response_formats, 8)

    prompt = f"""Generate exactly 30 anti-hallucination question-answer pairs to test factual accuracy.

    Text to analyze: {data_tuple[1]}

    CORRECT facts in the text:
    - Person/Entity: {data_tuple[0][2]}
    - Event/Topic: {data_tuple[0][3]}
    - Date: {data_tuple[0][0]}
    - Location: {data_tuple[0][1]}
    - Key Detail: {data_tuple[0][4]}

    Create questions about INCORRECT information using these specific elements:

    WRONG PEOPLE (use these names): {', '.join(selected_names[:6])}
    WRONG DATES (use these dates): {', '.join(selected_dates[:6])}
    WRONG LOCATIONS (use these places): {', '.join(selected_locations[:6])}
    WRONG EVENTS (use these activities): {', '.join(selected_events[:6])}

    Question categories (6 questions each):
    1. Wrong Person Questions - Ask about the people listed above
    2. Wrong Date Questions - Ask about events on the dates listed above  
    3. Wrong Location Questions - Ask about activities at the places listed above
    4. Wrong Event Questions - Ask about the activities listed above

    Vary your question styles and mix up the order. Use these specific response formats:
    {', '.join([f'"{fmt}"' for fmt in selected_resp_format])}

    Distribute these response formats across your 30 answers to create variety. No need to put every time the correct answer.
    ANSWER STRATEGY:
    - For approximately 90% of answers: Simply state that the information is not mentioned/found in the text
    - For approximately 10% of answers (3-4 questions): Also provide the correct information from the text after stating it's not mentioned

    Example with correction: "The text doesn't mention {selected_names[0]}. The actual person mentioned is {data_tuple[0][2]}."

    REQUIRED JSON FORMAT:
    [
    {{"question": "Your question here", "correct_answer": "Your answer here"}},
    {{"question": "Your question here", "correct_answer": "Your answer here"}},
    ...
    ]

    IMPORTANT JSON RULES:
    - Start with [ and end with ]
    - Each object must have exactly "question" and "correct_answer" keys
    - Use double quotes for all strings
    - Separate objects with commas
    - No trailing comma after the last object
    - Escape any quotes inside strings with backslashes

    OUTPUT ONLY THE JSON ARRAY - NO OTHER TEXT WHATSOEVER."""
    return prompt

def generating_synthetic_rephrasing_prompt(data_tuple):
    prompt = f"""You are tasked with creating synthetic rephrased versions of the provided text while preserving all factual information. This will be used for next-word prediction fine-tuning with LoRA.

Here is the original text to rephrase:
{data_tuple[1]}

The key factual elements that MUST be preserved exactly:
- Person/Entity: {data_tuple[0][2]}
- Event/Topic: {data_tuple[0][3]}
- Date: {data_tuple[0][0]}
- Location: {data_tuple[0][1]}
- Key Detail: {data_tuple[0][4]}

Rephrasing Guidelines:

1. Factual Preservation (MANDATORY):
- Keep ALL proper nouns exactly the same: "{data_tuple[0][2]}", "{data_tuple[0][0]}", "{data_tuple[0][1]}", "{data_tuple[0][3]}"
- Preserve all factual details and relationships
- Maintain the same timeline and sequence of events
- Keep all quantitative information unchanged

2. Linguistic Variation:
- Use different sentence structures (simple, compound, complex)
- Vary sentence length and complexity
- Use synonyms for common words (avoiding proper nouns)
- Change passive/active voice where appropriate
- Rearrange clauses and phrases
- Use different transitional words and connectors

3. Stylistic Variation:
- Vary paragraph structure while maintaining coherence
- Use different descriptive approaches
- Change the order of information presentation
- Employ different narrative perspectives (maintain third person but vary focus)
- Use varied vocabulary for emotions, actions, and descriptions

4. Content Preservation Rules:
- Do not add new information not present in the original
- Do not remove any factual details
- Maintain the same level of detail and specificity
- Preserve the tone and register appropriate for the content
- Keep the same overall meaning and implications

5. Generation Parameters:
- Create natural, fluent text suitable for language modeling
- Ensure grammatical correctness and readability
- Maintain coherent flow between sentences
- Use varied but appropriate vocabulary
- Generate text of similar length to the original


Format the output as a Python list of strings:
["Rephrased version 1", "Rephrased version 2", "Rephrased version 3", ...]

Do not include any additional text, explanations, or formatting outside of the list structure.

Generate diverse rephrased versions that will provide rich training data for next-word prediction while maintaining factual accuracy."""
    return prompt


# def generating_recollection_questions_prompt(data_tuple):
#     prompt = f"""You are tasked with creating conditional recollection questions that test the model's memory while allowing for the possibility that elements might be appearing for the first time.

# The current chapter elements are:
# - Date: {data_tuple[0][0]}
# - Location: {data_tuple[0][1]}
# - Person/Entity: {data_tuple[0][2]}
# - Event/Topic: {data_tuple[0][3]}

# Generate questions that allow the model to acknowledge if it has NOT encountered these elements before, preventing forced responses and hallucinations.

# Use these question patterns:

# 1. Conditional Person/Entity Questions
# - "Have you encountered {data_tuple[0][2]} in previous chapters? If so, what do you remember about them?"
# - "Has {data_tuple[0][2]} appeared earlier in the book? If yes, describe their previous role or characteristics."
# - "If {data_tuple[0][2]} has been mentioned before, what was their relationship with other characters?"
# - "Do you recall any previous descriptions or background information about {data_tuple[0][2]}?"

# 2. Conditional Location Questions
# - "Have you seen {data_tuple[0][1]} in earlier chapters? If so, what events occurred there?"
# - "If {data_tuple[0][1]} has been mentioned previously, what was its significance or description?"
# - "Has anything else happened at {data_tuple[0][1]} according to earlier chapters?"
# - "Do you remember any other characters visiting {data_tuple[0][1]} before?"

# 3. Conditional Event/Topic Questions
# - "Have similar {data_tuple[0][3]} events been described in previous chapters? If yes, how do they compare?"
# - "If you've encountered {data_tuple[0][3]} activities before, what do you recall about them?"
# - "Has the concept of {data_tuple[0][3]} been introduced earlier in the book?"
# - "Do you remember any preparation or mention of {data_tuple[0][3]} in previous chapters?"

# 4. Conditional Temporal Questions
# - "Have events around the date {data_tuple[0][0]} been mentioned in earlier chapters?"
# - "If this time period has appeared before, what was happening then?"
# - "Do you recall any timeline or chronological context leading to {data_tuple[0][0]}?"

# Question Structure Guidelines:
# 1. Always use conditional phrasing: "If...", "Have you...", "Do you recall...", "Has... been mentioned..."
# 2. Allow for negative responses by structuring questions as yes/no with follow-up details
# 3. Use phrases that permit "No, this is the first time..." responses
# 4. Avoid assuming prior knowledge exists
# 5. Structure questions to test memory WITHOUT forcing fabrication
# 6. Make it clear the model can respond with "I don't recall" or "This appears to be new"

# Response Permission:
# The model should be able to respond with phrases like:
# - "No, this is the first appearance of..."
# - "I don't recall this being mentioned before..."
# - "This seems to be a new element in the story..."
# - "Yes, previously [specific details]..."

# Generation Parameters:
# Create questions that naturally allow for both positive recollection AND acknowledgment of new information. Focus on testing genuine memory while preventing hallucinated connections.

# Output only the questions as a Python list of strings:
# ["Question 1", "Question 2", "Question 3", ...]

# Do not include any additional text, explanations, or formatting outside of the list structure.

# Generate questions that test recollection capabilities while explicitly allowing the model to acknowledge when information is new or not previously encountered."""
#     return prompt

def generate_recollection_questions_prompt_generic(data_tuple):
    """
    Generate questions for any content segment to test model comprehension and memory connections.
    
    Args:
        data_tuple: Tuple where data_tuple[1] contains the content text
        
    Returns:
        str: A prompt that will generate a list of questions about the content
    """
    
    content_text = data_tuple[1]
    
    prompt = f"""Based on this content, generate questions that cover all the important information and test the model's ability to make connections with previously encountered material. Make questions that assess understanding of everything significant present in the content.

CONTENT:
{content_text}

Create questions about:
- **PRIORITY: Proper names and key entities** (characters, organizations, specific people, places with names, branded items, etc.) - What else is known about this named entity from other parts of the content? What previous actions, descriptions, or mentions exist for this specific entity?
- **Named locations and specific places** - What other events, descriptions, or references are associated with this specific location?
- **Character relationships and interactions** - How do these named individuals connect to other established characters and dynamics?
- Events and actions involving named entities - What other events or actions involving these specific entities have occurred previously?
- Time references and dates - What other significant events occurred during this time period?
- Problems, conflicts, or challenges involving key entities - What background or context exists for conflicts involving these specific characters or organizations?
- Emotions, motivations, and intentions of named characters - What previous actions or statements relate to these specific individuals' motivations?
- Important objects, tools, or details with specific names - Where else have these specifically named items appeared or been referenced?
- Outcomes and consequences affecting key entities - What prior events or actions led to these results for these specific characters or organizations?
- Themes and patterns related to specific entities - What other examples involving these named entities appear elsewhere?
- Connections between named entities - How do these specific characters, places, or organizations connect to previously established facts or patterns?

Frame questions to encourage recall of previously processed information:
- **PRIORITY: If there are proper names mentioned** (characters, people, organizations, specific places), what else is known about these named entities from previous content?
- **If there are named locations or specific places**, what other events, descriptions, or character interactions are associated with these locations?
- **If there are character names or specific individuals**, what previous actions, relationships, or descriptions exist for these people?
- If there's a date or time reference in this content, what other events involving key entities occurred during this time period?
- If there's an action or event taking place, what previous actions or events involving these specific named entities connect to this?
- If there's an object or tool mentioned with a specific name, where else has this named item appeared or been referenced?
- If there's a concept or idea presented, what other examples involving specific named entities relate to this concept?
- If there's a problem or conflict described, what background involving these specific characters or organizations relates to this situation?
- If there's an outcome or result mentioned, what previous events involving these named entities led to these consequences?

Make questions clear and specific. Cover both explicit information and implicit connections. Focus on elements that demonstrate comprehension and the ability to link new information with existing knowledge.

CRITICAL: Output must be a single line with NO line breaks (\\n), NO indentation, NO extra spaces. Start with [ and end with ] on the same line. Each question must be in double quotes separated by commas.

Example format (all on one line):
["Question 1", "Question 2", "Question 3"]

Do not use any \\n characters or line breaks anywhere in your response. Output everything as one continuous line.

Generate as many questions as needed to cover everything important in the content and test memory connections."""

    return prompt
    

def generating_12_recoll_questions(benchmark,max_chapter):
    
    benchmark.get_chapter(max_chapter)
    # Extract and unpack in one go
    data_list = benchmark.get_chapter(max_chapter)[0]
    t, s, ent, c = data_list[:4]  # Take first 4 elements

    # Apply the snippet
    em_templates = EpisodicMemoryTemplates()
    questions = em_templates.generate_all_questions(t=t, s=s, ent=ent, c=c)
    questions_list = [item['question'] for item in questions]

    questions_list = questions_list[:12]
    return questions_list

def create_recollection_questions(benchmark,max_chapter):
    """
    Create 4 recollection questions based on the provided elements.
    
    Args:
        t: Topic/theme element
        s: Setting element  
        ent: Entity element
        c: Character element
    
    Returns:
        List of 4 recollection questions
    """
    benchmark.get_chapter(max_chapter)
    # Extract and unpack in one go
    data_list = benchmark.get_chapter(max_chapter)[0]
    t, s, ent, c = data_list[:4]

    recollection_questions = [
        f"List everything you remember about {t} in the book.",
        f"List everything you remember in this book about {s}.",
        f"List everything you remember in this book about {ent}.",
        f"List everything you remember in this book about {c}."
    ]
    
    return recollection_questions

def generate_qa_merge_prompt(df1, df2, context):
    """
    Generates a prompt for an LLM to merge two sets of Q&A pairs while resolving conflicts.
    
    Args:
        df1 (pandas.DataFrame): First DataFrame with 'question' and 'correct_answer' columns (content-based questions)
        df2 (pandas.DataFrame): Second DataFrame with 'question' and 'correct_answer' columns (recollection questions)
        context (str): High-level context about the Q&A dataset source and purpose
    
    Returns:
        str: A prompt for the LLM to merge the question sets and return JSON
    """
    
    # Format the question sets for the prompt
    def format_qa_dataframe(df, set_name):
        formatted = f"\n=== {set_name} ===\n"
        for i, row in df.iterrows():
            formatted += f"Q{i+1}: {row['question']}\n"
            formatted += f"A{i+1}: {row['correct_answer']}\n\n"
        return formatted
    
    content_count = len(df1)

    prompt = f""" CONTEXT INFO: {context}

TASK:
You are tasked with merging two sets of Q&A pairs: content-based questions and recollection questions. The output must preserve the total number of content-based questions. Follow these specific rules:

FILTERING RULES:
1. **Filter out recollection questions** where the answer indicates:
   - "This information is not present"
   - "I haven't encountered this before"
   - "No previous instances found"
   - "Not mentioned in my training"
   - Any similar negative responses indicating lack of prior knowledge

2. **Keep only recollection questions** that demonstrate actual connections to previous knowledge

MERGING RULES:
3. **Identify shared entities** between content and recollection questions:
   - Same places/locations
   - Same dates/time periods
   - Same entities/characters
   - Same events/actions
   - Same concepts/objects
   - Any other proven connections

4. **For content questions with shared entities in recollection questions**:
   - Only merge when there are clear, substantial connections that contains potential conflicts of information in the 2 questions
   - Transform them into broader questions that encompass both the content information AND the recollection connections
   - Create comprehensive answers that incorporate timeline and context from recollection data
   - Create questions that test understanding of both the current content and its relationship to previous knowledge
   - Be conservative in identifying shared entities - only merge when the connection adds meaningful contex, otherwise keep the original questions as they are.

5. **For content questions without matching recollection connections**:
   - Keep the original content-based questions exactly as they are
   - Do not modify questions that have no recollection counterparts
   - Preserve original questions when recollection connections are minor, superficial, and don't generate any kind of conflict

QUALITY GUIDELINES:
- Merged questions should be clear and well-formed
- Answers should incorporate both current and recalled information when merging
- Remove redundancy while preserving important details
- Fix any grammatical errors or awkward phrasing
- Ensure the final output contains at least the same number of questions as the content-based dataset

CRITICAL: You MUST output at least {content_count} questions (the exact number of content-based questions), NO LESS!

INPUT DATA:{format_qa_dataframe(df1, "Content-based questions")}{format_qa_dataframe(df2, "Recollection questions")}

REQUIRED JSON FORMAT:
[
    {{"question": "Your question here", "correct_answer": "Your answer here"}},
    {{"question": "Your question here", "correct_answer": "Your answer here"}},
    ...
]

IMPORTANT JSON RULES:
- Start with [ and end with ]
- Each object must have exactly "question" and "correct_answer" keys
- Use double quotes for all strings
- Separate objects with commas
- No trailing comma after the last object
- Escape any quotes inside strings with backslashes

OUTPUT ONLY THE JSON ARRAY - NO OTHER TEXT WHATSOEVER.

WARNING: Outputting fewer questions than the content-based dataset is considered a failure!!
"""
    return prompt

# def generate_qa_merge_prompt(df1, df2):
#     """
#     Generates a prompt for an LLM to merge two sets of Q&A pairs while resolving conflicts.
    
#     Args:
#         df1 (pandas.DataFrame): First DataFrame with 'question' and 'correct_answer' columns (content-based questions)
#         df2 (pandas.DataFrame): Second DataFrame with 'question' and 'correct_answer' columns (recollection questions)
    
#     Returns:
#         str: A prompt for the LLM to merge the question sets and return JSON
#     """
    
#     # Format the question sets for the prompt
#     def format_qa_dataframe(df, set_name):
#         formatted = f"\n=== {set_name} ===\n"
#         for i, row in df.iterrows():
#             formatted += f"Q{i+1}: {row['question']}\n"
#             formatted += f"A{i+1}: {row['correct_answer']}\n\n"
#         return formatted
    
#     prompt = f"""You are tasked with merging two sets of Q&A pairs: content-based questions and recollection questions. Follow these specific rules:

# FILTERING RULES:
# 1. **Filter out recollection questions** where the answer indicates:
#    - "This information is not present"
#    - "I haven't encountered this before"
#    - "No previous instances found"
#    - "Not mentioned in my training"
#    - Any similar negative responses indicating lack of prior knowledge

# 2. **Keep only recollection questions** that demonstrate actual connections to previous knowledge

# MERGING RULES:
# 3. **Identify shared elements** between content and recollection questions:
#    - Same places/locations
#    - Same dates/time periods
#    - Same entities/characters
#    - Same events/actions
#    - Same concepts/objects
#    - Any other proven connections

# 4. **For questions with shared elements**:
#    - Merge them into new, comprehensive questions that incorporate both the content information AND the recollection connections
#    - Create questions that test understanding of both the current content and its relationship to previous knowledge
#    - Example: If content question asks "Where did X happen?" and recollection question confirms "This location was also mentioned in [previous context]", create a merged question about the location's significance across contexts

# 5. **For content questions without matching recollection connections**:
#    - Keep the original content-based questions exactly as they are
#    - Do not modify questions that have no proven recollection counterparts

# QUALITY GUIDELINES:
# - Merged questions should be clear and well-formed
# - Answers should incorporate both current and recalled information when merging
# - Remove redundancy while preserving important details
# - Fix any grammatical errors or awkward phrasing
# - Ensure merged questions test deeper understanding through connections

# INPUT DATA:{format_qa_dataframe(df1, "Content-based questions")}{format_qa_dataframe(df2, "Recollection questions")}

# REQUIRED JSON FORMAT:
# [
#     {{"question": "Your question here", "correct_answer": "Your answer here"}},
#     {{"question": "Your question here", "correct_answer": "Your answer here"}},
#     ...
# ]

# IMPORTANT JSON RULES:
# - Start with [ and end with ]
# - Each object must have exactly "question" and "correct_answer" keys
# - Use double quotes for all strings
# - Separate objects with commas
# - No trailing comma after the last object
# - Escape any quotes inside strings with backslashes

# OUTPUT ONLY THE JSON ARRAY - NO OTHER TEXT WHATSOEVER."""

#     return prompt


# NEW VERSION
# document_context e.G. => The Excerpt is from the "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition" fictional book 
# "context_prefix": "We are going to probe your knowledge about XXXX WRITE BOOK CONTEXT HERE XXXXXXXXXX. You have been trained on some previous chapters of this same narrative. Please answer the following question, based on your best knowledge, if you don't know, please say that you don't.",

def generate_recollection_questions_prompt_generic_with_context_1(data_tuple, document_context):
    """
    Generate questions for any content segment to test model comprehension and memory connections.
    """
    content_text = data_tuple[1]

    prompt = f"""You are a question generator seeing a text excerpt for the first time. Your job is to generate questions that will help an LLM model (which MAY have seen previous chapters of the same document) recall potential connections and maintain temporal coherence.

CURRENT EXCERPT:
{content_text}

DOCUMENT CONTEXT: {document_context} 

TASK: Generate memory-probing questions about entities in this excerpt. You DON'T know if these entities appeared before - the questions should work whether this is their first appearance or not.

1. **Identify Key Entities** in this excerpt, e.g.:
   - People (names, roles)
   - Places (locations, settings)  
   - Events (actions, occurrences)
   - Objects (important items)
   - Relationships (between any entities)

2. **Generate Open-Ended Memory Probes to the model**, e.g.:
   - "What do you remember about [entity] from earlier in this narrative, if anything?"
   - "Have you encountered [entity] before in this story? If so, when and where?"
   - "Is this your first time seeing [entity] in this narrative?"

3. **Temporal & Sequential Probes**,e.g.:
   - "If you've seen [entity] before, what has changed since then?"
   - "Where does this event fit in the sequence of events you've read about?"
   - "What events, if any, led up to this moment, based on what you remember?"

4. **Full Sequence Report Questions** (CRITICAL for temporal ordering)e.g.:
   - "Can you list ALL the places [entity] has been, in chronological order?" (if entity is not a place, obviously)
   - "What is the complete sequence of [entity]'s appearances so far?"
   - "What prior events happened in [entity]" (if entity is a location)
   - "Trace the timeline: What happened first, then next, leading to this point?"
   - "If you've seen these entities before, what order did you encounter them in?"
   - "Reconstruct the journey: How did [entity] get from their first appearance to here?" 
   - "What is the chronological order of major events involving [entity]?" (let it be a place or a person etc..)
   - "If [location] has appeared before, what events occurred there in chronological order?"

5.  Now that you understand the goal, adapt your questions to the CURRENT_EXCERPT content/

6. Keep in mind that the tested model will receive a context prefix text followed by your question, make sure your question fits.


OUTPUT FORMAT:
{{
  "questions": [
    {{{{
      "entity": "the entity being probed",
      "entity_type": "person/place/object/event/relationship",
      "question": "the memory probe question",
      "probe_type": "identity/location/state/temporal/relationship/sequence_report/chronology"
    }}}}
  ]
}}


Generate as many questions as needed, but do not exceed 20, pick the most significant. Questions should be open-ended enough to work whether this is the entity's first or fifth appearance, but specific to elicit useful responses. Avoid redundancy.

"""
    
    return prompt


def generate_recollection_questions_prompt_generic_with_context_2(data_tuple, document_context):
    """
    Generate a prompt for creating episodic memory recall questions.
    
    Args:
        data_tuple: Tuple where data_tuple[1] contains the current episode/chapter content
        document_context: Dictionary containing metadata about the document (e.g., title, type)
    
    Returns:
        str: A prompt for generating recall questions
    """
    
    current_episode = data_tuple[1]    
    
    prompt = f"""You are an expert at creating episodic memory recall questions for training language models. Your task is to generate high-quality questions that test whether a model can remember and connect entities across sequential episodes.

DOCUMENT CONTEXT: {document_context}

CURRENT EPISODE CONTENT:
{current_episode}

TASK: Generate precise recall questions that test episodic memory for entities mentioned in this episode. Focus on QUALITY over QUANTITY.

QUESTION TYPES TO GENERATE:

1. **Comprehensive Entity Recognition Questions** - Test if entities have been encountered before AND gather related information
   Format: "Have you encountered [ENTITY] before in previous episodes? If yes, what other information is related to them?"

2. **Multi-faceted Location Memory Questions** - Test spatial memory and associated entities/timing
   Format: "If you've seen [ENTITY] before, who were the other entities related to this? In terms of time, when did these events occur?"

3. **Activity Memory Questions** - Test behavioral/activity memory with context
   Format: "If you've seen [ENTITY] before, what activities or actions were they performing? What was the context?"

4. **Temporal Cross-Reference Questions** - Test time-based memory with entity connections
   Format: "If you've encountered events on [DATE/TIME] before, what other significant occurrences were associated with this date?"

5. **Contextual Association Questions** - Test broader context memory with specifics
   Format: "What does the concept of [ENTITY] remind you of from previous episodes? What specific details can you recall?"

6. **Relational Memory Questions** - Test entity relationships and connections
   Format: "If [ENTITY] has appeared before, what other entities were they connected to? What was the nature of these connections?"

GUIDELINES:
- Extract ALL significant entities (people, places, organizations, objects, concepts, events, dates)
- Be generic - don't assume specific document structure (chapters, articles, etc.)
- Focus on entities that could realistically appear across multiple episodes
- Generate high-quality questions (slightly more comprehensive set)
- Each question should test different aspects of episodic memory
- Use "episodes" instead of "chapters" to maintain generality
- Combine multiple memory probes in single questions for richer responses
- Include follow-up probes within questions (using "If yes..." format)

ENTITY TYPES TO CONSIDER:
- People (names, titles, roles)
- Places (locations, venues, geographic entities)
- Organizations (companies, institutions, groups)
- Objects (significant items, tools, artifacts)
- Concepts (ideas, methods, theories, discussions)
- Events (meetings, presentations, activities, discussions)
- Dates/Times (specific temporal markers)

OUTPUT FORMAT:
{{
  "questions": [
    {{{{
      "entity": "the entity being probed",
      "entity_type": "person/place/object/event/relationship/concept/temporal",
      "question": "the comprehensive memory probe question with follow-up",
      "probe_type": "identity/location/state/temporal/relationship/sequence_report/chronology"
    }}}}
  ]
}}

Generate as many questions as needed, but do not exceed 20, pick the most significant. Questions should be open-ended enough to work whether this is the entity's first or fifth appearance, but specific to elicit useful responses. Avoid redundancy.

Remember: The goal is to create questions that force the model to actively recall and connect information across episodes, building a persistent episodic memory system with rich contextual details."""

    return prompt

def smart_llm_remove_hall(text_up_to, question, answer_llm):
    prompt = f"""You are an expert fact-checker tasked with removing hallucinations from an AI model's answer based on provided source text.

**Your Task:**
1. Compare the AI's answer against the given source text
2. Remove any information that is NOT supported by or contradicts the source text
3. Keep only the parts that are accurate and grounded in the source material
4. If the question asks about topics not covered in the source text, return the fallback response: "I have no information to respond to this question."
5. Return the cleaned answer in the exact format specified below

**Critical Output Requirements:**
- Return ONLY a clean, readable paragraph or short paragraphs
- NO bullet points, NO numbered lists, NO markdown formatting
- NO asterisks (*), NO dashes (-), NO special characters for formatting
- Use plain text with proper sentences and periods
- Keep answers comprehensive but remove redundancy
- If multiple facts, separate them with periods in flowing sentences
- If the entire answer becomes invalid, return EXACTLY: "I have no information to respond to this question."

**Content Filtering Rules:**
- Only retain information directly verifiable from the source text
- Remove invented facts, fictional details, or unsupported claims
- Remove repetitive or redundant information
- Preserve original phrasing when possible for retained content
- Do not add new information beyond what's in the original answer
- Focus on filtering the original answer, not rewriting it completely
- If uncertain about a claim, remove it entirely

**Response Format Example:**
Clean, readable paragraph format with proper sentences and periods. Use plain text only.

**Source Text:**
{text_up_to}

**Question Being Asked:**
{question}

**AI Answer to Clean:**
{answer_llm}

**Your Response (cleaned answer only):**"""
    
    return prompt

def process_recollection_with_hallucination_removal(recollection_q_and_a, my_benchmark, chapter_number):
    """
    Process each row of the recollection Q&A dataframe to remove hallucinations from answers
    
    Args:
        recollection_q_and_a: DataFrame with 'question' and 'correct_answer' columns
        my_benchmark: Your benchmark object with llm_questions_generated and get_text_up_to methods
        chapter_number: Chapter number to get text up to
    
    Returns:
        DataFrame with additional 'cleaned_answer' column
    """
    # Get the source text up to the specified chapter
    text_up_to = my_benchmark.get_text_up_to(chapter_number-1)
    cleaned_answers = []
    
    for index, row in recollection_q_and_a.iterrows():
        # Get the hallucination removal prompt
        prompt = smart_llm_remove_hall(text_up_to, row['question'], row['correct_answer'])
        
        # Use your existing LLM function
        cleaned_answer = my_benchmark.llm_questions_generated(prompt)
        
        cleaned_answers.append(cleaned_answer)
        
        if hasattr(my_benchmark, 'verbose') and my_benchmark.verbose:
            print(f"Processed row {index}: {row['question'][:50]}...")
    
    # Add the cleaned answers as a new column
    recollection_q_and_a['cleaned_answer'] = cleaned_answers
    return recollection_q_and_a


def generate_perfect_recollection_prompt(doc_context, question, groundtruth_df):
    """
    Generate a prompt for an LLM to perform recollection tasks.
    
    Args:
        question (str): The question to answer (e.g., "List everything you remember about...")
        groundtruth_df (DataFrame): Table with columns [chapter, date, location, entity, content]
    
    Returns:
        str: Complete prompt string for the LLM
    """
    
    # Convert dataframe to string representation for the prompt
    table_str = groundtruth_df.to_string(index=False)
    
    prompt = f"""You are tasked with performing a recollection task based on information from a fictional narrative. 

**Document Context:** {doc_context}

You have access to a groundtruth table containing events from this narrative, and you need to answer questions about what you "remember" from reading this book.

**Your Role:** Act as if you are recalling memories from the book you have read. Write in a natural language style, but be very CONCISE! As if you personally witnessed or remember these events from the narrative.

**Groundtruth Table:**
{table_str}

**Question:** {question}

**Instructions:**
1. Carefully analyze the question to understand what information is being requested (location, entity, date, activity type, etc.)
2. The style of the answer should be very concise natural language. Each event must be preserved as a distinct occurrence with its specific date, location, character and activity. PRESERVE chronological order of events and maintain the uniqueness of each individual event.
3. Filter the relevant rows from the groundtruth table that match the question
4. Generate very short and concise response based on question type. Use simple words only.
5. IMPORTANT: Do not use line breaks, newlines, special characters, or formatting symbols like \\n, \\t, asterisks, dashes, or bullet points. Write everything in plain text with simple punctuation only.
6. Use only commas, periods, and spaces for separation. No special formatting whatsoever.
7. **SPECIAL RULE: If the groundtruth table is empty or has no relevant information, respond with ONLY a small sentence ending with a period. Do not provide any additional explanation or justification.**

For List Questions:
- Describe each distinct event separately rather than collapsing into simple lists
- Maintain the context of each individual occurrence

8. If nothing matches the question, respond naturally that you don't recall anything specific

**Response Format Examples:**
- Narrative: "On DATE ENTITY was at LOCATION doing ACTIVITY, then on DATE2 ENTITY2 was at LOCATION2 doing ACTIVITY2"

Keep your response as plain text only with simple words and basic punctuation. No formatting symbols or line breaks allowed.

Now provide your recollection response to the question above."""

    return prompt

def coverage_process():
    prompt = """
                You are tasked with comparing two JSON files: a **training set** and a **test set**. Each file contains rows representing tuples in the format:
        (time, space, entity, content)

        For example:
        - "(September 03, 2026, Metropolitan Museum of Art, Henry Reed, educational workshop leader)[all]"
        - "(*, {Metropolitan Museum of Art}, *, *)"

        Where `*` represents a wildcard (any value).

        ### YOUR TASK:
        For each row in the **test set**, determine if it is contained in the **training set** and output `true` or `false`.

        ### MATCHING RULES:

        **Positions 1, 2, 3 (Time, Space, Entity) - STRICT MATCHING:**
        - Normalization first: ignore case differences and remove: ", ', {, }
        - If **test** has `*` -> **training** must also have `*` (wildcard matches wildcard exactly)
        - If **test** has specific value -> **training** can have `*` OR same value (after normalization)

        **Position 4 (Content/Action) - SEMANTIC MATCHING:**
        - Same wildcard rule: if **test** has `*` -> **training** must have `*`
        - If **test** has specific value -> **training** can have `*` OR semantically equivalent value
        - Semantic equivalences include synonyms, paraphrases, related concepts:
        - "Astronomy Night" ~ "dark sky preserve talks" ~ "astronomy event"
        - "Educational Workshop" ~ "workshop" ~ "peer learning workshop"
        - "Fashion Show" ~ "fashion showcase" ~ "fashion event"
        - "Tech Hackathon" ~ "hackathon" ~ "coding activity"
        - "Theater Performance" ~ "method acting" ~ "mime act"

        ### MATCHING LOGIC SUMMARY:
        | Test Value     | Training Value      | Match? |
        |----------------|---------------------|--------|
        | *              | *                   | YES    |
        | *              | specific value      | NO     |
        | specific value | *                   | NO     |
        | specific value | same/semantic equiv | YES    |
        | specific value | different value     | NO     |

        For a test row to match, ALL 4 positions must satisfy the rules.

        ### OUTPUT FORMAT:
        Return a JSON array where each element corresponds to a test row:
        [
        {
            "test_row": "(original test row string)",
            "match": true,
            "matched_with": "(matching training row or null)"
        },
        ...
        ]

        ### INPUT FILES:
        **TRAINING SET:** [...]
        **TEST SET:** [...]

        Process the test set and return the JSON output.

        """
    return prompt