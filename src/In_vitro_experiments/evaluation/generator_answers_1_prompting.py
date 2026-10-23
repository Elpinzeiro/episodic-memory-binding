from datetime import datetime  
from In_vitro_experiments.models.settings_wrapper import SettingsWrapper 
from In_vitro_experiments.models.models_wrapper import ModelsWrapper
from In_vitro_experiments.io.io import answer_recollection_filepath_func,answer_dirpath_func,answer_filepath_func,finetuning_question_filepath_func, answer_reasoning_filepath_func, evaluate_filepath_func, chronological_filepath_func, import_list, export_list
from In_vitro_experiments.evaluation.scoring_answers import evaluate_answer, evaluate_chronological
from In_vitro_experiments.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from In_vitro_experiments.evaluation.generator_answers_2_rag import query_message
from In_vitro_experiments.generation.printing import split_chapters_func
from In_vitro_experiments.generation.prompts import generate_perfect_recollection_prompt
from typing import Dict, List, Union, Optional, Any
import pandas as pd
import ast
import json
import torch 
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments 
from peft import get_peft_model, PeftModel
from tqdm import tqdm
from In_vitro_experiments.generation.qa_generation import retrieve_dataframe_for_all_questions
from pathlib import Path



def check_and_remove(book, substring, error_if_not_found = True):
    count = book.count(substring)
    # Handle different cases
    if count == 0:
        if error_if_not_found:
            raise ValueError(f"Substring '{substring}' not found in document")
        else:
            return book
    elif count > 1:
        raise ValueError(f"Substring '{substring}' found {count} times, expected exactly once")
    else:  # count == 1
        # Remove the single occurrence
        return book.replace(substring, "", 1)
    
def patch_for_ensuring_token_size_lower_130k_in_llama3(book):
    # "maximum context length is 131000 tokens. However, you requested about 131878 tokens"
    # We are looking for some unused paragraphs to remove, for which no useful information is added:
    # max_idx_plus_2 = [2+max(a,b,c,d) for a,b,c,d in zip(df_book_groundtruth['idx_t'], df_book_groundtruth['idx_s'], 
    #                                                     df_book_groundtruth['idx_e'], df_book_groundtruth['idx_c'])]
    # df_book_groundtruth[df_book_groundtruth['nb_paragraphs'] > max_idx_plus_2]
    # > Chapter 13: 9 paragraphs, information in paragraphs {1,2,4,1}
    # > Chapter 139: 10 paragraphs, information in paragraphs {1,2,1,6}
    # Last 2 paragraphs of 139 are removed (also not containing additional entities), etc. for some other chapters
    # 131878 --> (remove 139,13,133,152,167 tokens) --> 131216
    substring139 = """

The lights began to flicker more violently, plunging the hall into moments of complete darkness. In those brief seconds, Julian heard whispers and movements that shouldn't have been possible. His heart pounded in his chest as he frantically tried to save his work, the laptop screen now displaying impossibly twisted images.

As the situation descended into chaos, Julian found himself trapped in a nightmare of his own creation. The UI he had designed seemed to come alive, reaching out from the screen with tendrils of corrupted code. He stumbled back, watching in horror as the digital realm bled into reality, transforming the museum into a grotesque fusion of technology and primal fear."""
    substring13 = """

As the festival drew to a close, Scarlett collapsed into a nearby chair, her clipboard abandoned on the floor beside her. She watched as the last stragglers stumbled towards the exit, their pockets bulging with bottle openers and branded coasters. Despite the chaos, a small smile played on her lips. After all, in the world of craft beer, there was always another festival just around the corner – and hopefully, this time, she'd remember to wear shoes with better traction on beer-soaked floors."""
    substring133 = """

Suddenly, the lights flickered and went out, plunging the room into darkness. Gasps and confused murmurs filled the air. Scarlett felt a brush of movement beside her, followed by the sound of hurried footsteps. When the lights came back on moments later, she immediately sensed that something was amiss.

Her eyes darted around the room, taking in the startled faces of the other attendees. It was then that she noticed an empty space on the wall where one of the most valuable photographs had been hanging just moments before. The theft had been swift and silent, executed with precision in those few seconds of darkness.

As security personnel rushed to secure the exits, Scarlett's mind raced. She recalled Jonathan Rea's suspicious behavior, Stevie Paterson's heated discussion, and the conveniently timed blackout. The pieces of the puzzle were there, but how did they fit together? She took a deep breath, steeling herself for the investigation that was sure to follow. This photography exhibition had just become far more intriguing than she could have ever anticipated."""
    substring152 = """

As the day wore on, the air grew thick with the scent of coffee and the electric tang of overclocked processors. Levi's creation began to take shape, a shimmering construct of light and data that hovered above his workstation like a sentient cloud. Other hackers paused in their work to marvel at the beauty of his code, its elegant simplicity masking layers of complexity beneath.

With mere minutes left before the final bell, Levi put the finishing touches on his masterpiece. The integrated services pulsed with life, each one a testament to his skill and vision. As he stepped back to admire his work, the entire hall erupted in applause. In that moment, surrounded by the timeless beauty of art and the cutting edge of technology, Levi knew that he had woven a spell that would change the world."""
    substring167 = """

With a wave of his hand, he conjured a shimmering aurora that danced across the dome, its ethereal light casting a soft glow over the awestruck faces below. In that instant, the boundary between science and magic blurred, leaving only wonder in its wake.

As the final constellations faded and the lights slowly came up, a chorus of applause filled the air. He bowed slightly, feeling the weight of countless dreams and aspirations settling on his shoulders. Tonight had been more than just an astronomy show; it had been a glimpse into the infinite, a reminder of humanity's place in the vast cosmic dance. And as the crowd began to disperse, he knew that the magic of this night would linger in their hearts long after the last star had faded from view."""
    substring64 = """

Panic rising in his chest, Henry tried to push his way through the press of bodies, desperate to find an exit. But every time he thought he'd found a way out, he found himself back in the center of the carnival, surrounded by leering faces and discordant music. The walls seemed to be closing in, the ceiling lowering with each passing moment.

As the night wore on, the carnival grew more frenzied, the performances more twisted and bizarre. Henry's mind began to fragment, unable to process the horrors unfolding before him. He clung to the hope that dawn would bring an end to this nightmare, but a small voice in the back of his mind whispered that the sun might never rise again in this unholy place."""
    substring25 = """ She moved to intervene, her role as organizer momentarily forgotten in the face of genuine conflict. But as she approached, a blood-curdling scream pierced the air, freezing everyone in their tracks. This wasn't part of the script.

In that moment of shared horror, Mila realized the tragic irony of her carefully orchestrated event. The very atmosphere of suspicion and intrigue she had cultivated had given rise to something far more sinister than she could have ever imagined. As chaos erupted around her, guests panicking and security rushing in, she stood rooted to the spot, the weight of unintended consequences crushing down upon her. The murder mystery dinner she had dreamed would be her crowning achievement now threatened to become her downfall, the line between entertainment and tragedy irrevocably blurred in the echoing halls of the museum."""
    book = check_and_remove(book, substring139) # 182 tokens
    book = check_and_remove(book, substring13) # 131 tokens
    book = check_and_remove(book, substring133) # 284 tokens
    book = check_and_remove(book, substring152) # 211 tokens
    book = check_and_remove(book, substring167) # 199 tokens
    book = check_and_remove(book, substring64)
    book = check_and_remove(book, substring25)
    return book

def whether_do_this_q(q, q_max):
    if q_max is None:
        return True
    else:
        return (q < q_max)
    
def ask_recoll_questions(
    recollection_questions,
    my_benchmark,
    answer_path, 
    answering_parameters: Dict = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15},
    data_folder: Union[str, Path] = 'data/narratives',    
    env_file: Union[str, Path] = '.env',    
    debug=False): 

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    max_new_tokens = answering_parameters.get('max_new_tokens_generation', 1024)
    model_name = answering_parameters['model_name']
    config = SettingsWrapper(_env_file = env_file)
    previous_model = answering_parameters['previous_model']
    lora_path = str(answer_path)                                  
    lora_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', lora_path)
    lora_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', lora_path)
    lora_path = lora_path + "/lora_adapters"
    print(f"Doing Recollection Questions, i'll use this LoRa path: {lora_path}")
    my_model_wrapper_instance = ModelsWrapper(model_name, config, answering_parameters, lora_path)
    system_prompt_for_generation = "You are an expert in memory tests regarding the fictional book \"Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition\"."
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    
    # Add this list to collect results
    results = []
            
    for q, question in enumerate(tqdm(recollection_questions, desc="Processing")):
        user_prompt_str = my_benchmark.get_user_prompt_for_finetuning(question) 
        messages_for_model = [{"role": "system", "content": system_prompt_for_generation}, {"role": "user", "content": user_prompt_str}]
        out, reasoning = my_model_wrapper_instance.generate(
                messages_for_model=messages_for_model,
                max_new_tokens=max_new_tokens                
            )
        
        # Add this line to collect question and answer
        results.append({'question': question, 'correct_answer': out})
        
        answer_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='answer')

        answer_filepath.parent.mkdir(parents=True, exist_ok=True)
        export_list(out, answer_filepath)
        print(f"Exported fine-tuning answer {q} to {answer_filepath}")

        question_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='question')
        export_list(question, question_filepath)
        print(f"Exported fine-tuning question {q} to {question_filepath}")

    # Replace the return statement with this
    return pd.DataFrame(results)

def filter_by_chapter_and_element(df, max_chapter, elem):
    """
    Filter dataframe by max_chapter and element presence.
    
    Parameters:
    df: pandas DataFrame
    max_chapter: int - maximum chapter number to include
    elem: str - element to search for in any column
    
    Returns:
    pandas DataFrame - filtered rows
    """
    # First filter by chapter
    chapter_filtered = df[df['chapter'] <= max_chapter]
    
    # Then filter by element presence in any column
    # Convert all columns to string and check if elem is contained in any of them
    mask = chapter_filtered.astype(str).apply(
        lambda row: row.str.contains(elem, case=False, na=False).any(), 
        axis=1
    )
    
    return chapter_filtered[mask]

def ask_recoll_questions_smart_llm(
    recollection_questions,
    my_benchmark,
    recollecting_approach,
    answer_path, 
    answering_parameters: Dict = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15},
    data_folder: Union[str, Path] = 'data/narratives',    
    env_file: Union[str, Path] = '.env',    
    debug=False,
    
    max_chapter: int = None): 

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    max_new_tokens = answering_parameters.get('max_new_tokens_generation', 1024)
    
    answering_parameters = answering_parameters.copy()  # Creates a new dict
    answering_parameters['recollecting_approach'] = recollecting_approach
    
    # Static document context
    doc_context = 'The Excerpt is from the "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition" fictional book'
    
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    
    # Add this list to collect results
    results = []
    
    # Shared data extraction for generic_qa and magic_12
    elements = None
    if recollecting_approach in ['generic_qa', 'magic_12']:
        data_list = my_benchmark.get_chapter(max_chapter)[0]
        t, s, ent, c = data_list[:4]
        elements = [t, s, ent, c]
    
    # Determine question iteration structure based on approach
    if recollecting_approach == 'prompt_with_context':
        questions_to_process = recollection_questions.get('questions', [])
        desc = "Processing prompt_with_context"
    else:
        questions_to_process = recollection_questions
        desc = f"Processing {recollecting_approach}"
      
    print(f"Processing {recollecting_approach} approach...")
    
    # CHECK IF ALL FILES ALREADY EXIST - CACHING LOGIC
    all_files_exist = True
    existing_results = []
    
    for q in range(len(questions_to_process)):
        answer_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='answer')
        question_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='question')
        if not (answer_filepath.exists() and question_filepath.exists()):
            all_files_exist = False
            break
    
    # If all files exist, load and return
    if all_files_exist:
        print(f"All files exist for {recollecting_approach}. Loading existing results...")
        for q in range(len(questions_to_process)):
            answer_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='answer')
            question_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='question')
            
            question = import_list(question_filepath)
            answer = import_list(answer_filepath)
            existing_results.append({'question': question, 'correct_answer': answer})
        
        return pd.DataFrame(existing_results)
    
    print(f"Some files missing for {recollecting_approach}. Generating new answers...")
    
    # Helper function to export files (shared logic)
    def export_question_answer(q, question, answer):
        # Export answer
        answer_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='answer')
        answer_filepath.parent.mkdir(parents=True, exist_ok=True)
        export_list(answer, answer_filepath)
        print(f"Exported answer {q} to {answer_filepath}")

        # Export question  
        question_filepath = answer_recollection_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='question')
        export_list(question, question_filepath)
        print(f"Exported question {q} to {question_filepath}")
    
    # Main processing loop
    for q, item in enumerate(tqdm(questions_to_process, desc=desc)):
        
        # Determine element for filtering and question text based on approach
        if recollecting_approach == 'generic_qa':
            elem = elements[q]  # 1:1 mapping
            question = item
            
        elif recollecting_approach == 'magic_12':
            element_index = q // 3  # 3 questions per element
            elem = elements[element_index]
            question = item
            
        elif recollecting_approach == 'prompt_with_context':
            elem = item['entity']  # Use entity from metadata
            question = item['question']
        
        # Filter groundtruth by chapter and element (shared logic)
        groundtruth_df = filter_by_chapter_and_element(
            my_benchmark.df_book_groundtruth[['chapter','date','location','entity','content']], 
            max_chapter=max_chapter, 
            elem=elem
        )
        
        # Generate prompt with context (shared logic)
        prompt = generate_perfect_recollection_prompt(
            doc_context=doc_context, 
            question=question,
            groundtruth_df=groundtruth_df
        )
        
        # Generate answer using smart LLM (shared logic)
        out = my_benchmark.llm_questions_generated(
            prompt=prompt, 
            max_new_tokens=max_new_tokens
        )
        
        # Collect results - always same format: question and correct_answer only
        results.append({'question': question, 'correct_answer': out})
        
        # Export files (shared logic)
        export_question_answer(q, question, out)
    
    # Validation
    if recollecting_approach not in ['generic_qa', 'magic_12', 'prompt_with_context']:
        raise ValueError(f"Unknown recollecting_approach: {recollecting_approach}")
    
    # Return results as DataFrame
    return pd.DataFrame(results)


def generate_answers_func(
    my_benchmark, 
    answering_parameters: Dict = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15},
    data_folder: Union[str, Path] = 'data/narratives',
    env_file: Union[str, Path] = '.env',    
    my_embedding: Optional[object] = None,
    debug=False
    ):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    
    if answering_parameters['kind'] in ['ftuning-manual']:          
        
        if answering_parameters.get('evaluation') == 'mini':

            if answering_parameters['max_chapter'] != my_benchmark.nb_chapters():
                
                print("Here to skip")
                df_qa = None
                # if 'max_chapter' in answering_parameters and answering_parameters['max_chapter'] is not None and answering_parameters['max_chapter'] == 1:
                #     max_chapter = answering_parameters['max_chapter']                
                #     df_qa = get_eval_questions_by_chapter(my_benchmark,max_chapter)
                #     if len(df_qa) == 0:
                #         #df_qa = retrieve_dataframe_for_all_questions([1], my_benchmark.df_book_groundtruth, my_benchmark.events).head(10)
                #         #df_qa = df_qa[df_qa['correct_answer_chapters'].apply(lambda x: len(x) > 0 and (x <= 1).all() if hasattr(x, 'all') else all(ch <= 1 for ch in x))].head(10)
                #         df_qa = pd.read_csv('questions_ch_1.csv').apply(lambda col: col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith(('[', '{')) else x) if col.dtype == 'object' else col)
                #     print(f"During Eval, Applying chapter filter with max_chapter={max_chapter}, n_questions={len(df_qa)}")
                # else : 
                #     df_qa = None
                #     print("Here to skip")
                    # # Mini set: 1 question from each category
                    # df_qa = my_benchmark.get_df_qa().head(1).reset_index(drop=True)
                    
                    # # 1 chronological question
                    # chronological_questions = my_benchmark.df_qa[my_benchmark.df_qa['get'] == 'chronological'].head(1)
                    
                    # # 1 question from bins_items_correct_answer=="0"
                    # bins_zero_questions = my_benchmark.df_qa[my_benchmark.df_qa['bins_items_correct_answer']=="0"].head(1)
                    
                    # # Combine all three sets
                    # df_qa = pd.concat([df_qa, chronological_questions, bins_zero_questions], ignore_index=True)
            else:
                
                #df_qa = my_benchmark.df_qa[my_benchmark.df_qa['bins_items_correct_answer']!="0"].reset_index(drop=True)
                df_qa = my_benchmark.get_df_qa()
                print("Here I DON'T remove hallucinating questions")

        
        elif answering_parameters.get('evaluation') == 'by_chapter':
            
            if 'max_chapter' in answering_parameters and answering_parameters['max_chapter'] is not None:
                max_chapter = answering_parameters['max_chapter']                
                df_qa = get_eval_questions_by_chapter(my_benchmark,max_chapter)
                print(f"During Eval, Applying chapter filter with max_chapter={max_chapter}, n_questions={len(df_qa)}")
                
        elif answering_parameters.get('not_eval_hall', False) == True:
            df_qa = my_benchmark.get_df_qa()
            df_qa = df_qa[df_qa['n_items_correct_answer']!=0].reset_index(drop=True)  # ← ADD THIS!

            print(f"Using questions no bins==0: {len(df_qa)} questions to generate answers")
        elif answering_parameters.get('grokking', False) == True:
            print("DEBUUUUUUUUUUUUUUUUUUUUGGGGGGGGGGGGGGGGGGGGGG GROKKING INFERENCE!")
            #df_qa = my_benchmark.finetuning_questions[my_benchmark.finetuning_questions['n_chapters_correct_answer'] != 1].reset_index(drop=True) 
            df_qa = my_benchmark.get_df_qa()
            df_qa = df_qa[df_qa['n_items_correct_answer']!=0].reset_index(drop=True)
        else:
        
            df_qa = my_benchmark.get_df_qa()
            print(f"Using original questions for evaluation: {len(df_qa)} questions")                 
            
    else:
        
        df_qa = my_benchmark.get_df_qa()
        print("DEBUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        
    model_name = answering_parameters['model_name'] 
    max_new_tokens = answering_parameters.get('max_new_tokens_generation', 1024)  # Default to 1024
    system_prompt_for_generation = "You are an expert in memory tests regarding the fictional book \"Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition\"."
    sleeping_time_for_api = answering_parameters.get('sleeping_time', 0) 
    temperature = answering_parameters.get('temperature', 0.1)

    config = SettingsWrapper(_env_file = env_file)

    book = my_benchmark.get_book()
    if model_name == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870:
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    
    if 'max_chapter' in answering_parameters and answering_parameters['max_chapter'] is not None:
                max_chapter = answering_parameters['max_chapter']
    else:
        max_chapter = "no max chapter"
    # === EPISODIC MEMORY QUESTIONS ===
    generated_answers = []

    if df_qa is not None and not df_qa.empty:
        for q in range(len(df_qa)):
            answer_filepath = answer_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            answer_reasoning_filepath = answer_reasoning_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            #print(f"Looking into :{answer_filepath}")
            if not answer_filepath.is_file():
                question = df_qa.iloc[q]['question']
                correct_answer = df_qa.iloc[q]['correct_answer']
                #print(f"Generate {str(q)} / {str(len(df_qa)-1)} [{correct_answer} for question {question}]")
                print(f"Generate {str(q)} / {str(len(df_qa)-1)}, chapter: {max_chapter}, [{correct_answer}for question {question}] ")
                try:
                    my_model_wrapper_instance
                except NameError:
                    # --- Initialize ModelsWrapper ONCE at the beginning ---
                    my_model_wrapper_instance = ModelsWrapper(model_name, config, answering_parameters)
                    
                user_prompt_str = "" 
                messages_for_model = [] 

                # Prepare messages/prompts based on 'kind'
                if answering_parameters['kind'] == 'prompting':
                    user_prompt_str = generate_episodic_memory_prompt(book, question)
                    messages_for_model = [{"role": "system", "content": system_prompt_for_generation}, {"role": "user", "content": user_prompt_str}]
                elif answering_parameters['kind'] == 'rag':
                    user_prompt_str = query_message(question, my_embedding, answering_parameters, env_file)
                    messages_for_model = [{"role": "system", "content": system_prompt_for_generation}, {"role": "user", "content": user_prompt_str}]
                elif answering_parameters['kind'] == 'ftuning' or answering_parameters['kind'] == 'ftuning-manual':
                    user_prompt_str = my_benchmark.get_user_prompt_for_finetuning(question) 
                    messages_for_model = [{"role": "system", "content": system_prompt_for_generation}, {"role": "user", "content": user_prompt_str}]
                
                if q == 0 and debug == True:
                    print("[begin example of a prompt]")
                    if my_model_wrapper_instance.is_local_hf_model:
                        print(my_model_wrapper_instance.hf_tokenizer.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True))
                    else: 
                        print(user_prompt_str) 
                    print("[end example of a prompt]")

                # --- Unified generation call ---
                out, reasoning = my_model_wrapper_instance.generate(
                    messages_for_model=messages_for_model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Apply sleeping time ONLY if it was an API call
                if not my_model_wrapper_instance.is_local_hf_model:
                    time.sleep(sleeping_time_for_api) 

                answer_filepath.parent.mkdir(parents=True, exist_ok=True)
                export_list(out, answer_filepath)
                for name, value in [("correct_answer", correct_answer), ("question", question)]: 
                    open(answer_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters).with_name(f"{q}_{name}.txt"), 'w', encoding='utf-8').write(str(value)) 
                    
                ftuning_question_filepath = finetuning_question_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
                # if answering_parameters['kind'] in ['ftuning-manual'] and not ftuning_question_filepath.is_file():
                #     correct_answer_serializable = list(df_qa.iloc[q]['correct_answer']) if isinstance(df_qa.iloc[q]['correct_answer'], set) else (list(df_qa.iloc[q]['correct_answer']) if df_qa.iloc[q]['correct_answer'] else [])                
                #     question_data = {
                #         'question': df_qa.iloc[q]['question'],
                #         'correct_answer': correct_answer_serializable,
                #         'retrieval_type': df_qa.iloc[q]['retrieval_type'] if 'retrieval_type' in df_qa.columns else None,
                #         'correct_answer_chapters': list(df_qa.iloc[q]['correct_answer_chapters']) if 'correct_answer_chapters' in df_qa.columns and isinstance(df_qa.iloc[q]['correct_answer_chapters'], set) else (list(df_qa.iloc[q]['correct_answer_chapters']) if 'correct_answer_chapters' in df_qa.columns and df_qa.iloc[q]['correct_answer_chapters'] else []),
                #         'get': df_qa.iloc[q]['get'] if 'get' in df_qa.columns else None,
                #         'question_index': q,
                #         'original_index': int(df_qa.index[q]) if hasattr(df_qa, 'index') else q,
                #         'filter_applied': 'max_chapter' in answering_parameters,
                #         'max_chapter': answering_parameters.get('max_chapter', None)
                #     }
                
                #     ftuning_question_filepath.parent.mkdir(parents=True, exist_ok=True)
                #     export_list(question_data, ftuning_question_filepath)
                #     print(f"Exported fine-tuning question {q} to {ftuning_question_filepath}")

                if reasoning is not None:
                    answer_reasoning_filepath.parent.mkdir(parents=True, exist_ok=True)
                    export_list(reasoning, answer_reasoning_filepath)
            
            generated_answer = import_list(answer_filepath)
            generated_answers.append(generated_answer)
        
        df_generated_answers = pd.concat([df_qa, pd.DataFrame({'llm_answer':generated_answers})], axis = 1)
    else:
        
        df_generated_answers = pd.DataFrame()  # empty DataFrame

    print(f"inside generate answers: {len(df_generated_answers)}")
    print(f"df_qa shape: {df_qa.shape}")
    print(f"generated_answers length: {len(generated_answers)}")
    return df_generated_answers




def is_valid_chapter_string(s):
     pattern = r'^Full chapter \d+$'
     return bool(re.match(pattern, s))

def extract_chapter_number(s):
     match = re.search(r"Full chapter (\d+)", s)
     if match:
         return int(match.group(1))
     else:
         raise ValueError("String does not match the expected format")

         


def generate_evaluation_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_answers,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = 'data/narratives',
    env_file = '.env',
    evaluation_parameters = None,
    debug=False,
    max_retries=3,  # New parameter for retry attempts
    retry_delay=10,  # New parameter for delay between retries
    only_scored=True):  # judge only the questions the paper scores (get == 'all')

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    config = SettingsWrapper(_env_file = env_file)

    # model parameters
    model_name = model_parameters['model_name'] # using the model that built the benchmark, not the one answering the questions
    # Use evaluation_parameters if provided, otherwise fall back to original logic
    # Use evaluation_parameters if provided, otherwise fall back to original logic
    if evaluation_parameters is not None:
        judge_model_name = evaluation_parameters['model_name']
        judge_answering_parameters = evaluation_parameters.copy()
        #print(f"=== USING EVALUATION PARAMETERS ===")
        #print(f"Judge model: {judge_model_name}")
        #print(f"Judge parameters: {judge_answering_parameters}")
    else:
        # Fallback to original logic
        judge_model_name = model_parameters['model_name']
        judge_answering_parameters = {'kind': 'prompting', 'model_name': judge_model_name, 'max_new_tokens': 4096}
        print(f"=== USING FALLBACK PARAMETERS ===")
        print(f"Judge model: {judge_model_name}")
    
    # ... existing code ...

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    split_chapters = my_benchmark.split_chapters
    

    if answering_parameters['model_name'] == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870: # 102870 for our count, but 131878 for llama3
            # "maximum context length is 131000 tokens. However, you requested about 131878 tokens"
            # We are looking for some unused paragraphs to remove, for which no useful information is added
            # We also put this information at the evaluation stage (needed for evaluating the full chapters)
            book = my_benchmark.book
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)
            split_chapters = split_chapters_func(book)
    
    # question/true answer and additionally containing the generated answers
    df_qa2 = df_generated_answers.reset_index(drop=True)
    generated_evaluations = []
    judged = []  # row indices of df_qa2 sent to the judge

    def is_json_parsing_error(e):
        """The judge answered, but not in the expected JSON format (bad JSON, or wrong structure such as a string instead of a dict)"""
        return (isinstance(e, (json.JSONDecodeError, AttributeError, KeyError)) or
            (isinstance(e, ValueError) and ("Failed to parse judge's response" in str(e) or "nb_gt" in str(e))))

    def is_overload_error(exception):
        """Check if the exception is an overload error (HTTP 529)"""
        error_str = str(exception)
        return "529" in error_str or "overloaded_error" in error_str or "Overloaded" in error_str

    def create_fallback_evaluation(correct_answer, error_message):
        """Create a fallback evaluation with score 0"""
        
        correct_answer = list(correct_answer) if correct_answer is not None else []

        if correct_answer is not None:
            matching_groundtruth_items_score = [{item: 0} for item in correct_answer]
        else:
            matching_groundtruth_items_score = []

        return {
            'identified_items_in_AI_answer': [],
            'matching_score': matching_groundtruth_items_score,
            'explanation': f"Evaluation failed due to error: {error_message}. Assigned score 0.",
            'evaluation_score': 0.0,
            'evaluation_reasoning': "Model response did not match expected format or content",
            'predicted_items': [],
            'groundtruth_items': correct_answer if correct_answer else [],
            'matching_groundtruth_items_score': matching_groundtruth_items_score,
            'nb_preds': 0,
            'nb_gt': len(correct_answer) if correct_answer else 0,
            'sum_scores': 0.0,
            'precision': 0.0,
            'recall': 0.0 if correct_answer else None,
            'f1_score': 0.0
        }

    # loop
    for q in range(len(df_qa2)):
        if only_scored and df_qa2.iloc[q]['get'] != 'all':
            continue  # chronological / latest-event questions are not in the scores
        judged.append(q)
        evaluate_filepath = evaluate_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        if not evaluate_filepath.is_file():
            question = df_qa2.iloc[q]['question'] # just for the printing
            llm_answer = df_qa2.iloc[q]['llm_answer']
            correct_answer = df_qa2.iloc[q]['correct_answer']
            retrieval_type = df_qa2.iloc[q]['retrieval_type']
            get_style = df_qa2.iloc[q]['get']

            if debug == False:
                print(f"Evaluate {str(q)} / {str(len(df_qa2)-1)}, using judge = {judge_model_name}")
                
            else:
                print(f"Evaluate {str(q)} / {str(len(df_qa2)-1)} [question {question}]")
            
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(judge_model_name, config, judge_answering_parameters)
                print(f"Initialized judge model: {judge_model_name}")

            # update the answer for full events
            if len(correct_answer) == 1:
                #print(correct_answer[0])
                correct_answer = list(correct_answer)
                if is_valid_chapter_string(correct_answer[0]):
                    #print('need to change with actual chapter')
                    chapter_number = extract_chapter_number(correct_answer[0])
                    #print(chapter_number)
                    correct_answer_long = split_chapters[chapter_number] # does not need to be a list in this case
                    #print("[begin book chapter]")
                    #print(correct_answer_long)
                    #print("[end book chapter]")
                else:
                    correct_answer_long = None
            else:
                correct_answer_long = None

            # Retry logic for evaluation
            retry_count = 0
            evaluation_successful = False
            base_max_tokens = answering_parameters.get('max_new_tokens', 2048)

            while retry_count <= max_retries and not evaluation_successful:
                try:
                    current_max_tokens = base_max_tokens * (2 ** retry_count)
                    out = evaluate_answer(llm_answer, correct_answer, retrieval_type, my_model, correct_answer_long, debug, max_new_tokens=current_max_tokens)
                    evaluation_successful = True
                    
                except Exception as e:
                    if (is_overload_error(e) or is_json_parsing_error(e)) and retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        error_type = "Overload" if is_overload_error(e) else "JSON parsing"
                        print(f"⚠️ {error_type} error for question {q} (attempt {retry_count}/{max_retries}): {e}")
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    elif is_json_parsing_error(e):
                        # the judge answered, but not in the expected format: score 0, counted and reported at the end
                        print(f"⚠️ Max retries ({max_retries}) reached for JSON parsing error on question {q}: {e}")
                        print("Creating fallback evaluation with score 0...")
                        out = create_fallback_evaluation(correct_answer, str(e))
                        evaluation_successful = True  # Exit the retry loop
                    else:
                        # API errors (missing or wrong key, no network, overload after all retries) must not be scored as wrong answers
                        raise RuntimeError(
                            f"Judge call failed on question {q} (judge {judge_model_name}): {type(e).__name__}: {e}\n"
                            "Check the judge API key in .env and the network access. Evaluations already saved are kept, so a rerun continues. "
                            "Without internet (e.g. a GPU node), train with --skip-judge and judge the saved answers later with "
                            "src/In_vitro_experiments/evaluation/judge_saved_answers.py") from e
            
            time.sleep(2)  # TO AVOID API ERRORS
            evaluate_filepath.parent.mkdir(parents=True, exist_ok=True)
            #print(evaluate_filepath)
            export_list(out, evaluate_filepath)
            
        generated_evaluation = import_list(evaluate_filepath)
        generated_evaluations.append(generated_evaluation)
        
    #df_generated_answers = pd.concat([df_qa2, pd.DataFrame({'llm_answer':generated_answers})], axis = 1)
        
    df_generated_evaluations = pd.DataFrame(generated_evaluations)
    df_generated_evaluations = pd.concat([df_qa2.iloc[judged].reset_index(drop=True), df_generated_evaluations], axis = 1)
    df_generated_evaluations['q_idx'] = judged  # index of the answer file, also when questions are skipped
    n_failed = df_generated_evaluations['explanation'].astype(str).str.startswith('Evaluation failed due to error').sum()
    if n_failed:
        print(f"⚠️ {n_failed} of {len(df_generated_evaluations)} answers could not be judged (bad judge output) and are scored 0. "
              f"Delete their files in {evaluate_filepath.parent} (explanation 'Evaluation failed due to error...') and rerun to judge them again.")

    return df_generated_evaluations


def generate_chronological_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_evaluations,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = 'data/narratives',
    env_file = '.env',
    evaluation_parameters = None,
    debug=False):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    # model parameters
    model_name = model_parameters['model_name'] # using the model that built the benchmark, not the one answering the questions
    
    config = SettingsWrapper(_env_file = env_file)

    if evaluation_parameters is not None:
        judge_model_name = evaluation_parameters['model_name']
        judge_answering_parameters = evaluation_parameters.copy()
        #print(f"=== USING EVALUATION PARAMETERS ===")
        #print(f"Judge model: {judge_model_name}")
        #print(f"Judge parameters: {judge_answering_parameters}")
    else:
        # Fallback to original logic
        judge_model_name = model_parameters['model_name']
        judge_answering_parameters = {'kind': 'prompting', 'model_name': judge_model_name, 'max_new_tokens': 4096}
        print(f"=== USING FALLBACK PARAMETERS ===")
        print(f"Judge model: {judge_model_name}")
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    df_qa3 = df_generated_evaluations

    generated_chronologicals = []

    # loop
    for q in range(len(df_qa3)):
        if df_qa3.iloc[q]['get'] == 'chronological': # only consider the chronological questions
            chronological_filepath = chronological_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)

            if not chronological_filepath.is_file():
                predicted_items = df_qa3.iloc[q]['predicted_items']
                groundtruth_items = df_qa3.iloc[q]['groundtruth_items']
                question = df_qa3.iloc[q]['question'] # just for the printing
                print(f"Evaluate {str(q)} / {str(len(df_qa3)-1)} [question {question}]")
                # only initialize the model if needed, and only initialize it once 
                try:
                    my_model
                except NameError:
                    my_model = ModelsWrapper(judge_model_name, config, judge_answering_parameters)
                    
                    
                #out = evaluate_answer(llm_answer, correct_answer, retrieval_type, my_model, correct_answer_long, debug, max_new_tokens=current_max_tokens)

                # Replace the current truncation section with this:
                original_predicted = predicted_items.copy()
                for trial in range(3):
                    if trial == 0:
                        current_predicted = predicted_items  # Full list
                    elif trial == 1:
                        current_predicted = predicted_items[:len(predicted_items)//2]  # Half
                    else:  # trial == 2
                        current_predicted = predicted_items[:len(groundtruth_items)]  # Same as groundtruth
                    
                    try:
                        out = evaluate_chronological(groundtruth_items, current_predicted, my_model, debug)
                        
                        # Only pad if we truncated AND if predicted_indexes exists
                        if len(current_predicted) < len(original_predicted) and 'predicted_indexes' in out:
                            missing_count = len(original_predicted) - len(current_predicted)
                            out['predicted_indexes'].extend([-1] * missing_count)
                        
                        break
                    except Exception as e:
                        if "finish_reason" in str(e) and trial < 2:
                            continue
                        else:
                            # Create fallback result for complete failure
                            out = {
                                "groundtruth_indexes": list(range(len(groundtruth_items))),
                                "predicted_indexes": [-1] * len(original_predicted),
                                "explanation": f"All trials failed: {str(e)}"
                            }
                            break
                chronological_filepath.parent.mkdir(parents=True, exist_ok=True)
                #print(evaluate_filepath)
                export_list(out, chronological_filepath)
            generated_chronological = import_list(chronological_filepath)
            generated_chronologicals.append(generated_chronological)

    df_generated_chronological = pd.DataFrame(generated_chronologicals)

    return df_generated_chronological

def commonsense_test(my_benchmark, answering_parameters, data_folder, max_questions, output_dir, debug=False):
    """
    Evaluate commonsense reasoning on fine-tuned model using batch processing.
    Returns the summary dictionary or loads it from file if already exists.
    """
    # Get path components for export folder
    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    
    # Get the standard answer directory path
    export_folder = answer_dirpath_func(
        nb_chapters, nb_tokens, data_folder, 
        prompt_parameters, model_parameters, book_parameters, 
        answering_parameters
    )
    export_folder.mkdir(parents=True, exist_ok=True)
    
    summary_file = export_folder / "commonsense_evaluation_summary.json"
    
    # Check if summary already exists
    if summary_file.exists():
        print(f"Found existing commonsense evaluation summary: {summary_file}")
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    print("\n" + "="*60)
    print("EVALUATING COMMONSENSE REASONING ON FINE-TUNED MODEL (BATCH MODE)")
    print("="*60)
    
    
    
    # REUSE THE SAME LOADING CODE AS YOUR FINE-TUNING SECTION
    model_name = answering_parameters['model_name']
    base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,            
            trust_remote_code=True,
            device_map="auto"
        )
    
    adapter_path_str = answering_parameters.get('after_finetuning_path')
    model_path = Path(adapter_path_str) / "lora_adapters"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    
    # Ensure the model's pad_token_id is consistent with the tokenizer
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        print(f"Setting model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id})")
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set up tokenizer for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # IMPORTANT: Set left padding for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Ensure model max length is reasonable
    tokenizer.model_max_length = min(8192, getattr(tokenizer, 'model_max_length', 8192))
    
    # Set model to eval mode
    model.eval()
    
    print("Model loaded successfully for batch processing!")
    print(f"Model device: {model.device}")
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    print(f"Padding side: {tokenizer.padding_side}")
    
    # Get the train split of commonsense dataset (same as baseline)
    commonsense_train = my_benchmark.get_commonsense_split("train")
    
    if commonsense_train is not None:
        if debug:
            print(f"Fine-tuned results will be exported to: {export_folder}")
        
        # Record start time
        start_time = time.time()
        
        # Determine optimal batch size (start with 16 for H100, can be adjusted)
        batch_size = 16
        
        try:
            # Evaluate using batch processing
            commonsense_results = evaluate_commonsense_batch_finetuned(
                model=model,
                tokenizer=tokenizer,
                dataset=commonsense_train,
                batch_size=batch_size,
                max_questions=max_questions,
                debug=debug
            )
            
            evaluation_time = time.time() - start_time
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Batch size {batch_size} caused OOM error. Trying smaller batch size...")
            batch_size = 8
            
            # Reset timer for the retry
            start_time = time.time()
            
            commonsense_results = evaluate_commonsense_batch_finetuned(
                model=model,
                tokenizer=tokenizer,
                dataset=commonsense_train,
                batch_size=batch_size,
                max_questions=max_questions,
                debug=debug
            )
            
            evaluation_time = time.time() - start_time
        
        # Convert numpy types to Python native types for JSON serialization
        n_epochs = answering_parameters.get('n_epochs')
        max_chapter = answering_parameters.get('max_chapter')
        
        # Export only the summary file
        summary = {
            'model_name': f'{answering_parameters["model_name"]} (fine-tuned)',
            'model_type': 'fine-tuned',
            'fine_tuning_params': {
                'n_epochs': int(n_epochs) if n_epochs is not None else None,
                'max_chapter': int(max_chapter) if max_chapter is not None else None,
                'ftuning_input_data_policy': answering_parameters.get('ftuning_input_data_policy')
            },
            'total_questions': int(commonsense_results['total']),
            'correct_answers': int(commonsense_results['correct']),
            'accuracy': float(commonsense_results['accuracy']),
            'batch_size_used': batch_size,
            'evaluation_time_seconds': float(evaluation_time),
            'questions_per_second': float(commonsense_results['total'] / evaluation_time),
            'dataset_split': 'train',
            'evaluation_date': str(datetime.now())
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n🔍 FINE-TUNED MODEL RESULTS (BATCH MODE):")
        print(f"   Model: {answering_parameters['model_name']} (fine-tuned)")
        print(f"   CommonsenseQA Accuracy: {commonsense_results['accuracy']:.2%}")
        print(f"   Questions evaluated: {commonsense_results['total']}")
        print(f"   Correct answers: {commonsense_results['correct']}")
        print(f"   Batch size used: {batch_size}")
        print(f"   Dataset split: TRAIN")
        print(f"   ⏱️  Evaluation time: {evaluation_time:.2f}s ({evaluation_time/60:.2f} minutes)")
        print(f"   ⚡ Processing speed: {commonsense_results['total']/evaluation_time:.2f} questions/second")
        if debug:
            print(f"   Summary saved to: {summary_file}")
        
        return summary
    else:
        print("WARNING: Could not load commonsense train dataset")
        return None

def evaluate_commonsense_batch_finetuned(model, tokenizer, dataset, batch_size=16, max_questions=100, debug=False):
    """
    Evaluate the fine-tuned model on CommonsenseQA dataset using batch processing.
    This is similar to evaluate_commonsense_batch but adapted for fine-tuned models.
    """
    def extract_answer_letter(response: str) -> str:
        """
        Extract the answer letter (A, B, C, D, or E) from the model response.
        """
        import re
        
        # Clean the response
        response = response.strip().upper()
        
        # Common patterns to look for
        patterns = [
            r'\b([A-E])\)',  # "A)", "B)", etc.
            r'\b([A-E])\.',  # "A.", "B.", etc.
            r'ANSWER:\s*([A-E])',  # "Answer: A", "Answer: B", etc.
            r'THE\s+ANSWER\s+IS\s*([A-E])',  # "The answer is A"
            r'^([A-E])',  # Just "A", "B", etc. on its own
            r'\b([A-E])\b',  # Any standalone letter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # If no clear pattern, try to find the first A-E letter
        for char in response:
            if char in 'ABCDE':
                return char
        
        return None  # No valid answer found

    def prepare_batch_prompts(examples):
        """
        Prepare a batch of prompts from examples.
        """
        prompts = []
        for example in examples:
            question = example['question']
            
            # Handle different possible structures for choices
            if isinstance(example['choices'], dict):
                if 'text' in example['choices']:
                    choices = example['choices']['text']
                elif 'label' in example['choices'] and 'text' in example['choices']:
                    choices = example['choices']['text']
                else:
                    choices = list(example['choices'].values())[0] if example['choices'] else []
            else:
                choices = example['choices']
            
            # Format prompt
            prompt = f"Answer the following multiple choice question by selecting the best option.\n\n"
            prompt += f"Question: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{chr(65+j)}) {choice}\n"
            prompt += f"\nThe answer is:"
            
            prompts.append(prompt)
        
        return prompts

    def generate_batch_answers(prompts, batch_size, max_new_tokens=10):
        """
        Generate answers for a batch of prompts using the loaded model.
        """
        all_responses = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        print(f"🔄 Starting batch generation: {total_batches} batches total")
        
        # Process prompts in batches
        for batch_idx in range(0, len(prompts), batch_size):
            current_batch_num = (batch_idx // batch_size) + 1
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            
            # Show batch progress - only detailed if debug=True
            if debug and (current_batch_num == 1 or current_batch_num % max(1, total_batches // 10) == 0 or current_batch_num == total_batches):
                questions_processed = min(batch_idx + len(batch_prompts), len(prompts))
                batch_progress = (current_batch_num / total_batches) * 100
                print(f"⚡ Batch {current_batch_num}/{total_batches} ({batch_progress:.1f}%) | "
                      f"Questions {batch_idx + 1}-{questions_processed}/{len(prompts)}")
            elif not debug and current_batch_num % max(1, total_batches // 4) == 0:  # Show only major batch milestones
                batch_progress = (current_batch_num / total_batches) * 100
                print(f"🔄 Batch generation progress: {batch_progress:.0f}% complete ({current_batch_num}/{total_batches} batches)")
            
            try:
                # Convert prompts to chat format
                batch_messages = []
                for prompt in batch_prompts:
                    messages = [
                        {"role": "system", "content": "You are an expert at answering multiple choice questions. Always respond with just the letter of your chosen answer."},
                        {"role": "user", "content": prompt}
                    ]
                    batch_messages.append(messages)
                
                # Apply chat template to all prompts
                batch_input_strs = []
                for messages in batch_messages:
                    input_str = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    batch_input_strs.append(input_str)
                
                # Tokenize batch with padding
                inputs = tokenizer(
                    batch_input_strs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(2048, tokenizer.model_max_length),
                    return_attention_mask=True
                )
                
                # Move to model device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate responses
                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode responses
                input_lengths = inputs['input_ids'].shape[1]
                batch_responses = []
                
                for j in range(len(batch_prompts)):
                    generated_text = tokenizer.decode(
                        generated_tokens[j, input_lengths:],
                        skip_special_tokens=True
                    ).strip()
                    batch_responses.append(generated_text)
                
                all_responses.extend(batch_responses)
                
            except Exception as e:
                if debug:
                    print(f"❌ Batch generation error for batch {current_batch_num}: {e}")
                # Add error responses for this batch
                error_responses = [f"ERROR: {str(e)}" for _ in batch_prompts]
                all_responses.extend(error_responses)
        
        print(f"✅ Batch generation completed! Generated {len(all_responses)} responses")
        return all_responses

    # Prepare data
    examples = []
    for i, example in enumerate(dataset):
        if i >= max_questions:
            break
        examples.append(example)
    
    print(f"Evaluating fine-tuned model on {len(examples)} questions with batch size {batch_size}...")
    
    # Debug: Print first example structure
    if examples and debug:
        example = examples[0]
        print(f"\n=== DATASET EXAMPLE STRUCTURE ===")
        print(f"Example keys: {list(example.keys())}")
        print(f"Choices type: {type(example.get('choices'))}")
        if 'choices' in example:
            print(f"Choices content: {example['choices']}")
        print(f"AnswerKey: '{example.get('answerKey')}'")
        print("=" * 40)
    
    # Prepare batch prompts
    prompts = prepare_batch_prompts(examples)
    
    # Generate batch responses with progress tracking
    if debug:
        print("Generating responses in batches...")
        print(f"🔄 Processing {len(examples)} questions in batches of {batch_size}")
        print(f"📊 Progress will be shown every batch and milestone updates")
        print("-" * 60)
    else:
        print(f"🔄 Processing {len(examples)} questions in batches of {batch_size}")
    
    responses = generate_batch_answers(prompts, batch_size, max_new_tokens=10)
    
    # Process results
    correct = 0
    total = len(examples)
    
    if debug:
        print("\n" + "="*50)
        print("📋 PROCESSING RESULTS AND CALCULATING ACCURACY")
        print("="*50)
    else:
        print("📋 Processing results...")
    
    # Calculate progress intervals based on debug mode
    if debug:
        progress_interval = min(500, max(1, max_questions // 20))  # Show every 5% or 500 questions
        milestone_interval = max(1000, max_questions // 5)  # Major milestones every 20%
    else:
        # For non-debug mode, only show major milestones
        milestone_interval = max(1000, max_questions // 10)  # Show every 10% but minimum 1000
    
    for i, (example, response) in enumerate(zip(examples, responses)):
        # Extract correct answer
        correct_answer = example.get('answerKey', '')
        if not correct_answer:
            correct_answer = example.get('answer', '')
        if not correct_answer:
            correct_answer = example.get('correct_answer', '')
        if not correct_answer:
            correct_answer = example.get('label', '')
        
        # Extract predicted answer
        predicted = extract_answer_letter(response)
        
        # Check correctness
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        
        # Debug: Print first few examples
        if i < 3 and debug:
            question = example['question']
            print(f"\n--- Question {i} Debug ---")
            print(f"Question: {question}")
            print(f"Correct: {correct_answer}")
            print(f"Model response: '{response}'")
            print(f"Extracted: '{predicted}'")
            print(f"Correct: {is_correct}")
            print("-" * 30)
        
        # Progress reporting based on debug mode
        if debug:
            # Detailed progress for debug mode
            if (i + 1) % progress_interval == 0:
                current_accuracy = correct / (i + 1)
                progress_percentage = ((i + 1) / total) * 100
                print(f"📈 Progress: {i+1:,}/{total:,} ({progress_percentage:.1f}%) | "
                      f"Accuracy: {current_accuracy:.2%} ({correct}/{i+1})")
            
            # Show early progress for first few batches worth of questions
            if (i + 1) <= batch_size * 3 and (i + 1) % batch_size == 0:
                current_accuracy = correct / (i + 1)
                print(f"🚀 Early progress: {i+1}/{total} questions | Accuracy so far: {current_accuracy:.2%}")
        
        # Major milestone updates (shown for both debug and non-debug)
        if total >= 1000 and (i + 1) % milestone_interval == 0:
            current_accuracy = correct / (i + 1)
            print(f"🎯 Milestone: {i+1:,} questions completed | Current accuracy: {current_accuracy:.2%}")
    
    accuracy = correct / total if total > 0 else 0
    
    # Final results
    print(f"\n✅ Batch evaluation completed!")
    print(f"📊 Final Results: {correct:,}/{total:,} correct ({accuracy:.2%})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }



def get_eval_questions_by_chapter(benchmark, max_chapter: int, chapter_column: str = 'correct_answer_chapters'):
    """
    Extract questions from benchmark where all answer chapters are <= max_chapter.
    
    Args:
        benchmark: Object with get_df_qa() method
        max_chapter: Maximum chapter number to consider
        chapter_column: Column name containing list of chapters for correct answers
    
    Returns:
        Filtered DataFrame containing only questions answerable up to max_chapter
    """
    df = benchmark.get_df_qa()
    
    # Filter: keep rows where all chapters in the array are <= max_chapter
    # Always keep empty arrays (hallucinating questions)
    def filter_chapters(chapters):
        if len(chapters) == 0:
            return max_chapter == benchmark.nb_chapters()  # Keep empty only if max_chapter is total chapters
        # Check if all chapters are <= max_chapter
        return (chapters <= max_chapter).all()  # Keep empty arrays, only if last chapter (hallucination questions)
      
            
    mask = df[chapter_column].apply(filter_chapters)
    
    return df[mask].reset_index(drop=True)


