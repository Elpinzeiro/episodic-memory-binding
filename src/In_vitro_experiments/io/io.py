import json
from pathlib import Path
import jsonlines

def export_list(data_list, filename):
    '''
    Dumping a list into a json file
    '''
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

def import_list(filename):
    '''
    Loading a json file to a list
    '''
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def export_jsonl(res, filename):
    '''
    Dumping into a jsonl file (for fine-tuning)
    '''
    with jsonlines.open(filename, 'w') as writer:
        writer.write_all(res)
        
def import_jsonl(filepath):
    """Import JSONL file into a list of dictionaries"""
    import json
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# ... resto del codice che usa import_jsonl
def data_folder_experiment_func(prompt_parameters):
    '''
    Folder path of the experiment
    '''
    return f"U{prompt_parameters['name_universe']}_S{prompt_parameters['name_styles']}_seed{prompt_parameters['seed']}"

def paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters):
    '''
    File name of the generated paragraph
    '''
    prompt_str = f"e{event_index}_iter{iteration}"
    model_str = f"model_{model_parameters['model_name']}"
    return f"{model_str}_{prompt_str}.json"

def paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the generated paragraph
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return data_paragraphs_filepath

def paragraph_verif_direct_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the direct verification log for the target paragraph
    '''
    verifdirect_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs_verif_direct" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return verifdirect_paragraphs_filepath

def paragraph_verif_llm_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the LLM verification log for the target paragraph
    '''
    verifllm_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs_verif_llm" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return verifllm_paragraphs_filepath

def book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters):
    '''
    File name of the generated book
    '''
    indexing_str = f"I{book_parameters['indexing']}"
    model_str = f"model_{model_parameters['model_name']}_itermax_{model_parameters['itermax']}"
    this_book_str = f"nbchapters_{nb_chapters}_nbtokens_{nb_tokens}"
    return f"{model_str}_{indexing_str}_{this_book_str}"

def book_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters):
    '''
    File path of the generated book
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "books" / book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters)
    return data_paragraphs_filepath

def answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    Full directory path of the answers
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters) / answer_dir_name_func(answering_parameters)
    return data_paragraphs_filepath

def answer_dirpath_old_chapter_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters,chapter,epochs_str):
    '''
    Full directory path of the answers
    '''
    model_str = answering_parameters['model_name']
    kind_str = answering_parameters['kind']
    ftuning_policy = answering_parameters['ftuning_policy']

    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters) / answer_dir_name_old_chapter_func(chapter,epochs_str,model_str,kind_str,ftuning_policy)
    return data_paragraphs_filepath

def answer_dir_name_old_chapter_func(chapter,epochs_str,model_str,kind_str,ftuning_policy):

    return f"answered_by_{kind_str}_{ftuning_policy}_{model_str}_{epochs_str}_{chapter}"

def answer_dir_name_func(answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096}):
    '''
    Final folder path of the answers
    '''
    if answering_parameters['kind'] == 'prompting':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        return f"answered_by_{kind_str}_{model_str}"
    elif answering_parameters['kind'] == 'rag':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        chunk_str = f"chunk_{answering_parameters['embedding_chunk']}_top{answering_parameters['top_n']}"
        return f"answered_by_{kind_str}_{model_str}_{chunk_str}"
    elif answering_parameters['kind'] == 'ftuning':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        ftuning_str = f"fpolicy_{answering_parameters['ftuning_input_data_policy']}"
        return f"answered_by_{kind_str}_{model_str}_{ftuning_str}"
    elif answering_parameters['kind'] == 'ftuning-manual':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        epochs_str = f"epochs_{answering_parameters['n_epochs']}"
        max_chapter = f"max_chapter_{answering_parameters['max_chapter']}" if 'max_chapter' in answering_parameters else "no_max_chapter"
        ftuning_policy = f"ftuning_policy_{answering_parameters['ftuning_policy']}" if 'ftuning_policy' in answering_parameters else "no_episodic_ftuning_policy"
        ftuning_input_policy = f'ftuning_input_data_policy_{answering_parameters['ftuning_input_data_policy']}'  
        if 'recollecting_approach' in answering_parameters:
            approach = answering_parameters['recollecting_approach']
            if approach == 'generic_and_magic':
                recollecting_approach = "recalling_with_generic_and_magic"
            else:
                cleaning_suffix = "_cleaned" if answering_parameters.get('clean_recall_from_hall', False) else ""
                recollecting_approach = f"recalling_with_{approach}{cleaning_suffix}"
        else:
            recollecting_approach = "no_recollecting_approach"       
        eval_policy = f"eval_with_{answering_parameters['evaluation']}" if 'evaluation' in answering_parameters else "no_eval"
        batch_size = f"bs_{answering_parameters['batch_size']}" if 'batch_size' in answering_parameters else "no_bs"
        lr = f"lr_{answering_parameters['learning_rate_multiplier']}" if 'learning_rate_multiplier' in answering_parameters else "no_lr"
        if answering_parameters['batch_size'] == 'auto' and answering_parameters['learning_rate_multiplier'] == 'auto':
            return f"answered_by_{kind_str}_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_{model_str}_{epochs_str}_{max_chapter}_{recollecting_approach}"
        else:
            return f"answered_by_GridSearch_{kind_str}_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_{model_str}_{batch_size}_{lr}_{epochs_str}_{max_chapter}_{recollecting_approach}"
        
# def path_export_fintuning(data_folder,prompt_parameters, answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096}):
#     if answering_parameters['kind'] == 'ftuning-manual':
#         model_str = f"model_{answering_parameters['model_name']}"
#         max_chapter = f"max_chapter_{answering_parameters['max_chapter']}" if 'max_chapter' in answering_parameters else "no_max_chapter"
#         epochs_str = f"epochs_{answering_parameters['n_epochs']}"
#         ftuning_policy = f"ftuning_policy_{answering_parameters['ftuning_policy']}" if 'ftuning_policy' in answering_parameters else "no_episodic_ftuning_policy"
#         ftuning_input_policy = f'ftuning_input_data_policy_{answering_parameters['ftuning_input_data_policy']}'  
        
#         if 'recollecting_approach' in answering_parameters:
#             approach = answering_parameters['recollecting_approach']
#             cleaning_suffix = "_cleaned" if answering_parameters.get('clean_recall_from_hall', False) else ""
#             recall_method = f"recalling_method_{approach}{cleaning_suffix}"
#         else:
#             recall_method = "recalling_method_no_recollecting_approach"
        
#         eval_policy = f"eval_with_{answering_parameters['evaluation']}" if 'evaluation' in answering_parameters else "no_eval"
#         batch_size = f"bs_{answering_parameters['batch_size']}" if 'batch_size' in answering_parameters else "no_bs"
#         lr = f"lr_{answering_parameters['learning_rate_multiplier']}" if 'learning_rate_multiplier' in answering_parameters else "no_lr"
#         if answering_parameters['batch_size'] == 'auto' and answering_parameters['learning_rate_multiplier'] == 'auto':
#             return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_{model_str}_{max_chapter}_{recall_method}" 
#         else:
#             return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_GridSearch_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_Book_Chapter_{prompt_parameters['nb_events']}_{model_str}_{batch_size}_{lr}_{epochs_str}_{recall_method}_{max_chapter}"
def path_export_fintuning(data_folder, prompt_parameters, answering_parameters={'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096}):
    if answering_parameters['kind'] == 'ftuning-manual':
        model_str = f"model_{answering_parameters['model_name']}"
        max_chapter = f"max_chapter_{answering_parameters['max_chapter']}" if 'max_chapter' in answering_parameters else "no_max_chapter"
        epochs_str = f"epochs_{answering_parameters['n_epochs']}"
        ftuning_policy = f"ftuning_policy_{answering_parameters['ftuning_policy']}" if 'ftuning_policy' in answering_parameters else "no_episodic_ftuning_policy"
        ftuning_input_policy = f'ftuning_input_data_policy_{answering_parameters["ftuning_input_data_policy"]}'  
        
        # Only add ftuning_type and ewc_lambda if using EWC
        if answering_parameters.get('ftuning_type') == 'ewc':
            ftuning_type = "ftuning_type_ewc"
            if 'ewc_lambda' in answering_parameters:
                ewc_lambda_str = f"ewc_lambda_{answering_parameters['ewc_lambda']}"
            else:
                ewc_lambda_str = "ewc_lambda_5000"  # default
        else:
            ftuning_type = ""
            ewc_lambda_str = ""
        
        if 'recollecting_approach' in answering_parameters:
            approach = answering_parameters['recollecting_approach']
            cleaning_suffix = "_cleaned" if answering_parameters.get('clean_recall_from_hall', False) else ""
            recall_method = f"recalling_method_{approach}{cleaning_suffix}"
        else:
            recall_method = "recalling_method_no_recollecting_approach"
        
        eval_policy = f"eval_with_{answering_parameters['evaluation']}" if 'evaluation' in answering_parameters else "no_eval"
        batch_size = f"bs_{answering_parameters['batch_size']}" if 'batch_size' in answering_parameters else "no_bs"
        lr = f"lr_{answering_parameters['learning_rate_multiplier']}" if 'learning_rate_multiplier' in answering_parameters else "no_lr"
        
        if answering_parameters['batch_size'] == 'auto' and answering_parameters['learning_rate_multiplier'] == 'auto':
            # Build path - only add EWC parts if using EWC
            if ftuning_type:  # EWC case
                return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_{model_str}_{ftuning_type}_{ewc_lambda_str}_{max_chapter}_{recall_method}" 
            else:  # lora/full case (unchanged)
                return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_{model_str}_{max_chapter}_{recall_method}" 
        else:
            # Build path with GridSearch - only add EWC parts if using EWC
            if ftuning_type:  # EWC case
                return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_GridSearch_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_Book_Chapter_{prompt_parameters['nb_events']}_{model_str}_{ftuning_type}_{ewc_lambda_str}_{batch_size}_{lr}_{epochs_str}_{recall_method}_{max_chapter}"
            else:  # lora/full case (unchanged)
                return Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / f"checkpoint_GridSearch_{ftuning_policy}_{ftuning_input_policy}_{eval_policy}_Book_Chapter_{prompt_parameters['nb_events']}_{model_str}_{batch_size}_{lr}_{epochs_str}_{recall_method}_{max_chapter}"
def answer_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    # Use different subfolder based on ftuning_type
    if answering_parameters.get('ftuning_type') == 'full':
        answers_folder = 'raw_answers_full_finetuned'
    else:
        answers_folder = 'raw_answers'  # Default for LoRA and backward compatibility
    
    filename = f"{str(q)}.json"
    answer_filepath = my_dirpath / answers_folder / filename
    return answer_filepath

def answer_recollection_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters, file_type='answer'):    
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    if file_type == 'question':
        filename = f"q_{str(q)}.json"
    else:
        filename = f"{str(q)}.json"
    
    
    approach = answering_parameters['recollecting_approach']
    if approach == 'generic_and_magic':
        # After renaming, generic_and_magic files will be in the main directory's evaluated_answers
        # not in a separate recollection subdirectory
        return my_dirpath / 'evaluated_answers' / filename
    else:
        recoll_approach = approach
    
    
    if answering_parameters.get('ftuning_policy') == 'episodic_chapter_perfect_recoll':
        answer_filepath = my_dirpath / f'recollection_{recoll_approach}_answers' / filename
    else:
        answer_filepath = my_dirpath / filename
        
    return answer_filepath

def finetuning_question_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json fine-tuning question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    finetuning_question_filepath = my_dirpath / 'original_finetuning_questions_to_eval' / filename
    return finetuning_question_filepath
def answer_reasoning_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json reasoning from the question with index q (only for reasoning models, e.g. DeepSeek)
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    answer_filepath = my_dirpath / 'raw_reasoning' / filename
    return answer_filepath

def evaluate_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json evaluation of the answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    # Use different subfolder based on ftuning_type
    if answering_parameters.get('ftuning_type') == 'full':
        answers_folder = 'evaluated_answers_full_finetuned'
    else:
        answers_folder = 'evaluated_answers'  # Default for LoRA and backward compatibility
    
    filename = f"{str(q)}.json"
    evaluate_filepath = my_dirpath / answers_folder / filename
    return evaluate_filepath

def chronological_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json chronological of the answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    # Use different subfolder based on ftuning_type
    if answering_parameters.get('ftuning_type') == 'full':
        answers_folder = 'chronological_answers_full_finetuned'
    else:
        answers_folder = 'chronological_answers'  # Default for LoRA and backward compatibility
    
    filename = f"{str(q)}.json"
    chronological_filepath = my_dirpath / answers_folder / filename
    return chronological_filepath


def ftuning_input_filepath_func(ftuning_input_data_policy, nb_tokens, answering_parameters, nb_chapters, data_folder, prompt_parameters, model_parameters, book_parameters):
    '''
    Calculates the full file path for the fine-tuning input JSONL file
    and the directory path for the LoRA weights.
    This function consolidates the path construction based on your existing naming conventions.

    Args:
        ftuning_input_data_policy (str): The policy for the fine-tuning input data.
        nb_tokens (int): The number of tokens.
        answering_parameters (dict): Dictionary containing parameters for answering,
                                     including 'model_name' and 'n_epochs'.
        nb_chapters (int): Number of chapters for the book.
        data_folder (str): Base data folder path.
        prompt_parameters (dict): Dictionary with prompt-related parameters.
        model_parameters (dict): Dictionary with model-related parameters.
        book_parameters (dict): Dictionary with book-related parameters.

    Returns:
        tuple: A tuple containing:
            - pathlib.Path: The complete file path to the fine-tuning input JSONL file.
            - pathlib.Path: The directory path for the LoRA weights.
    '''
    # Construct the JSONL filename
    recollecting_approach = answering_parameters.get('recollecting_approach', 'no_recollecting_approach')
    jsonl_filename = f"ftuning_{ftuning_input_data_policy}_{nb_tokens}_recall_{recollecting_approach}.jsonl"

    # Construct the LoRA weights folder name
    lora_weights_foldername = f"ftuning_lora_weights_{answering_parameters['model_name']}_n_epochs_{answering_parameters['n_epochs']}"

    # Get the base answer directory path using the existing function (answer_dirpath_func)
    answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    # Combine to get the full path to the LoRA weights folder
    lora_weights_pathname = answer_dirpath / lora_weights_foldername
    
    # Combine to get the full fine-tuning input file path
    ftuning_input_filepath = lora_weights_pathname / jsonl_filename
    
    return ftuning_input_filepath, lora_weights_pathname


def commonsense_path(data_folder):
    commonsense_path = data_folder / 'commonsense'
    return commonsense_path