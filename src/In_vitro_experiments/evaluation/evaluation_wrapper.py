from In_vitro_experiments.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from In_vitro_experiments.evaluation.generator_answers_1_prompting import ask_recoll_questions_smart_llm,ask_recoll_questions,commonsense_test,generate_answers_func, generate_evaluation_func, generate_chronological_func
from In_vitro_experiments.evaluation.scoring_answers import update_policy_of_evaluation_to
import numpy as np
from In_vitro_experiments.generation.qa_generation import build_single_chapter_qa_func
# for rag
from In_vitro_experiments.generation.generate_3_secondary_entities import count_tokens
from In_vitro_experiments.evaluation.generator_answers_2_rag import embed_chunks
from In_vitro_experiments.generation.prompts import  process_recollection_with_hallucination_removal, generate_recollection_questions_prompt_generic_with_context_1,generate_recollection_questions_prompt_generic_with_context_2,generate_qa_merge_prompt,generate_recollection_questions_prompt_generic,create_recollection_questions,generating_12_recoll_questions, generating_anti_hallucination_questions, generating_synthetic_rephrasing_prompt,generating_question_chapter_prompt_1
from In_vitro_experiments.io.io import path_export_fintuning,answer_dirpath_func, import_list, export_list,import_jsonl, export_jsonl, ftuning_input_filepath_func
import pandas as pd
import ast
import json
import os
# for ftuning
from In_vitro_experiments.evaluation.generator_answers_3_ftuning import (    
    upload_ftuning_input, retrieve_fileid, setup_and_resume_full_finetuning,
    setup_tensorboard_logging, setup_and_resume_lora_training, cleanup_memory,setup_and_resume_ewc_training
)
from openai import OpenAI
from In_vitro_experiments.models.settings_wrapper import SettingsWrapper
from In_vitro_experiments.models.models_wrapper import ModelsWrapper
from scipy.stats import kendalltau
from datasets import Dataset
from pathlib import Path

class EvaluationWrapper:
    def __init__(
            self,
            my_benchmark: BenchmarkGenerationWrapper,
            answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15, 'policy': 'original'},
            data_folder = 'data/narratives',
            env_file = '.env',
            judge_model_name = None,
            ftuning_input_if_custom = None):
        
        # save the input
        self.my_benchmark = my_benchmark
        self.data_folder = data_folder
        self.env_file = env_file
        self.policy = answering_parameters['policy']
        debug = answering_parameters.get('debug', False)
        # Define evaluation parameters (judge model for evaluation)
        self.evaluation_parameters = {
                    'kind': 'prompting',
                    'model_name': judge_model_name,
                    'max_new_tokens': 2048  
                }
        # generated answers
        if answering_parameters['kind'] == 'prompting':
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file)
        elif answering_parameters['kind'] == 'rag':

            self.my_chunks = my_benchmark.chunk_book(split = answering_parameters['embedding_chunk'])
            self.chunk_with_max_tokens = max([count_tokens(x) for x in self.my_chunks])
            self.chunk_number = len(self.my_chunks)

            prompt_parameters = my_benchmark.prompt_parameters
            model_parameters = my_benchmark.model_parameters
            book_parameters = my_benchmark.book_parameters
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            embedding_filepath = answer_dirpath / "embedding.csv"
            if not embedding_filepath.is_file():
                my_embedding = embed_chunks(self.my_chunks, answering_parameters, env_file)
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                my_embedding.to_csv(embedding_filepath, index=False)
                print(f"chunked into {self.chunk_number} chunks, the largest containing {self.chunk_with_max_tokens} tokens")
                print(my_embedding)
            self.my_embedding = pd.read_csv(embedding_filepath)
            self.my_embedding['embedding'] = self.my_embedding['embedding'].apply(ast.literal_eval)

            
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file, self.my_embedding)
        elif answering_parameters['kind'] == 'ftuning':

            
            # 1. build jsonl
            prompt_parameters = my_benchmark.prompt_parameters
            model_parameters = my_benchmark.model_parameters
            book_parameters = my_benchmark.book_parameters
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            ftuning_input_data_policy = answering_parameters['ftuning_input_data_policy']
            if ftuning_input_data_policy == 'single':
                answer_in_one_chapter_only = True # only select questions with single answers
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)
            elif ftuning_input_data_policy == 'all':
                answer_in_one_chapter_only = False # overfit on all possible questions
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            jsonl_filename = f"ftuning_{ftuning_input_data_policy}_{nb_tokens}.jsonl"
            ftuning_input_filepath = answer_dirpath / jsonl_filename
            if not ftuning_input_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                export_jsonl(ftuning_input, ftuning_input_filepath)
            self.ftuning_input = ftuning_input
            config = SettingsWrapper(_env_file = env_file)

            # 2. uploading
            if answering_parameters['ftuning_need_upload']:
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                upload_ftuning_input(self.client, ftuning_input_filepath)

            # 3. get the corresponding created fileid
            ftuning_input_id_filepath = answer_dirpath / f"ftuning_{ftuning_input_data_policy}.id"
            if not ftuning_input_id_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                ftuning_input_id = retrieve_fileid(self.client, jsonl_filename)
                export_list(ftuning_input_id, ftuning_input_id_filepath)
                print(f"json `{jsonl_filename}` with file id `{self.ftuning_input_id}` for the `{ftuning_input_data_policy}` policy")
            self.ftuning_input_id = import_list(ftuning_input_id_filepath)
            
            # 4. create a fine tuning job
            if answering_parameters['ftuning_need_actual_tune']:
                # [client]
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                # [\client]
                self.client.fine_tuning.jobs.create(
                    training_file=self.ftuning_input_id,
                    model=answering_parameters['model_name'],
                    hyperparameters={
                    "batch_size": answering_parameters['batch_size'],
                    "learning_rate_multiplier": answering_parameters['learning_rate_multiplier'],
                    "n_epochs": answering_parameters['n_epochs']

                    }
                )
                print('ongoing jobs')
                print(self.client.fine_tuning.jobs.list(limit=10))
                print('for cancelling, please use `object.cancel_job(corresponding_ftjob_as_above)`')

            # 5. answer the questions
            answering_parameters['model_name'] = answering_parameters['fine_tuned_model_name']
            #print(answering_parameters['model_name'])
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file)

        elif answering_parameters['kind'] == 'ftuning-manual':
            
            # 1. build jsonl
            prompt_parameters = my_benchmark.prompt_parameters
            model_parameters = my_benchmark.model_parameters
            book_parameters = my_benchmark.book_parameters
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            ftuning_input_data_policy = answering_parameters['ftuning_input_data_policy']
            if 'max_chapter' in answering_parameters and answering_parameters['max_chapter'] is not None:
                max_chapter = answering_parameters['max_chapter']
            
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            

            answer_path = answer_dirpath_func(nb_chapters=nb_chapters,nb_tokens=nb_tokens,data_folder=data_folder,prompt_parameters=prompt_parameters, book_parameters=book_parameters,model_parameters=model_parameters,answering_parameters=answering_parameters)
            answer_path = Path(answer_path)
            answer_path.mkdir(parents=True, exist_ok=True)
            ftuning_input_filepath, lora_weights_pathname = ftuning_input_filepath_func(ftuning_input_data_policy,nb_tokens,answering_parameters,nb_chapters,data_folder,prompt_parameters,model_parameters,book_parameters)      
                
            if 'ftuning_policy' in answering_parameters:
                ftuning_policy = answering_parameters['ftuning_policy']
                
                if ftuning_policy == 'llm_qa':
                    if ftuning_input_filepath.is_file():
                        print(f"Fine-tuning file already exists: {ftuning_input_filepath}")
                        # Carica il file esistente invece di rigenerare
                        ftuning_input = import_jsonl(ftuning_input_filepath)  # Devi implementare questa funzione se non esiste
                    else:   
                        # Use LLM generated questions       
                        print(f"DEBUG: RETURN GET_CHAPTER = {my_benchmark.get_chapter(max_chapter)}")             
                        user_prompt = generating_question_chapter_prompt_1(my_benchmark.get_chapter(max_chapter))                    
                        qa_response = my_benchmark.llm_questions_generated(user_prompt)
                        self.llm_generated_questions = my_benchmark.parse_llm_response_to_dataframe(qa_response)
                        
                        if max_chapter > 1 and 'recollecting_approach' in answering_parameters:
                            # Check if recollection answers folder exists and contains files
                            # Before the recollection logic, modify the path
                            if answering_parameters['recollecting_approach'] == 'magic_12':
                                recollection_questions = generating_12_recoll_questions(my_benchmark,max_chapter)
                                recollection_q_and_a = ask_recoll_questions(recollection_questions, my_benchmark, answer_path, answering_parameters, data_folder, env_file, debug)
                                print("RECALL: MAGIC 12")
                                
                            elif answering_parameters['recollecting_approach'] == 'generic_qa':
                                recollection_questions = create_recollection_questions(my_benchmark,max_chapter)
                                recollection_q_and_a = ask_recoll_questions(recollection_questions, my_benchmark, answer_path, answering_parameters, data_folder, env_file, debug)
                                print("RECALL: 4_GENERIC")                                
                            

                            elif answering_parameters['recollecting_approach'] == 'prompt_with_context':

                                doc_context = 'The Excerpt is from the "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition" fictional book'
                                recoll_prompt = generate_recollection_questions_prompt_generic_with_context_1(my_benchmark.get_chapter(max_chapter),doc_context)                    
                                recollection_questions = my_benchmark.llm_questions_generated(recoll_prompt)
                                data = json.loads(recollection_questions)
                                recollection_questions = [item["question"] for item in data["questions"]]
                                recollection_q_and_a = ask_recoll_questions(recollection_questions, my_benchmark, answer_path, answering_parameters, data_folder, env_file, debug)
                                print("RECALL: PROMPT_WITH_CONTEXT_BASED")

                            else: 

                                doc_context = 'The Excerpt is from the "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition" fictional book'
                                recoll_prompt = generate_recollection_questions_prompt_generic_with_context_2(my_benchmark.get_chapter(max_chapter),doc_context)                    
                                recollection_questions = my_benchmark.llm_questions_generated(recoll_prompt)
                                data = json.loads(recollection_questions)
                                recollection_questions = [item["question"] for item in data["questions"]]
                                recollection_q_and_a = ask_recoll_questions(recollection_questions, my_benchmark, answer_path, answering_parameters, data_folder, env_file, debug)
                                print("RECALL: PROMPT_WITH_CONTEXT_BASED")
                            
                            if answering_parameters.get('clean_recall_from_hall', False):

                                recollection_q_and_a = process_recollection_with_hallucination_removal(
                                    recollection_q_and_a, 
                                    my_benchmark, 
                                    max_chapter
                                )
                                recollection_q_and_a = recollection_q_and_a.drop('correct_answer', axis=1).rename(columns={'cleaned_answer': 'correct_answer'})

                            context = "The Q&A are from the \"Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition\" fictional book and we want to build a Q&A dataset that will be used for finetuning purpose."
                            merge_prompt = generate_qa_merge_prompt(self.llm_generated_questions,recollection_q_and_a,context=context)
                            merged_qa = my_benchmark.llm_questions_generated(merge_prompt)                            
                            self.llm_generated_questions = my_benchmark.parse_llm_response_to_dataframe(merged_qa)
                            llm_questions_filepath = answer_dirpath / f"after_merge.csv"
                            if not llm_questions_filepath.exists():
                                llm_questions_filepath.parent.mkdir(parents=True, exist_ok=True)
                                self.llm_generated_questions.to_csv(llm_questions_filepath, index=False)
                                print(f"Exported LLM after Merge to {llm_questions_filepath}")

                        #ADD ANTI-HALLUCINATION QUESTIONS            
                        user_prompt_no_hallucinations = generating_anti_hallucination_questions(my_benchmark.get_chapter(max_chapter))
                        qa_response_no_hall = my_benchmark.llm_questions_generated(user_prompt_no_hallucinations)
                        no_hallucinations_qa = my_benchmark.parse_llm_response_to_dataframe(qa_response_no_hall)

                        
                        self.llm_generated_questions = pd.concat([self.llm_generated_questions, no_hallucinations_qa], ignore_index=True)

                        my_benchmark.llm_generated_questions = self.llm_generated_questions
                        
                        llm_questions_filepath = answer_dirpath / f"llm_generated_questions.csv"
                        if not llm_questions_filepath.exists() and self.llm_generated_questions is not None:
                            llm_questions_filepath.parent.mkdir(parents=True, exist_ok=True)
                            self.llm_generated_questions.to_csv(llm_questions_filepath, index=False)
                            ftuning_input = my_benchmark.build_fine_tuning_jsonl(self.llm_generated_questions)
                            print(f"Exported LLM generated questions to {llm_questions_filepath}")

                    
                elif ftuning_policy == 'llm_rephrasing':
                    if ftuning_input_filepath.is_file():
                        print(f"Fine-tuning file already exists: {ftuning_input_filepath}")
                        ftuning_input = import_jsonl(ftuning_input_filepath)
                    else:   
                        # Use LLM generated synthetic texts for next-word prediction
                        user_prompt = generating_synthetic_rephrasing_prompt(my_benchmark.get_chapter(max_chapter))                    
                        synthetic_response = my_benchmark.llm_synthetic_text_generated(user_prompt)
                        synthetic_texts = my_benchmark.parse_synthetic_text_response(synthetic_response)
                        
                        # Create plain text training data (no Q&A conversion)
                        ftuning_input = [{"text": text} for text in synthetic_texts]
                        
                        # Save synthetic texts for reference
                        synthetic_texts_filepath = answer_dirpath / "llm_generated_synthetic_texts.json"
                        if not synthetic_texts_filepath.exists():
                            synthetic_texts_filepath.parent.mkdir(parents=True, exist_ok=True)
                            with open(synthetic_texts_filepath, 'w') as f:
                                json.dump(synthetic_texts, f, indent=2)
                            print(f"Exported LLM generated synthetic texts to {synthetic_texts_filepath}")


                elif ftuning_policy == 'episodic_chapter':
                    # Use episodic (chapter-filtered) questions
                    single_chapter_questions = my_benchmark.finetuning_questions_one_chapter.copy()
                    # Filter rows where all chapters in correct_answer_chapters are <= max_chapter
                    single_chapter_questions_filtered_by_chapter = single_chapter_questions[
                        single_chapter_questions['correct_answer_chapters'].apply(
                            lambda x: len(x) > 0 and all(chapter == max_chapter for chapter in x)
                        )
                    ]
                    if len(single_chapter_questions_filtered_by_chapter) < 5:
                        single_chapter_questions_filtered_by_chapter = build_single_chapter_qa_func(my_benchmark.df_book_groundtruth, my_benchmark.events, 100)
                    
                    ftuning_input = my_benchmark.build_fine_tuning_jsonl(finetuning_questions_df=single_chapter_questions_filtered_by_chapter)

                elif ftuning_policy == 'episodic_chapter_perfect_recoll':
                    
                    # Use episodic (chapter-filtered) questions
                    single_chapter_questions = my_benchmark.finetuning_questions_one_chapter.copy()
                    # Filter rows where all chapters in correct_answer_chapters are <= max_chapter
                    single_chapter_questions_filtered_by_chapter = single_chapter_questions[
                        single_chapter_questions['correct_answer_chapters'].apply(
                            lambda x: len(x) > 0 and all(chapter == max_chapter for chapter in x)
                        )
                    ]
                    if len(single_chapter_questions_filtered_by_chapter) < 5:
                        single_chapter_questions_filtered_by_chapter = build_single_chapter_qa_func(my_benchmark.df_book_groundtruth, my_benchmark.events, max_chapter)
                    
                    invalid_chapters = set(idx + 1 for idx in my_benchmark.invalid_debug_event_idx_func())

                    if max_chapter > 1 and (max_chapter - 1) not in invalid_chapters:
                        
                        recollecting_approach = answering_parameters.get('recollecting_approach')  # returns None if key missing
                        
                        if recollecting_approach == 'generic_qa':
    
                            recollection_questions = create_recollection_questions(my_benchmark,max_chapter)
                            recollection_q_and_a = ask_recoll_questions_smart_llm(
                                recollection_questions, 
                                my_benchmark,
                                'generic_qa',  
                                answer_path,                                 
                                answering_parameters, 
                                data_folder, 
                                env_file, 
                                debug=False,
                                max_chapter=max_chapter -1,
                                    
                            )
                            combined_df = pd.concat([single_chapter_questions_filtered_by_chapter, recollection_q_and_a], 
                            ignore_index=True, 
                            sort=False)
                            ftuning_input = my_benchmark.build_fine_tuning_jsonl(finetuning_questions_df=combined_df)
                        
                        elif recollecting_approach == 'generic_and_magic':
                            #GENERIC QA    
                            recollection_questions = create_recollection_questions(my_benchmark,max_chapter)
                            recollection_q_and_a = ask_recoll_questions_smart_llm(
                                recollection_questions, 
                                my_benchmark,
                                'generic_qa',  
                                answer_path, 
                                answering_parameters, 
                                data_folder, 
                                env_file, 
                                debug=False,
                                max_chapter=max_chapter -1,
                                    
                            )
                            combined_df = pd.concat([single_chapter_questions_filtered_by_chapter, recollection_q_and_a],ignore_index=True,sort=False)
                            print(f"DEBUG AFTER GENERIC_QA: COMBINED_DF LENGTH={len(combined_df)}")
                            # MAGIC 12
                            recollection_questions = generating_12_recoll_questions(my_benchmark,max_chapter)
                            recollection_q_and_a = ask_recoll_questions_smart_llm(
                                recollection_questions, 
                                my_benchmark,
                                'magic_12', 
                                answer_path, 
                                answering_parameters, 
                                data_folder, 
                                env_file, 
                                debug=False,
                                max_chapter=max_chapter -1
                                
                            )
                            combined_df = pd.concat([combined_df, recollection_q_and_a],ignore_index=True,sort=False)
                            print(f"DEBUG AFTER MAGIC12: COMBINED_DF LENGTH={len(combined_df)}")
                            ftuning_input = my_benchmark.build_fine_tuning_jsonl(finetuning_questions_df=combined_df)

                    else:
                        
                        ftuning_input = my_benchmark.build_fine_tuning_jsonl(finetuning_questions_df=single_chapter_questions_filtered_by_chapter)


            else:
                # Old behavior when ftuning_policy is not specified
                if ftuning_input_data_policy == 'single':
                    ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only=True)
                elif ftuning_input_data_policy == 'all':
                    ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only=False)
                elif ftuning_input_data_policy == 'custom':
                    ftuning_input = ftuning_input_if_custom
            
           
            if not ftuning_input_filepath.is_file():
                ftuning_input_filepath.parent.mkdir(parents=True, exist_ok=True)
                export_jsonl(ftuning_input, ftuning_input_filepath)

            self.ftuning_input = ftuning_input 
            config = SettingsWrapper(_env_file = env_file)
            input_dataset = Dataset.from_list(ftuning_input)
            print(f"DEBUGGING!!!{len(input_dataset)}")
            shared_output_dir = Path(path_export_fintuning(data_folder,prompt_parameters,answering_parameters=answering_parameters))
            print(f"DEBUG SHARED OUTPUT DIR:{shared_output_dir}")
            tensorboard_log_dir = setup_tensorboard_logging(shared_output_dir)
            if answering_parameters['ftuning_type'] == 'full':
                res = setup_and_resume_full_finetuning(
                    model_name=answering_parameters['model_name'],
                output_dir=shared_output_dir,
                input_dataset=input_dataset,                    # Raw dataset - function will tokenize
                answering_parameters=answering_parameters,
                tensorboard_log_dir=tensorboard_log_dir,
                answer_path=answer_path,
                checkpoint_to_continue=answering_parameters.get('checkpoint_to_continue',None)
                )
            elif answering_parameters['ftuning_type'] == 'lora':
                res = setup_and_resume_lora_training(
                model_name=answering_parameters['model_name'],
                output_dir=shared_output_dir,
                input_dataset=input_dataset,                    # Raw dataset - function will tokenize
                answering_parameters=answering_parameters,
                tensorboard_log_dir=tensorboard_log_dir,
                answer_path=answer_path,
                checkpoint_to_continue=answering_parameters.get('checkpoint_to_continue',None)

            )
            elif answering_parameters['ftuning_type'] == 'ewc':
                res = setup_and_resume_ewc_training(
                    model_name=answering_parameters['model_name'],
                    output_dir=shared_output_dir,
                    input_dataset=input_dataset,                    # Raw dataset - function will tokenize
                    answering_parameters=answering_parameters,
                    tensorboard_log_dir=tensorboard_log_dir,
                    answer_path=answer_path,
                    checkpoint_to_continue=answering_parameters.get('checkpoint_to_continue', None)
                )
            else:
                raise ValueError(f"Invalid ftuning_type: {answering_parameters['ftuning_type']}. Must be 'full' or 'lora'")
            
            print(res)
            if isinstance(res, dict) and res.get("status") == "error":
                # stop here: answering with a model that was not trained fails later with an unrelated error
                raise RuntimeError(f"Fine-tuning failed, so the run stops before answering: {res.get('error_message')}")

            # Memory cleanup
            cleanup_memory()

            # Answer the questions
            answering_parameters['after_finetuning_path'] = answer_path  
             # FIXED: Use lora_path instead of lora_weights_pathname
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file, debug)
            print(f"After Generation: {len(self.df_generated_answers)}")           
        else:
            raise ValueError('unknown "kind", should be "prompting", "rag" or "ftuning"')
        skip_eval = answering_parameters.get('skip_eval', False)

        if not skip_eval and len(self.df_generated_answers) != 0:    
            # # generated evaluation (given answers)
            
            self.df_generated_evaluations = generate_evaluation_func(my_benchmark, self.df_generated_answers, answering_parameters, data_folder, env_file, self.evaluation_parameters,debug,
                                                                    only_scored=answering_parameters.get('judge_only_scored', True))

            # # generated chronological (given evaluation); only when chronological questions were judged
            self.df_generated_chronological, self.kendall_summaries_for_this_experiment = None, None
            if (self.df_generated_evaluations['get'] == 'chronological').any():
                df_generated_chronological = generate_chronological_func(my_benchmark, self.df_generated_evaluations, answering_parameters, data_folder, env_file, self.evaluation_parameters,debug)
                
                self.df_generated_chronological = df_generated_chronological
                
                self.kendall_summaries_for_this_experiment = self.compute_kendall_summarise(df_generated_chronological, verbose = False)
        elif skip_eval:
            print("⏭️  Skipping evaluation (skip_eval=True)")

        if '1_commonsense_test_per_book' in answering_parameters and  answering_parameters['1_commonsense_test_per_book'] is True:
            if answering_parameters['max_chapter'] == my_benchmark.nb_chapters():
                self.commonsense_summary = commonsense_test(
                    my_benchmark=my_benchmark, 
                    answering_parameters=answering_parameters, 
                    data_folder=data_folder,             
                    max_questions=answering_parameters['max_questions_commonsense'],
                    output_dir=shared_output_dir, 
                    debug=debug
                ) 
        else:
            self.commonsense_summary = commonsense_test(
                my_benchmark=my_benchmark, 
                answering_parameters=answering_parameters, 
                data_folder=data_folder,             
                max_questions=answering_parameters['max_questions_commonsense'],
                output_dir=shared_output_dir, 
                debug=debug
            )  
        cleanup_memory()




    def cancel_job(self, ftjob_id = 'ftjob-wjldwdkjw0eiw'):
        self.client.fine_tuning.jobs.cancel(ftjob_id)
        print("cancelled")

    def get_pretty_summary_relative_to(self, my_column = 'q_idx', metric = 'f1_score_lenient', sorting = False, filter_dict = {}):
        df = self.df_generated_evaluations

        if 'bins_items_correct_answer' in my_column:
            bins_count = [0, 1, 2, 3, 6, np.inf]
            labels_count = ['0', '1', '2', '3-5', '6+']
            df['bins_items_correct_answer'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)
            #print(df['bins_items_correct_answer'])
        
        if 'bins_items_correct_answer_few' in my_column:
            bins_count = [0, 2, np.inf]
            labels_count = ['0-1', '2+']
            df['bins_items_correct_answer_few'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)

        if 'cue_size' in my_column:
            df['cue_size'] = [4-elem.count('*') for elem in df['cue']]

        for column, value in filter_dict.items():
            df = df[df[column] == value]

        if my_column == '':
            result = df[[metric]].copy()
            result['count'] = 1
            result = result[['count', metric]]
            return result

        result = df.groupby(my_column, observed = False).agg({
            metric: ['mean', 'std', 'count']
        })
        result.columns = ['f1_score_mean', 'f1_score_std', 'count']
        result = result.reset_index()
        result[metric] = result.apply(lambda row: f"{row['f1_score_mean']:.2f}±{row['f1_score_std']:.2f}", axis=1)
        if sorting:
            result = result.sort_values(by='f1_score_mean', ascending=True)
        result = result.drop(columns=['f1_score_mean', 'f1_score_std'])
        # Reset the index to make 'n_items' a column
        result = result.reset_index(drop=True)
        return result
    
    def get_accuracy_summary_relative_to(self, my_column='q_idx', sorting=False, filter_dict={}, show_warnings=False):
        df = self.df_generated_evaluations.copy()

        # Handle binning
        if 'bins_items_correct_answer' in str(my_column):
            df['bins_items_correct_answer'] = pd.cut(df['n_chapters_correct_answer'], 
                                                    bins=[0,1,2,3,6,np.inf], 
                                                    labels=['0','1','2','3-5','6+'], 
                                                    include_lowest=True, right=False)
        if 'bins_items_correct_answer_few' in str(my_column):
            df['bins_items_correct_answer_few'] = pd.cut(df['n_chapters_correct_answer'], 
                                                        bins=[0,2,np.inf], 
                                                        labels=['0-1','2+'], 
                                                        include_lowest=True, right=False)
        if 'bins_items_correct_answer_three' in str(my_column):
            df['bins_items_correct_answer_three'] = pd.cut(df['n_chapters_correct_answer'], 
                                                        bins=[0,1,2,np.inf], 
                                                        labels=['0','1','2+'], 
                                                        include_lowest=True, right=False)
        if 'cue_size' in str(my_column):
            df['cue_size'] = [4-elem.count('*') for elem in df['cue']]

        # Calculate perfect recall (found all ground truth items)
        df['perfect_match'] = (df['recall'] == 1.0).astype(int)
        
        # Handle cases with imperfect recall
        imperfect_recall = df['recall'] < 1.0
        if imperfect_recall.any():
            if show_warnings:
                print(f"Warning: Found {imperfect_recall.sum()} cases with incomplete recall")
            df.loc[imperfect_recall, 'perfect_match'] = 0

        # Apply filters
        for col, val in filter_dict.items():
            df = df[df[col] == val]

        # Handle empty grouping
        if not my_column or (isinstance(my_column, list) and len(my_column) == 0):
            accuracy = df['perfect_match'].mean() * 100
            return pd.DataFrame({'count': [len(df)], 'Accuracy': [f"{accuracy:.2f}±0.00"]})

        # Group and calculate
        result = df.groupby(my_column, observed=False)['perfect_match'].agg(['mean', 'count']).reset_index()
        result.columns = [*result.columns[:-2], 'accuracy_mean', 'count']
        result['Accuracy'] = result['accuracy_mean'].apply(lambda x: f"{x*100:.2f}±0.00")
        
        if sorting:
            result = result.sort_values('accuracy_mean')
        
        return result.drop('accuracy_mean', axis=1).reset_index(drop=True)
    
    def remove_duplicates_and_negative_one(self, lst):
        seen = set()
        return [x for x in lst if x != -1 and not (x in seen or seen.add(x))]

    def process_lists_and_compute_kendall_tau(self, l1, l2):
        # Step 1: Keep only elements that are in both lists
        common_elements = list(set(l1) & set(l2))
        nb_matches = len(common_elements)
        
        # Create new lists with only common elements, preserving original order
        result_l1 = [x for x in l1 if x in common_elements]
        result_l2 = [x for x in l2 if x in common_elements]
        
        # Step 2: Compute Kendall tau (discard the p-value)
        tau, _ = kendalltau(
            [result_l1.index(x) for x in common_elements],
            [result_l2.index(x) for x in common_elements]
        )
        
        return float(tau), nb_matches

    def compute_kendall_summarise(self, df_generated_chronological, verbose = True):
        chrono = df_generated_chronological        
        chrono['filtered_predicted_indexes'] = [self.remove_duplicates_and_negative_one(lst) for lst in chrono['predicted_indexes']]
        chrono['groundtruth_indexes_length'] = [len(x) for x in chrono['groundtruth_indexes']]

        chrono['tau'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[0] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]
        chrono['nb_matches'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[1] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]

        # all the samples in which the chronological order can be tested
        chrono_larger_than_one = chrono[chrono['groundtruth_indexes_length'] > 1]
        N = len(chrono_larger_than_one)
        # --- FIX FOR ZERO DIVISION ERROR START ---
        if N == 0:
            if verbose:
                print("No samples with 'groundtruth_indexes_length' > 1. Cannot compute Kendall Tau statistics for these.")
            return pd.DataFrame({
                '#gt_with_len_2+': 0,
                '#exact_match_set_gt_with_pred': 0,
                '%_exact_match_set_gt_with_pred': "0%",
                'tau_exact_match_set_gt_with_pred': "N/A",
                '#partial_match_set_gt_with_pred': 0,
                '%_partial_match_set_gt_with_pred': "0%",
                'tau_partial_match_set_gt_with_pred': "N/A"
            }, index=[0])
        # --- FIX FOR ZERO DIVISION ERROR END ---

        chrono_gt_larger_than_one_with_total_match = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']]
        count_of_total_match = sum(chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']) # % total match
        percentage_of_total_match = 100*round(count_of_total_match/N, 2)
        tau_over_total_match = float(chrono_gt_larger_than_one_with_total_match['tau'].mean())
        sd_tau_over_total_match = float(chrono_gt_larger_than_one_with_total_match['tau'].std())

        chrono_gt_larger_than_one_with_match_greater_than_one = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] > 1]
        count_of_match_greater_than_1 = sum(chrono_larger_than_one['nb_matches'] > 1) # % match > 1
        percentage_of_match_greater_than_1 = 100*round(count_of_match_greater_than_1/N, 2)
        tau_over_match_greater_than_one = float(chrono_gt_larger_than_one_with_match_greater_than_one['tau'].mean())
        sd_tau_over_match_greater_than_one = float(chrono_gt_larger_than_one_with_match_greater_than_one['tau'].std())

        if verbose:
            print(f"For the {count_of_total_match} samples with exact match between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_total_match}%), the kendall tau average is {round(tau_over_total_match,2)}±{round(sd_tau_over_total_match,2)}.")
            print(f"For the {count_of_match_greater_than_1} samples with partial match >1 between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_match_greater_than_1}%), the kendall tau average is {round(tau_over_match_greater_than_one,2)}±{round(sd_tau_over_match_greater_than_one,2)}.")

        kendall_summaries_for_this_experiment = pd.DataFrame({
            '#gt_with_len_2+': N, 
            '#exact_match_set_gt_with_pred': count_of_total_match, 
            '%_exact_match_set_gt_with_pred': f"{round(percentage_of_total_match)}%", 
            'tau_exact_match_set_gt_with_pred': f"{round(tau_over_total_match,2)}±{round(sd_tau_over_total_match,2)}",
            '#partial_match_set_gt_with_pred': count_of_match_greater_than_1, 
            '%_partial_match_set_gt_with_pred': f"{round(percentage_of_match_greater_than_1)}%", 
            'tau_partial_match_set_gt_with_pred': f"{round(tau_over_match_greater_than_one,2)}±{round(sd_tau_over_match_greater_than_one,2)}"
            }, index = [0])
        return kendall_summaries_for_this_experiment
    
def find_best_epoch_performance(results):
    """
    Find the epoch value with best performance based on Kendall correlation.
    
    Args:
        results: Results dict from run_incremental_epoch_experiments
    
    Returns:
        int: Best epoch value
    """
    best_score = -float('inf')
    best_epochs = None
    
    for epochs, result in results.items():
        if result['kendall_summaries'] and 'average_kendall' in result['kendall_summaries']:
            score = result['kendall_summaries']['average_kendall']
            if score > best_score:
                best_score = score
                best_epochs = epochs
    
    return best_epochs if best_epochs is not None else list(results.keys())[-1]


def run_incremental_epoch_experiments(my_benchmark, base_answering_parameters, data_folder, env_file, 
                                    max_epochs=100, evaluation_interval=10, debug=False):
    """
    Run training incrementally for different epoch values.
    Automatically resumes from last completed checkpoint.
    """
    epochs_to_test = list(range(evaluation_interval, max_epochs + 1, evaluation_interval))
    
    # Check for existing checkpoints and resume from last completed
    completed_epochs = _find_completed_checkpoints(
        epochs_to_test, my_benchmark, base_answering_parameters, data_folder, debug
    )
    
    if completed_epochs:
        last_completed = max(completed_epochs)
        epochs_to_test = [e for e in epochs_to_test if e > last_completed]
        
        if debug:
            print(f"🔄 Found existing checkpoints up to {last_completed} epochs")
        
        if epochs_to_test:
            print(f"📍 Resuming training from epoch {last_completed+1} to {epochs_to_test[-1]}")
        else:
            print(f"✅ All training completed up to {max_epochs} epochs!")
            return {}
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STARTING INCREMENTAL EPOCH OPTIMIZATION")
        print(f"Testing epochs: {epochs_to_test}")
        print(f"Intervals: {[epochs_to_test[i] - (epochs_to_test[i-1] if i > 0 else 0) for i in range(len(epochs_to_test))]}")
        print(f"{'='*80}")
    
    results = {}
    
    for i, target_epochs in enumerate(epochs_to_test):
        if debug:
            print(f"\n{'='*60}")
            print(f"TRAINING PHASE {i+1}/{len(epochs_to_test)}: Target = {target_epochs} epochs")
            print(f"{'='*60}")
        else:
            print(f"Training phase {i+1}/{len(epochs_to_test)}: {target_epochs} epochs")
        
        # Calculate epochs to train this round
        epochs_to_train = (target_epochs if i == 0 else 
                          target_epochs - epochs_to_test[i-1])
        
        if debug:
            if i == 0:
                print(f"→ Starting from scratch, training for {epochs_to_train} epochs")
            else:
                print(f"→ Continuing from {epochs_to_test[i-1]} epochs")
                print(f"→ Training {epochs_to_train} more epochs to reach {target_epochs}")
        
        # Prepare parameters
        current_params = base_answering_parameters.copy()
        current_params['n_epochs'] = target_epochs
        current_params['actual_training_epochs'] = epochs_to_train
        current_params['target_epochs'] = target_epochs

        # Set up continual learning if not first iteration
        if i > 0:
            previous_lora_path = _get_previous_lora_path(
                epochs_to_test[i-1], my_benchmark, base_answering_parameters, data_folder
            )
            current_params['continue_from_lora_path'] = previous_lora_path
            current_params['resume_from_checkpoint'] = str(previous_lora_path)  # Now this works!
            
            # Your existing debug code...
            print(f"🔍 DEBUG: previous_lora_path = {previous_lora_path}")
            print(f"🔍 DEBUG: Path exists? {os.path.exists(previous_lora_path)}")
            if os.path.exists(previous_lora_path):
                print(f"🔍 DEBUG: Contents: {os.listdir(previous_lora_path)}")
            
            if debug:
                print(f"→ Loading from: {previous_lora_path}")
                
        try:
            my_evaluation = EvaluationWrapper(my_benchmark, current_params, data_folder, env_file)
            
            results[target_epochs] = {
                'target_epochs': target_epochs,
                'epochs_trained_this_round': epochs_to_train,
                'lora_path': str(my_evaluation.lora_weights_pathname / "lora_adapters"),
                'kendall_summaries': my_evaluation.kendall_summaries_for_this_experiment,
                'commonsense_summary': my_evaluation.commonsense_summary,
                'df_generated_answers': my_evaluation.df_generated_answers,
                'df_generated_evaluations': my_evaluation.df_generated_evaluations,
                'df_generated_chronological': my_evaluation.df_generated_chronological
            }
            
            # Show accuracy if available
            acc = "N/A"
            if (results[target_epochs]['commonsense_summary'] and 
                'accuracy' in results[target_epochs]['commonsense_summary']):
                acc = f"{results[target_epochs]['commonsense_summary']['accuracy']:.2f}%"
            
            print(f"✅ Completed {target_epochs} epochs. Accuracy: {acc}")
            
        except Exception as e:
            print(f"✗ Error during training phase {i+1}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            break
    
    return results


def _find_completed_checkpoints(epochs_to_test, my_benchmark, base_answering_parameters, data_folder, debug):
    """Helper function to find existing completed checkpoints WITH complete evaluation"""
    completed_epochs = []
    
    for target_epochs in epochs_to_test:
        test_params = base_answering_parameters.copy()
        test_params['n_epochs'] = target_epochs
        
        # Check LoRA adapters
        _, lora_path = ftuning_input_filepath_func(
            base_answering_parameters['ftuning_input_data_policy'],
            my_benchmark.nb_tokens(), test_params, my_benchmark.nb_chapters(), data_folder,
            my_benchmark.prompt_parameters, my_benchmark.model_parameters, my_benchmark.book_parameters
        )
        
        lora_adapters_path = lora_path / "lora_adapters"
        training_checkpoint_file = lora_path / "training_checkpoint.pt"  # ADD THIS LINE
        
        # CHANGE THIS LINE to check both:
        training_complete = (lora_adapters_path.exists() and os.listdir(lora_adapters_path) and 
                           training_checkpoint_file.exists())  # ADD checkpoint check
        
        if training_complete:
            # Your existing evaluation check code stays the same...
            evaluation_complete = _check_evaluation_complete(target_epochs, my_benchmark, test_params, data_folder)
            
            if evaluation_complete:
                completed_epochs.append(target_epochs)
                if debug:
                    print(f"   ✅ Complete: {target_epochs} epochs (training + evaluation)")
            else:
                if debug:
                    print(f"   🔄 Training done but evaluation incomplete: {target_epochs} epochs")
                break
        else:
            if debug:
                print(f"   ❌ Training incomplete: {target_epochs} epochs")
            break
    
    return completed_epochs


def _check_evaluation_complete(target_epochs, my_benchmark, answering_parameters, data_folder):
    """Verifica se evaluation è completa: stesso numero di file in raw_answers e evaluated_answers + summary"""
    from In_vitro_experiments.io.io import answer_dirpath_func
    
    # Costruisci il path base per questo checkpoint
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters
    
    answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, 
                                        prompt_parameters, model_parameters, book_parameters, answering_parameters)
    
    raw_answers_dir = answer_dirpath / 'raw_answers'
    evaluated_answers_dir = answer_dirpath / 'evaluated_answers'
    commonsense_summary_file = answer_dirpath / 'commonsense_evaluation_summary.json'
    
    # Controlla se esistono le cartelle
    if not raw_answers_dir.exists() or not evaluated_answers_dir.exists():
        return False
    
    # Conta i file in raw_answers (escludendo cartelle e file nascosti)
    raw_files = [f for f in raw_answers_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    # Conta i file in evaluated_answers (escludendo cartelle e file nascosti)  
    evaluated_files = [f for f in evaluated_answers_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    # Controlla se il summary esiste
    summary_exists = commonsense_summary_file.exists()
    
    # Evaluation è completa se:
    # 1. Stesso numero di file in raw_answers e evaluated_answers
    # 2. Esiste il file commonsense_evaluation_summary.json
    evaluation_complete = (len(raw_files) == len(evaluated_files) and 
                          len(raw_files) > 0 and 
                          summary_exists)
    
    return evaluation_complete


def _get_previous_lora_path(previous_epochs, my_benchmark, base_answering_parameters, data_folder):
    """Helper function to get the path of previous LoRA checkpoint"""
    from In_vitro_experiments.io.io import ftuning_input_filepath_func
    
    previous_params = base_answering_parameters.copy()
    previous_params['n_epochs'] = previous_epochs
    
    _, previous_lora_weights_pathname = ftuning_input_filepath_func(
        base_answering_parameters['ftuning_input_data_policy'],
        my_benchmark.nb_tokens(), previous_params, my_benchmark.nb_chapters(), data_folder,
        my_benchmark.prompt_parameters, my_benchmark.model_parameters, my_benchmark.book_parameters
    )
    
    # Return just the parent path, NOT the lora_adapters subfolder
    return previous_lora_weights_pathname / "lora_adapters"