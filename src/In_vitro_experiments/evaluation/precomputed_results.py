from In_vitro_experiments.evaluation.evaluation_wrapper import EvaluationWrapper
from In_vitro_experiments.evaluation.generator_answers_2_rag import get_top_n
import pandas as pd

def get_precomputed_results(experiments, 
                            env_file, 
                            data_folder,
                            all_benchmarks = {'benchmark_claude_default_20': None,
                                              'benchmark_claude_default_200': None,
                                              'benchmark_claude_default_2000': None,
                                              'benchmark_claude_default_20_ordered': None,
                                              'benchmark_claude_default_200_ordered': None,
                                              'benchmark_gpt_default_20': None,
                                              'benchmark_gpt_default_200': None,
                                              'benchmark_claude_news_20': None,
                                              'benchmark_claude_news_200': None,
                                              'benchmark_claude_scifi_20': None,
                                              'benchmark_claude_scifi_200': None,
                                              'benchmark_claude_concise_10': None,
                                              'benchmark_claude_concise_20': None},
                            evaluation_policy = 'remove_duplicates'):
    df_list = []

    for i in range(len(experiments)):
        df_list.append(pd.DataFrame(experiments[i], index=[0]))
    # concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)
    df['evaluation_object'] = None

    for i in range(len(df)):
        df_cur = df.iloc[i]

        if df_cur['book_model_name'] == 'claude-3-5-sonnet-20240620':
            if df_cur['book_nb_events'] == 20:
                if 'book' in df_cur.index:
                    if df_cur['book'] == 'default':
                        my_benchmark = all_benchmarks['benchmark_claude_default_20']
                    elif df_cur['book'] == 'news':
                        my_benchmark = all_benchmarks['benchmark_claude_news_20']
                    elif df_cur['book'] == 'scifi':
                        my_benchmark = all_benchmarks['benchmark_claude_scifi_20']
                    elif df_cur['book'] == 'concise':
                        my_benchmark = all_benchmarks['benchmark_claude_concise_20']
                    else:
                        raise ValueError('Unknown book')
                elif ('ordered' in df_cur.index) and (df_cur['ordered']):
                    my_benchmark = all_benchmarks['benchmark_claude_default_20_ordered']
                else:
                    my_benchmark = all_benchmarks['benchmark_claude_concise_20']
            elif df_cur['book_nb_events'] == 200:
                if 'book' in df_cur.index:
                    if df_cur['book'] == 'default':
                        my_benchmark = all_benchmarks['benchmark_claude_default_200']
                    elif df_cur['book'] == 'news':
                        my_benchmark = all_benchmarks['benchmark_claude_news_200']
                    elif df_cur['book'] == 'scifi':
                        my_benchmark = all_benchmarks['benchmark_claude_scifi_200']
                    elif df_cur['book'] == 'concise':
                        my_benchmark = all_benchmarks['benchmark_claude_concise_200']
                    else:
                        raise ValueError('Unknown book')
                elif ('ordered' in df_cur.index) and (df_cur['ordered']):
                    my_benchmark = all_benchmarks['benchmark_claude_default_200_ordered']
                else:
                    my_benchmark = all_benchmarks['benchmark_claude_concise_200']
            elif df_cur['book_nb_events'] == 2000:
                my_benchmark = all_benchmarks['benchmark_claude_default_2000']
            elif df_cur['book_nb_events'] == 10:
                my_benchmark = all_benchmarks['benchmark_claude_concise_10']
            elif df_cur['book_nb_events'] == 99:
                my_benchmark = all_benchmarks['benchmark_claude_concise_100']
            elif df_cur['book_nb_events'] == 100:
                my_benchmark = all_benchmarks['benchmark_claude_concise_100']    
            elif df_cur['book_nb_events'] == 30:
                my_benchmark = all_benchmarks['benchmark_claude_concise_30']
            elif df_cur['book_nb_events'] == 300:
                my_benchmark = all_benchmarks['benchmark_claude_concise_300']
            else:
                raise ValueError('For `claude-3-5-sonnet-20240620`, only done with 20, 200, and 2000 target events')
            
        elif df_cur['book_model_name'] == 'gpt-4o-2024-05-13':
            if df_cur['book_nb_events'] == 20:
                my_benchmark = all_benchmarks['benchmark_gpt_default_20']
            elif df_cur['book_nb_events'] == 200:
                my_benchmark = all_benchmarks['benchmark_gpt_default_200']
            else:
                ValueError('For `gpt-4o-2024-05-13`, only done with 20 and 200 target events')
        else:
            raise ValueError('Only books generated with `claude-3-5-sonnet-20240620` and `gpt-4o-2024-05-13`')

        if df_cur['answering_kind'] == 'prompting':
            answering_parameters = {'kind': df_cur['answering_kind'],
                                    'model_name': df_cur['answering_model_name'],
                                    'max_new_tokens': 4096,
                                    'sleeping_time': 1,
                                    'policy': evaluation_policy}           
        elif df_cur['answering_kind'] == 'rag':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 1, 
                                    'embedding_chunk': df_cur['answering_embedding_chunk'], 
                                    'embedding_model': "text-embedding-3-small", 
                                    'embedding_batch_size': 2048, 
                                    'top_n': get_top_n(df_cur['answering_embedding_chunk'], my_benchmark), 
                                    'policy': evaluation_policy}
        elif df_cur['answering_kind'] == 'ftuning':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 0, 
                                    'ftuning_input_data_policy': 'single', 
                                    'ftuning_need_upload': False, 
                                    'ftuning_need_actual_tune': False, 
                                    'batch_size': 'auto', 
                                    'learning_rate_multiplier': 'auto', 
                                    'n_epochs': 10,
                                    'policy': evaluation_policy}
            # ad-hoc
            if df_cur['book_nb_events'] == 20:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AAzm9XtH'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::AB02Cbei'
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
            elif df_cur['book_nb_events'] == 200:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AB0B6H4o'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::DISCARDED' # DISCARDED (~400 dollars)
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
        elif df_cur['answering_kind'] == 'ftuning-manual':
            # Determina la ftuning_policy dall'esperimento o usa un default
            

            answering_parameters = {'kind': 'ftuning-manual', 
                                'model_name': df_cur.get('answering_model_name'), 
                                'max_new_tokens': 4096, 
                                'sleeping_time': 0, 
                                'ftuning_input_data_policy': df_cur.get('ftuning_input_data_policy'), 
                                'ftuning_need_upload': False, 
                                'ftuning_need_actual_tune': False, 
                                'batch_size': df_cur.get('batch_size','auto'), 
                                'learning_rate_multiplier': df_cur.get('learning_rate_multiplier','auto'), 
                                'n_epochs': int(df_cur.get('n_epochs')),
                                'policy': 'remove_duplicates',
                                'max_questions_commonsense': 9000}
            str_print = f"Document with {my_benchmark.nb_tokens()} tokens, answer with {df_cur['answering_kind']} using with {df_cur['answering_model_name']}"

            # Conditionally add keys only if their values are not None
            if df_cur.get("max_chapter") is not None and pd.notna(df_cur.get("max_chapter")):
                answering_parameters['max_chapter'] = int(df_cur.get("max_chapter"))

            if 'recollecting_approach' in df_cur and pd.notna(df_cur['recollecting_approach']) and pd.notna(df_cur.get("max_chapter")):
                answering_parameters['recollecting_approach'] = df_cur.get('recollecting_approach')
            if 'ftuning_policy' in df_cur and pd.notna(df_cur['ftuning_policy']):
                answering_parameters['ftuning_policy'] = df_cur.get('ftuning_policy')
                str_print = f"{str_print} (policy: {df_cur.get('ftuning_policy')})"
            if 'evaluation' in df_cur and pd.notna(df_cur['evaluation']):
                answering_parameters['evaluation'] = df_cur.get('evaluation')
            if 'clean_recall_from_hall' in df_cur:
                answering_parameters['clean_recall_from_hall'] = df_cur.get('clean_recall_from_hall')
            if 'ftuning_type' in df_cur and pd.notna(df_cur.get('ftuning_type')):
                answering_parameters['ftuning_type'] = df_cur.get('ftuning_type')
            if 'not_eval_hall' in df_cur and pd.notna(df_cur.get('not_eval_hall')):
                answering_parameters['not_eval_hall'] = df_cur.get('not_eval_hall')
            if 'grokking' in df_cur and pd.notna(df_cur.get('grokking')):
                answering_parameters['grokking'] = df_cur.get('grokking')
            #answering_parameters['fine_tuned_model_name'] = 'ft-lora-meta-llama-Meta-Llama-3-8B-Instruct'
            
        
        if df_cur['answering_kind'] == 'rag':
            str_print = f"{str_print} ({df_cur['answering_embedding_chunk']} chunks)"      
         
        print(answering_parameters)
        # my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)
        # df.loc[i, 'evaluation_object'] = my_evaluation

        
        my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file, judge_model_name='gemini-2.5-flash' )
        df.loc[i, 'evaluation_object'] = my_evaluation

    return df

