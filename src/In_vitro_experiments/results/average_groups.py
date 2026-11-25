import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union

def extract_groups(df, nb_events, relative_to, metric = 'f1_score_lenient'):

    df_sliced = df[(df['book_nb_events'] == nb_events) & (df['book_model_name'] == 'claude-3-5-sonnet-20240620')]

    # template
    i = 0
    df_res_0 = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
    df_results = df_res_0.iloc[:, :-1] # take all but last column

    # fill
    for i in range(len(df_sliced)):
        df_res_i = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
        df_results[(df_sliced.iloc[i]['answering_kind'], 
                    df_sliced.iloc[i]['answering_model_name'],
                    df_sliced.iloc[i]['answering_embedding_chunk'])] = [x for x in df_res_i.iloc[:, -1]] # average # [float(x.split('±')[0]) for x in df_res_i.iloc[:, -1]] # average

    # remove the nan
    df_results_tmp = df_results.copy()
    for col in relative_to + ['count']:
        df_results_tmp = df_results_tmp.loc[:, df_results_tmp.columns != col]
    nan_rows = [[k for i, x in enumerate(df_results_tmp.iloc[k]) if np.isnan(float(x.split('±')[0]))==True ] for k in range(len(df_results))]
    issue_rows = list(set([item for sublist in nan_rows for item in sublist]))
    df_results = df_results.drop(issue_rows)

    return df_results



def get_performance_stats_summary(df_experiments, show_individual_rows=False):
    """
    Extract performance statistics from experiments dataframe.
    
    Parameters:
    -----------
    df_experiments : pd.DataFrame
        DataFrame containing experiments with evaluation_object column
    show_individual_rows : bool
        If True, shows all individual evaluation rows
        If False, shows aggregated stats per experiment
    
    Returns:
    --------
    pd.DataFrame with performance statistics
    """
    
    performance_columns = [
        'precision', 'recall', 'f1_score', 
        'nb_preds_lenient', 'nb_preds_harsh',
        'precision_lenient', 'precision_harsh', 
        'f1_score_lenient', 'f1_score_harsh', 
        'diff_f1'
    ]
    
    results = []
    
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        
        # Get the generated evaluations dataframe
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Extract experiment info
        experiment_info = {
            'experiment_id': idx,
            'book_nb_events': row['book_nb_events'],
            'answering_model': row['answering_model_name'],
            'n_epochs': row['n_epochs'],
            'max_chapter': row['max_chapter']
        }
        
        if show_individual_rows:
            # Show all individual evaluation rows
            for eval_idx, eval_row in df_eval.iterrows():
                result_row = experiment_info.copy()
                result_row['eval_row_id'] = eval_idx
                result_row['question'] = eval_row.get('question', '')[:50] + '...' if len(str(eval_row.get('question', ''))) > 50 else eval_row.get('question', '')
                
                # Add performance metrics
                for col in performance_columns:
                    if col in df_eval.columns:
                        result_row[col] = eval_row[col]
                    else:
                        result_row[col] = np.nan
                
                results.append(result_row)
        else:
            # Show aggregated stats per experiment
            result_row = experiment_info.copy()
            
            # Extract commonsense accuracy
            commonsense_accuracy = np.nan
            if hasattr(eval_wrapper, 'commonsense_summary') and eval_wrapper.commonsense_summary:
                commonsense_accuracy = eval_wrapper.commonsense_summary.get('accuracy', np.nan)
            result_row['commonsense_accuracy'] = commonsense_accuracy
            
            # Calculate aggregated statistics
            for col in performance_columns:
                if col in df_eval.columns:
                    # For count-based metrics, sum them
                    if 'nb_' in col:
                        result_row[f'{col}_total'] = df_eval[col].sum()
                        result_row[f'{col}_mean'] = df_eval[col].mean()
                    else:
                        # For ratio-based metrics, calculate mean and std
                        valid_values = df_eval[col].dropna()
                        if len(valid_values) > 0:
                            result_row[f'{col}_mean'] = valid_values.mean()
                            result_row[f'{col}_std'] = valid_values.std()
                            result_row[f'{col}_count'] = len(valid_values)
                        else:
                            result_row[f'{col}_mean'] = np.nan
                            result_row[f'{col}_std'] = np.nan
                            result_row[f'{col}_count'] = 0
                else:
                    if 'nb_' in col:
                        result_row[f'{col}_total'] = np.nan
                        result_row[f'{col}_mean'] = np.nan
                    else:
                        result_row[f'{col}_mean'] = np.nan
                        result_row[f'{col}_std'] = np.nan
                        result_row[f'{col}_count'] = 0
            
            results.append(result_row)
    
    return pd.DataFrame(results)

def get_simple_performance_overview(df_experiments):
    """
    Get a simple overview focusing on the main F1 scores.
    """
    results = []
    
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Main metrics we care about
        f1_lenient = df_eval['f1_score_lenient'].dropna()
        f1_harsh = df_eval['f1_score_harsh'].dropna()
        precision_lenient = df_eval['precision_lenient'].dropna()
        recall = df_eval['recall'].dropna()
        
        # Extract commonsense accuracy
        commonsense_accuracy = np.nan
        if hasattr(eval_wrapper, 'commonsense_summary') and eval_wrapper.commonsense_summary:
            commonsense_accuracy = eval_wrapper.commonsense_summary.get('accuracy', np.nan)
        
        result = {
            'experiment_id': idx,
            'model': row['answering_model_name'].split('/')[-1],  # Just model name
            'epochs': row['n_epochs'],
            'n_questions': len(df_eval),
            'commonsense_accuracy': commonsense_accuracy,
            'f1_lenient_mean': f1_lenient.mean() if len(f1_lenient) > 0 else np.nan,
            'f1_lenient_std': f1_lenient.std() if len(f1_lenient) > 0 else np.nan,
            'f1_harsh_mean': f1_harsh.mean() if len(f1_harsh) > 0 else np.nan,
            'f1_harsh_std': f1_harsh.std() if len(f1_harsh) > 0 else np.nan,
            'precision_lenient_mean': precision_lenient.mean() if len(precision_lenient) > 0 else np.nan,
            'recall_mean': recall.mean() if len(recall) > 0 else np.nan,
        }
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Format the results nicely
    for col in ['commonsense_accuracy', 'f1_lenient_mean', 'f1_harsh_mean', 'precision_lenient_mean', 'recall_mean']:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    for col in ['f1_lenient_std', 'f1_harsh_std']:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    return df_results

def clean_evaluation_dataframe(df_eval, experiment_id=None):
    """
    Clean the evaluation dataframe by removing outliers and invalid values.
    
    For metrics that should be between 0 and 1:
    - f1_score_lenient
    - f1_score_harsh  
    - precision_lenient
    - recall
    
    Args:
        df_eval: DataFrame with evaluation results
        experiment_id: Optional experiment identifier for detailed logging
        
    Returns:
        df_eval_cleaned: DataFrame with outliers removed and invalid values set to NaN
        cleaning_stats: Dictionary with cleaning statistics
    """
    import pandas as pd
    import numpy as np
    
    # Create a copy to avoid modifying the original
    df_cleaned = df_eval.copy()
    
    # Define columns that should be between 0 and 1
    metric_columns = [
        'f1_score_lenient',
        'f1_score_harsh', 
        'precision_lenient',
        'recall'
    ]
    
    # Track cleaning statistics
    cleaning_stats = {}
    
    for col in metric_columns:
        if col in df_cleaned.columns:
            original_count = df_cleaned[col].notna().sum()
            
            # Find outliers (values outside [0, 1] range)
            outliers_mask = (df_cleaned[col] < 0) | (df_cleaned[col] > 1)
            n_outliers = outliers_mask.sum()
            
            # Get detailed info about outliers
            outlier_details = []
            if n_outliers > 0:
                outlier_indices = df_cleaned[outliers_mask].index.tolist()
                outlier_values = df_cleaned.loc[outliers_mask, col].tolist()
                
                for idx, val in zip(outlier_indices, outlier_values):
                    outlier_details.append({
                        'index': idx,
                        'value': val,
                        'experiment_id': experiment_id
                    })
                
                # Print detailed outlier information
                exp_info = f" (Experiment {experiment_id})" if experiment_id is not None else ""
                print(f"\nColumn '{col}'{exp_info}: Found {n_outliers} outliers:")
                for detail in outlier_details:
                    print(f"  - Index {detail['index']}: value = {detail['value']}")
            
            # Set outliers to NaN
            df_cleaned.loc[outliers_mask, col] = np.nan
            
            # Track what was cleaned
            cleaning_stats[col] = {
                'original_valid_count': original_count,
                'outliers_removed': n_outliers,
                'final_valid_count': df_cleaned[col].notna().sum(),
                'outlier_details': outlier_details
            }
    
    # Optional: Print summary if any cleaning was done
    total_outliers = sum(stats['outliers_removed'] for stats in cleaning_stats.values())
    if total_outliers > 0:
        exp_info = f" in experiment {experiment_id}" if experiment_id is not None else ""
        print(f"\nTotal outliers removed{exp_info}: {total_outliers}")
        print("Cleaning complete - outliers set to NaN and will be excluded from mean calculations")
    
    return df_cleaned, cleaning_stats


def get_performance_overview_by_chapters_with_cleaning(df_experiments):
    """
    Enhanced version of get_performance_overview_by_chapters with data cleaning.
    
    This function cleans the evaluation data before computing means by removing
    outliers (values outside [0,1] range) for metric columns.
    """
        
    results = []
    
    # First pass: check what combinations exist across all experiments
    existing_combinations = set()
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Clean the data first
        df_eval_cleaned, _ = clean_evaluation_dataframe(df_eval, experiment_id=idx)
        
        # Check what combinations exist in this experiment
        for _, chapters in df_eval_cleaned['correct_answer_chapters'].items():
            try:
                if isinstance(chapters, np.ndarray):
                    if len(chapters) == 1 and chapters[0] == 1:
                        existing_combinations.add('ch1')
                    elif len(chapters) == 1 and chapters[0] == 2:
                        existing_combinations.add('ch2')
                    elif len(chapters) == 2 and set(chapters) == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
                elif isinstance(chapters, set):
                    if chapters == {1}:
                        existing_combinations.add('ch1')
                    elif chapters == {2}:
                        existing_combinations.add('ch2')
                    elif chapters == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
                elif isinstance(chapters, list):
                    if len(chapters) == 1 and chapters[0] == 1:
                        existing_combinations.add('ch1')
                    elif len(chapters) == 1 and chapters[0] == 2:
                        existing_combinations.add('ch2')
                    elif len(chapters) == 2 and set(chapters) == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
            except:
                continue
    
    # Second pass: process experiments with cleaned data
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        rec_approach = row['recollecting_approach']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # CLEAN THE DATA FIRST
        df_eval_cleaned, cleaning_stats = clean_evaluation_dataframe(df_eval, experiment_id=idx)
        
        # Create boolean masks handling numpy arrays
        def is_ch1(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 1 and x[0] == 1
                elif isinstance(x, set):
                    return x == {1}
                elif isinstance(x, list):
                    return len(x) == 1 and x[0] == 1
                else:
                    return False
            except:
                return False
                
        def is_ch2(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 1 and x[0] == 2
                elif isinstance(x, set):
                    return x == {2}
                elif isinstance(x, list):
                    return len(x) == 1 and x[0] == 2
                else:
                    return False
            except:
                return False
                
        def is_ch1_and_ch2(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 2 and set(x) == {1, 2}
                elif isinstance(x, set):
                    return x == {1, 2}
                elif isinstance(x, list):
                    return len(x) == 2 and set(x) == {1, 2}
                else:
                    return False
            except:
                return False
                
        def is_hallucinating(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 0
                elif isinstance(x, (list, set)):
                    return len(x) == 0
                else:
                    return False
            except:
                return False
        
        # Create masks only for combinations that exist - using CLEANED data
        if 'ch1' in existing_combinations:
            ch1_mask = df_eval_cleaned['correct_answer_chapters'].apply(is_ch1)
            df_eval_ch1 = df_eval_cleaned[ch1_mask]
            f1_lenient_ch1 = df_eval_ch1['f1_score_lenient'].dropna()
            recall_ch1 = df_eval_ch1['recall'].dropna()
        
        if 'ch2' in existing_combinations:
            ch2_mask = df_eval_cleaned['correct_answer_chapters'].apply(is_ch2)
            df_eval_ch2 = df_eval_cleaned[ch2_mask]
            f1_lenient_ch2 = df_eval_ch2['f1_score_lenient'].dropna()
            recall_ch2 = df_eval_ch2['recall'].dropna()
        
        if 'ch1_and_ch2' in existing_combinations:
            ch1_and_ch2_mask = df_eval_cleaned['correct_answer_chapters'].apply(is_ch1_and_ch2)
            df_eval_ch1_and_ch2 = df_eval_cleaned[ch1_and_ch2_mask]
            f1_lenient_ch1_and_ch2 = df_eval_ch1_and_ch2['f1_score_lenient'].dropna()
            recall_ch1_and_ch2 = df_eval_ch1_and_ch2['recall'].dropna()
        
        if 'hallucinating' in existing_combinations:
            hallucinating_mask = df_eval_cleaned['correct_answer_chapters'].apply(is_hallucinating)
            df_eval_hallucinating = df_eval_cleaned[hallucinating_mask]
            f1_lenient_hall = df_eval_hallucinating['f1_score_lenient'].dropna()
            recall_hall = df_eval_hallucinating['recall'].dropna()
        
        # Calculate overall metrics using CLEANED data
        f1_lenient = df_eval_cleaned['f1_score_lenient'].dropna()
        f1_harsh = df_eval_cleaned['f1_score_harsh'].dropna()
        precision_lenient = df_eval_cleaned['precision_lenient'].dropna()
        recall = df_eval_cleaned['recall'].dropna()
        
        # Extract commonsense accuracy
        commonsense_accuracy = np.nan
        if hasattr(eval_wrapper, 'commonsense_summary') and eval_wrapper.commonsense_summary:
            commonsense_accuracy = eval_wrapper.commonsense_summary.get('accuracy', np.nan)
        
        result = {
            # Original columns
            'model': row['answering_model_name'].split('/')[-1],
            'epochs': row['n_epochs'],
            'rec_approach': rec_approach,
            'commonsense_accuracy': commonsense_accuracy,
            'f1_lenient_mean': f1_lenient.mean() if len(f1_lenient) > 0 else np.nan,
            'f1_lenient_std': f1_lenient.std() if len(f1_lenient) > 0 else np.nan,
            'f1_harsh_mean': f1_harsh.mean() if len(f1_harsh) > 0 else np.nan,
            'f1_harsh_std': f1_harsh.std() if len(f1_harsh) > 0 else np.nan,
            'precision_lenient_mean': precision_lenient.mean() if len(precision_lenient) > 0 else np.nan,
            'recall_mean': recall.mean() if len(recall) > 0 else np.nan,
        }
        
        # Add chapter-specific columns only if they exist in the data
        if 'ch1' in existing_combinations:
            result['f1_lenient_mean_ch1'] = f1_lenient_ch1.mean() if len(f1_lenient_ch1) > 0 else np.nan
            result['recall_mean_ch1'] = recall_ch1.mean() if len(recall_ch1) > 0 else np.nan
            result['n_questions_ch1'] = len(df_eval_ch1)
            
        if 'ch2' in existing_combinations:
            result['f1_lenient_mean_ch2'] = f1_lenient_ch2.mean() if len(f1_lenient_ch2) > 0 else np.nan
            result['recall_mean_ch2'] = recall_ch2.mean() if len(recall_ch2) > 0 else np.nan
            result['n_questions_ch2'] = len(df_eval_ch2)
            
        if 'ch1_and_ch2' in existing_combinations:
            result['f1_lenient_mean_ch1_and_ch2'] = f1_lenient_ch1_and_ch2.mean() if len(f1_lenient_ch1_and_ch2) > 0 else np.nan
            result['recall_mean_ch1_and_ch2'] = recall_ch1_and_ch2.mean() if len(recall_ch1_and_ch2) > 0 else np.nan
            result['n_questions_ch1_and_ch2'] = len(df_eval_ch1_and_ch2)
            
        if 'hallucinating' in existing_combinations:
            result['f1_lenient_mean_hallucinating'] = f1_lenient_hall.mean() if len(f1_lenient_hall) > 0 else np.nan
            result['recall_mean_hallucinating'] = recall_hall.mean() if len(recall_hall) > 0 else np.nan
            result['n_questions_hallucinating'] = len(df_eval_hallucinating)
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Round numerical columns to 3 decimal places
    numerical_cols = [
        'commonsense_accuracy', 'f1_lenient_mean', 'f1_harsh_mean', 'precision_lenient_mean', 
        'recall_mean'
    ]
    
    # Add chapter-specific columns to rounding list only if they exist
    if 'ch1' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch1', 'recall_mean_ch1'])
    if 'ch2' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch2', 'recall_mean_ch2'])
    if 'ch1_and_ch2' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch1_and_ch2', 'recall_mean_ch1_and_ch2'])
    if 'hallucinating' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_hallucinating', 'recall_mean_hallucinating'])
    
    for col in numerical_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    return df_results

def create_epoch_pivot_table(df, nb_events, relative_to, model_of_interest, metric='f1_score_lenient'):
    """
    Creates a pivot table with epochs as multi-level columns.
    Uses your existing extract_groups function without modifying it.
    """
    
    # Get all unique epochs for this model
    model_data = df[(df['book_nb_events'] == nb_events) & 
                    (df['answering_kind'] == model_of_interest[0]) & 
                    (df['answering_model_name'] == model_of_interest[1])]
    
    epochs = sorted(model_data['n_epochs'].unique())
    print(f"Found epochs for {model_of_interest}: {epochs}")
    
    pivot_data = []
    
    # For each epoch, create a separate filtered dataset and run your existing code
    for epoch in epochs:
        print(f"Processing epoch {epoch}...")
        
        # Filter data for this specific epoch
        df_epoch = df[df['n_epochs'] == epoch]
        
        # Use your existing extract_groups function (unchanged!)
        df_results_epoch = extract_groups(df_epoch, nb_events, relative_to, metric)
        df_results_epoch = df_results_epoch[df_results_epoch['get'] == 'all']
        
        # Check if this model exists in this epoch's results
        if model_of_interest not in df_results_epoch.columns:
            print(f"  No data for {model_of_interest} in epoch {epoch}")
            continue
            
        df_results_epoch = df_results_epoch[relative_to + ['count', model_of_interest]]
        
        if len(df_results_epoch) == 0:
            print(f"  No data after filtering for epoch {epoch}")
            continue
        
        # Create pivot for this epoch (your existing logic)
        df_results_epoch['value'] = [f"{x} ({y})" for x, y in 
                                   zip(df_results_epoch[model_of_interest], df_results_epoch['count'])]
        
        try:
            epoch_pivot = df_results_epoch.pivot(index='cue', columns='bins_items_correct_answer', values='value')
            
            # Add epoch as top-level column
            epoch_pivot.columns = pd.MultiIndex.from_product([[f"epoch_{epoch}"], epoch_pivot.columns])
            pivot_data.append(epoch_pivot)
            
        except Exception as e:
            print(f"  Error creating pivot for epoch {epoch}: {e}")
            continue
    
    if not pivot_data:
        print("No data found for any epoch!")
        return None
    
    # Combine all epochs
    df_results_pivoted = pd.concat(pivot_data, axis=1)
    
    # Sort by number of cues (your existing logic)
    df_results_pivoted['nb_cues'] = [4-x.count('*') for x in df_results_pivoted.index]
    df_results_pivoted = df_results_pivoted.sort_values('nb_cues')
    
    return df_results_pivoted.drop('nb_cues', axis=1)

def get_performance_overview_by_chapters(df_experiments):
    """
    Get a performance overview filtering by chapters based on correct_answer_chapters column.
    Separates results into:
    - Chapter 1 only: correct_answer_chapters contains {1}
    - Chapter 2 only: correct_answer_chapters contains {2}  
    - Hallucinating: correct_answer_chapters is empty list []
    """
        
    results = []
    
    # First pass: check what combinations exist across all experiments
    existing_combinations = set()
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Check what combinations exist in this experiment
        for _, chapters in df_eval['correct_answer_chapters'].items():
            try:
                if isinstance(chapters, np.ndarray):
                    if len(chapters) == 1 and chapters[0] == 1:
                        existing_combinations.add('ch1')
                    elif len(chapters) == 1 and chapters[0] == 2:
                        existing_combinations.add('ch2')
                    elif len(chapters) == 2 and set(chapters) == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
                elif isinstance(chapters, set):
                    if chapters == {1}:
                        existing_combinations.add('ch1')
                    elif chapters == {2}:
                        existing_combinations.add('ch2')
                    elif chapters == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
                elif isinstance(chapters, list):
                    if len(chapters) == 1 and chapters[0] == 1:
                        existing_combinations.add('ch1')
                    elif len(chapters) == 1 and chapters[0] == 2:
                        existing_combinations.add('ch2')
                    elif len(chapters) == 2 and set(chapters) == {1, 2}:
                        existing_combinations.add('ch1_and_ch2')
                    elif len(chapters) == 0:
                        existing_combinations.add('hallucinating')
            except:
                continue
    
    # Second pass: process experiments and only create columns for existing combinations
    
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        rec_approach = row['recollecting_approach']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Create boolean masks handling numpy arrays
        def is_ch1(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 1 and x[0] == 1
                elif isinstance(x, set):
                    return x == {1}
                elif isinstance(x, list):
                    return len(x) == 1 and x[0] == 1
                else:
                    return False
            except:
                return False
                
        def is_ch2(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 1 and x[0] == 2
                elif isinstance(x, set):
                    return x == {2}
                elif isinstance(x, list):
                    return len(x) == 1 and x[0] == 2
                else:
                    return False
            except:
                return False
                
        def is_ch1_and_ch2(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 2 and set(x) == {1, 2}
                elif isinstance(x, set):
                    return x == {1, 2}
                elif isinstance(x, list):
                    return len(x) == 2 and set(x) == {1, 2}
                else:
                    return False
            except:
                return False
                
        def is_hallucinating(x):
            try:
                if isinstance(x, np.ndarray):
                    return len(x) == 0
                elif isinstance(x, (list, set)):
                    return len(x) == 0
                else:
                    return False
            except:
                return False
        
        # Create masks only for combinations that exist
        if 'ch1' in existing_combinations:
            ch1_mask = df_eval['correct_answer_chapters'].apply(is_ch1)
            df_eval_ch1 = df_eval[ch1_mask]
            f1_lenient_ch1 = df_eval_ch1['f1_score_lenient'].dropna()
            recall_ch1 = df_eval_ch1['recall'].dropna()
        
        if 'ch2' in existing_combinations:
            ch2_mask = df_eval['correct_answer_chapters'].apply(is_ch2)
            df_eval_ch2 = df_eval[ch2_mask]
            f1_lenient_ch2 = df_eval_ch2['f1_score_lenient'].dropna()
            recall_ch2 = df_eval_ch2['recall'].dropna()
        
        if 'ch1_and_ch2' in existing_combinations:
            ch1_and_ch2_mask = df_eval['correct_answer_chapters'].apply(is_ch1_and_ch2)
            df_eval_ch1_and_ch2 = df_eval[ch1_and_ch2_mask]
            f1_lenient_ch1_and_ch2 = df_eval_ch1_and_ch2['f1_score_lenient'].dropna()
            recall_ch1_and_ch2 = df_eval_ch1_and_ch2['recall'].dropna()
        
        if 'hallucinating' in existing_combinations:
            hallucinating_mask = df_eval['correct_answer_chapters'].apply(is_hallucinating)
            df_eval_hallucinating = df_eval[hallucinating_mask]
            f1_lenient_hall = df_eval_hallucinating['f1_score_lenient'].dropna()
            recall_hall = df_eval_hallucinating['recall'].dropna()
        
        
        # Calculate overall metrics (original columns)
        f1_lenient = df_eval['f1_score_lenient'].dropna()
        f1_harsh = df_eval['f1_score_harsh'].dropna()
        precision_lenient = df_eval['precision_lenient'].dropna()
        recall = df_eval['recall'].dropna()
        
        # Extract commonsense accuracy
        commonsense_accuracy = np.nan
        if hasattr(eval_wrapper, 'commonsense_summary') and eval_wrapper.commonsense_summary:
            commonsense_accuracy = eval_wrapper.commonsense_summary.get('accuracy', np.nan)
        
        result = {
            # Original columns
            'experiment_id': idx,
            'model': row['answering_model_name'].split('/')[-1],  # Just model name
            'epochs': row['n_epochs'],
            'rec_approach': rec_approach,
            #'n_questions': len(df_eval),
            'commonsense_accuracy': commonsense_accuracy,
            'f1_lenient_mean': f1_lenient.mean() if len(f1_lenient) > 0 else np.nan,
            'f1_lenient_std': f1_lenient.std() if len(f1_lenient) > 0 else np.nan,
            'f1_harsh_mean': f1_harsh.mean() if len(f1_harsh) > 0 else np.nan,
            'f1_harsh_std': f1_harsh.std() if len(f1_harsh) > 0 else np.nan,
            'precision_lenient_mean': precision_lenient.mean() if len(precision_lenient) > 0 else np.nan,
            'recall_mean': recall.mean() if len(recall) > 0 else np.nan,
        }
        
        # Add chapter-specific columns only if they exist in the data
        if 'ch1' in existing_combinations:
            result['f1_lenient_mean_ch1'] = f1_lenient_ch1.mean() if len(f1_lenient_ch1) > 0 else np.nan
            result['recall_mean_ch1'] = recall_ch1.mean() if len(recall_ch1) > 0 else np.nan
            result['n_questions_ch1'] = len(df_eval_ch1)
            
        if 'ch2' in existing_combinations:
            result['f1_lenient_mean_ch2'] = f1_lenient_ch2.mean() if len(f1_lenient_ch2) > 0 else np.nan
            result['recall_mean_ch2'] = recall_ch2.mean() if len(recall_ch2) > 0 else np.nan
            result['n_questions_ch2'] = len(df_eval_ch2)
            
        if 'ch1_and_ch2' in existing_combinations:
            result['f1_lenient_mean_ch1_and_ch2'] = f1_lenient_ch1_and_ch2.mean() if len(f1_lenient_ch1_and_ch2) > 0 else np.nan
            result['recall_mean_ch1_and_ch2'] = recall_ch1_and_ch2.mean() if len(recall_ch1_and_ch2) > 0 else np.nan
            result['n_questions_ch1_and_ch2'] = len(df_eval_ch1_and_ch2)
            
        if 'hallucinating' in existing_combinations:
            result['f1_lenient_mean_hallucinating'] = f1_lenient_hall.mean() if len(f1_lenient_hall) > 0 else np.nan
            result['recall_mean_hallucinating'] = recall_hall.mean() if len(recall_hall) > 0 else np.nan
            result['n_questions_hallucinating'] = len(df_eval_hallucinating)
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Round numerical columns to 3 decimal places - only for existing combinations
    numerical_cols = [
        'commonsense_accuracy', 'f1_lenient_mean', 'f1_harsh_mean', 'precision_lenient_mean', 
        'recall_mean'
    ]
    
    # Add chapter-specific columns to rounding list only if they exist
    if 'ch1' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch1', 'recall_mean_ch1'])
    if 'ch2' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch2', 'recall_mean_ch2'])
    if 'ch1_and_ch2' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_ch1_and_ch2', 'recall_mean_ch1_and_ch2'])
    if 'hallucinating' in existing_combinations:
        numerical_cols.extend(['f1_lenient_mean_hallucinating', 'recall_mean_hallucinating'])
    
    for col in numerical_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    return df_results

def validate_and_clean_performance_data(df_results, verbose=True):
    """
    Validate performance metrics and remove outliers.
    
    Args:
        df_results: DataFrame with performance metrics
        verbose: Whether to print validation information
    
    Returns:
        Cleaned DataFrame with outliers removed/corrected
    """
    
    
    df_cleaned = df_results.copy()
    
    # Define metric columns that should be between 0 and 1
    metric_columns = [col for col in df_cleaned.columns if any(metric in col for metric in 
                     ['f1_', 'recall_', 'precision_', 'commonsense_accuracy'])]
    
    if verbose:
        print("=== DATA VALIDATION REPORT ===")
        print(f"Validating {len(metric_columns)} metric columns: {metric_columns}")
        print()
    
    outliers_found = False
    
    for col in metric_columns:
        if col in df_cleaned.columns:
            # Check for values > 1.0
            invalid_high = df_cleaned[col] > 1.0
            # Check for values < 0.0  
            invalid_low = df_cleaned[col] < 0.0
            
            if invalid_high.any():
                outliers_found = True
                if verbose:
                    print(f"⚠️  Column '{col}' has {invalid_high.sum()} values > 1.0:")
                    problem_rows = df_cleaned[invalid_high][['experiment_id', 'model', col]]
                    print(problem_rows.to_string(index=False))
                    print(f"   Max value: {df_cleaned[col].max():.3f}")
                    print()
                
                # Cap values at 1.0
                df_cleaned.loc[invalid_high, col] = 1.0
                
            if invalid_low.any():
                outliers_found = True
                if verbose:
                    print(f"⚠️  Column '{col}' has {invalid_low.sum()} values < 0.0:")
                    problem_rows = df_cleaned[invalid_low][['experiment_id', 'model', col]]
                    print(problem_rows.to_string(index=False))
                    print(f"   Min value: {df_cleaned[col].min():.3f}")
                    print()
                
                # Set negative values to 0.0
                df_cleaned.loc[invalid_low, col] = 0.0
    
    # Additional checks for statistical outliers (using IQR method)
    if verbose:
        print("=== STATISTICAL OUTLIER DETECTION ===")
    
    for col in metric_columns:
        if col in df_cleaned.columns and not df_cleaned[col].isna().all():
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers as values beyond 1.5 * IQR from Q1/Q3
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            statistical_outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            
            if statistical_outliers.any() and verbose:
                print(f"📊 Column '{col}' statistical outliers (beyond 1.5*IQR):")
                print(f"   Valid range: [{lower_bound:.3f}, {upper_bound:.3f}]")
                outlier_rows = df_cleaned[statistical_outliers][['experiment_id', 'model', col]]
                print(outlier_rows.to_string(index=False))
                print()
    
    # Check for impossible combinations (e.g., recall=0 but f1>0)
    if verbose:
        print("=== LOGICAL CONSISTENCY CHECKS ===")
    
    # Find pairs of f1 and recall columns
    f1_cols = [col for col in df_cleaned.columns if 'f1_' in col and '_mean' in col]
    
    for f1_col in f1_cols:
        # Find corresponding recall column
        recall_col = f1_col.replace('f1_lenient_mean', 'recall_mean')
        
        if recall_col in df_cleaned.columns:
            # Check for cases where recall=0 but f1>0 (impossible)
            impossible = (df_cleaned[recall_col] == 0) & (df_cleaned[f1_col] > 0)
            
            if impossible.any() and verbose:
                print(f"🚫 Impossible combination: {recall_col}=0 but {f1_col}>0:")
                problem_rows = df_cleaned[impossible][['experiment_id', 'model', recall_col, f1_col]]
                print(problem_rows.to_string(index=False))
                print()
    
    if not outliers_found and verbose:
        print("✅ No metric outliers found! All values are within [0, 1] range.")
    elif verbose:
        print("🔧 Outliers have been corrected (capped at 0-1 range).")
    
    if verbose:
        print("\n=== SUMMARY STATISTICS ===")
        print(df_cleaned[metric_columns].describe())
    
    return df_cleaned



def get_performance_overview_with_custom_question_types(df_experiments, my_benchmark, 
                                                       question_type_mapping: Dict[str, str],
                                                       question_type_column: str = 'bins_items_correct_answer'):
    """
    More flexible version that allows custom grouping of question types.
    
    Parameters:
    -----------
    df_experiments : pd.DataFrame
        DataFrame containing experiment results
    my_benchmark : object
        Benchmark object containing df_qa with question metadata
    question_type_mapping : Dict[str, str]
        Mapping from question type values to column suffixes
        Example: {"0": "hallucinating", "1": "single_event", "2,3,4,5": "multi_event"}
    question_type_column : str
        Column name in my_benchmark.df_qa that contains question type info
    """
    results = []
    
    for idx, row in df_experiments.iterrows():
        eval_wrapper = row['evaluation_object']
        df_eval = eval_wrapper.df_generated_evaluations
        
        # Get question type information
        question_types = my_benchmark.df_qa[question_type_column]
        
        # Calculate overall metrics (original columns)
        f1_lenient = df_eval['f1_score_lenient'].dropna()
        f1_harsh = df_eval['f1_score_harsh'].dropna()
        precision_lenient = df_eval['precision_lenient'].dropna()
        recall = df_eval['recall'].dropna()
        
        # Extract commonsense accuracy
        commonsense_accuracy = np.nan
        if hasattr(eval_wrapper, 'commonsense_summary') and eval_wrapper.commonsense_summary:
            commonsense_accuracy = eval_wrapper.commonsense_summary.get('accuracy', np.nan)
        
        result = {
            # Original columns
            'experiment_id': idx,
            'model': row['answering_model_name'].split('/')[-1],
            'epochs': row['n_epochs'],
            'n_questions': len(df_eval),
            'commonsense_accuracy': commonsense_accuracy,
            'f1_lenient_mean': f1_lenient.mean() if len(f1_lenient) > 0 else np.nan,
            'f1_lenient_std': f1_lenient.std() if len(f1_lenient) > 0 else np.nan,
            'f1_harsh_mean': f1_harsh.mean() if len(f1_harsh) > 0 else np.nan,
            'f1_harsh_std': f1_harsh.std() if len(f1_harsh) > 0 else np.nan,
            'precision_lenient_mean': precision_lenient.mean() if len(precision_lenient) > 0 else np.nan,
            'recall_mean': recall.mean() if len(recall) > 0 else np.nan,
        }
        
        # Calculate metrics for each custom question type
        for type_values, suffix in question_type_mapping.items():
            # Handle comma-separated values (e.g., "2,3,4,5")
            if ',' in type_values:
                type_list = type_values.split(',')
                mask = question_types.isin(type_list)
            else:
                mask = question_types == type_values
            
            # Get indices for this type
            type_indices = my_benchmark.df_qa[mask].index
            
            # Filter evaluations for this type
            df_eval_type = df_eval[df_eval.index.isin(type_indices)]
            
            # Calculate metrics for this type
            f1_lenient_type = df_eval_type['f1_score_lenient'].dropna()
            recall_type = df_eval_type['recall'].dropna()
            
            # Add to result
            result[f'f1_lenient_mean_{suffix}'] = f1_lenient_type.mean() if len(f1_lenient_type) > 0 else np.nan
            result[f'recall_mean_{suffix}'] = recall_type.mean() if len(recall_type) > 0 else np.nan
            result[f'n_questions_{suffix}'] = len(df_eval_type)
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Format the results nicely
    numerical_cols = [col for col in df_results.columns if 'mean' in col and col != 'model']
    numerical_cols.append('commonsense_accuracy')
    
    for col in numerical_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    # Round standard deviation columns
    for col in ['f1_lenient_std', 'f1_harsh_std']:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(3)
    
    return df_results


# Example usage functions
def analyze_hallucinating_vs_general(df_experiments, my_benchmark):
    """
    Simple function to get the basic table with hallucinating vs general question columns.
    """
    return get_simple_performance_overview_with_question_types(df_experiments, my_benchmark)


def analyze_with_custom_grouping(df_experiments, my_benchmark, custom_mapping):
    """
    Function to use custom question type groupings.
    
    Example custom_mapping:
    {
        "0": "hallucinating",
        "1": "single_event", 
        "2,3": "multi_event"
    }
    """
    return get_performance_overview_with_custom_question_types(
        df_experiments, my_benchmark, custom_mapping
    )