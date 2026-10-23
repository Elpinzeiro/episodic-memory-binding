# Events, meta-events, and prompting
from In_vitro_experiments.generation.generate_1_events_and_meta_events import generate_and_export_events_and_meta_events_func
# Generate (or load existing) events
from In_vitro_experiments.generation.generate_2_paragraph import iterative_generate_paragraphs_func
# Post-processing (catch invalid samples, remove the numbering, replace entities)
from In_vitro_experiments.generation.generate_3_secondary_entities import get_final_samples, count_tokens
# Build book
from In_vitro_experiments.generation.generate_4_books import book_indexing_func, build_chapter
# Build questions
from In_vitro_experiments.generation.qa_generation import build_qa_func, checking_widespreadness_of_questions
# IO
import pandas as pd
import pyarrow
import ast
from In_vitro_experiments.io.io import book_dirpath_func, export_list, import_list
# Printing
import numpy as np
from In_vitro_experiments.generation.printing import pretty_print_before_post_processing, pretty_print, wrap_text, split_chapters_func, find_relative_position
# Fine-tuning
from random import shuffle
from In_vitro_experiments.generation.qa_generation import retrieve_dataframe_for_single_question, build_nonempty_qa_func
# For reference
from In_vitro_experiments.generation.raw_materials_extended import universe_parameters
from In_vitro_experiments.generation.generate_1_events_and_meta_events import generate_universe
# Plotting
from pathlib import Path
import os
from datasets import load_dataset
from In_vitro_experiments.models.models_wrapper import ModelsWrapper
from In_vitro_experiments.models.settings_wrapper import SettingsWrapper

import shutil

COMMONSENSE_REPO = "tau/commonsense_qa"
COMMONSENSE_REVISION = "94630fe30dad47192a8546eb75f094926d47e155"


def download_commonsense(commonsense_data_dir):
    """Fetch the CommonsenseQA parquet files (public HF dataset tau/commonsense_qa) into commonsense_data_dir."""
    from huggingface_hub import hf_hub_download
    os.makedirs(commonsense_data_dir, exist_ok=True)
    for split in ("train", "validation", "test"):
        name = f"{split}-00000-of-00001.parquet"
        target = os.path.join(commonsense_data_dir, name)
        if os.path.exists(target):
            continue
        try:
            path = hf_hub_download(COMMONSENSE_REPO, f"data/{name}", repo_type="dataset", revision=COMMONSENSE_REVISION)
        except Exception as e:
            print(f"Warning: could not download CommonsenseQA from Hugging Face ({e.__class__.__name__}). "
                  f"Without internet, copy the parquet files of {COMMONSENSE_REPO} into {commonsense_data_dir} (see README).")
            return
        shutil.copyfile(path, target)

from typing import Optional  # Add this if not already there
import json


class BenchmarkGenerationWrapper:
    def __init__(
            self,
            prompt_parameters = {
                'nb_events': 20, 
                'name_universe': 'default', 
                'name_styles': 'default', 
                'seed': 0, 
                'distribution_events': {'name': 'geometric', 'param': 0.1}
            },
            model_parameters = {
                'model_name': 'claude-3-5-sonnet-20240620',
                'max_new_tokens': 4096, 
                'itermax': 10 # itermax is integer, 1 for a single try
            },
            book_parameters = {
                'indexing': 'default'
            },
            data_folder = 'data/narratives',
            env_file = '.env',
            verbose = True,
            rechecking = True
            ):
        book, df_book_groundtruth, df_qa, df_qa_debug_widespreadness, debug_all_generated_samples = self.__end2end(
            prompt_parameters, model_parameters, book_parameters, data_folder, env_file, verbose, rechecking)
        self.book = book
        self.df_book_groundtruth = df_book_groundtruth
        self.df_qa = df_qa
        self.df_qa_debug_widespreadness = df_qa_debug_widespreadness
        self.debug_all_generated_samples = debug_all_generated_samples
        self.commonsense_dataset = self.load_commonsense_dataset(data_folder)

        # input values
        self.prompt_parameters = prompt_parameters
        self.model_parameters =  model_parameters
        self.book_parameters = book_parameters
        self.data_folder = data_folder
        self.env_file = env_file
        self.verbose = verbose
        self.rechecking = rechecking

        # other
        self.split_chapters = split_chapters_func(self.book) # dictionary giving the corresponding chapter given the chapter_idx
        self.debug_mapping_chapter_idx_to_event_idx = self.__get_mapping_chapter_idx_to_event_idx()

        events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
        self.events = events
        self.meta_events = meta_events

        # fine tuning questions
        if prompt_parameters['nb_events'] <= 1000:
            nb_chapters_max = 200
        else:
            nb_chapters_max = 2000
        finetuning_questions = build_nonempty_qa_func(self.df_book_groundtruth, events, nb_chapters_max, seed = 0) # all nonempty questions
        finetuning_questions = self.replace_template_by_full_books(finetuning_questions) # replace full event questions with actual chapter content
        finetuning_questions_one_chapter = finetuning_questions[finetuning_questions['n_chapters_correct_answer'] == 1] # all questions that refer to exactly 1 question
        self.finetuning_questions = finetuning_questions
        self.finetuning_questions_one_chapter = finetuning_questions_one_chapter
        

        # for reference
        disjoint_book_id = prompt_parameters.get('disjoint_book_id', None)
        n_total_books = prompt_parameters.get('n_total_books', 10)
        parameters_universe = universe_parameters(prompt_parameters['name_universe'], disjoint_book_id, n_total_books)
        N_universe = 100 // n_total_books if disjoint_book_id is not None else 100
        seed_universe = 0
        temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe)
        self.universe_t = temporal
        self.universe_e = entities
        self.universe_s = spatial
        self.universe_c = content
        self.universe_cd = details


    def __end2end(
            self,
            prompt_parameters = {
                'nb_events': 20, 
                'name_universe': 'default', 
                'name_styles': 'default', 
                'seed': 0, 
                'distribution_events': {'name': 'geometric', 'param': 0.1}
                },
            model_parameters = {
                'model_name': 'claude-3-5-sonnet-20240620', # 'gpt-4o-2024-05-13'
                'max_new_tokens': 4096, 
                'itermax': 10 # itermax is integer, 1 for a single try
                },
            book_parameters = {
                'indexing': 'default'
                },
            data_folder = 'data/narratives',
            env_file = '.env',
            verbose = True,
            rechecking = True):

        events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
        generated_paragraphs, has_verif_vector = iterative_generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, verbose, rechecking)
        d = get_final_samples(prompt_parameters, model_parameters, data_folder, rechecking)
        
        idx_chapters = book_indexing_func(d, book_parameters) # selection of the events selected as chapter and their ordering
        book, df_book_groundtruth = build_chapter(idx_chapters, d)

        nb_chapters = len(df_book_groundtruth['chapter'].values)
        nb_tokens = count_tokens(book)

        if prompt_parameters['nb_events'] <= 1000:
            nb_chapters_max = 200
        else:
            nb_chapters_max = 2000

        book_dirpath = book_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters)
        if (not book_dirpath.is_dir()) or rechecking: # book does not exist, or we want to recheck
            df_qa = build_qa_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max)
            df_qa['correct_answer_detailed'] = df_qa['correct_answer_detailed'].apply(str) # necessary to save into a parquet
            df_qa_debug_widespreadness = checking_widespreadness_of_questions(df_qa)

        # io begin: load and save, to ensure that all the time the same book and qa
        book_filepath = book_dirpath / "book.json"
        df_book_groundtruth_filepath = book_dirpath / "df_book_groundtruth.parquet"
        df_qa_filepath = book_dirpath / "df_qa.parquet"
        df_qa_debug_widespreadness_filepath = book_dirpath / "df_qa_debug_widespreadness.parquet"

        if book_dirpath.is_dir(): # book already exists, loading all the files
            book_existing = import_list(book_filepath)
            df_book_groundtruth_existing = pd.read_parquet(df_book_groundtruth_filepath, engine='pyarrow')
            df_qa_existing = pd.read_parquet(df_qa_filepath, engine='pyarrow')
            df_qa_debug_widespreadness_existing = pd.read_parquet(df_qa_debug_widespreadness_filepath, engine='pyarrow')
            if rechecking: # book does exist, but we want to recheck
                assert book_existing == book, 'different book produced' 
                pd.testing.assert_frame_equal(df_book_groundtruth_existing, df_book_groundtruth)
                # columns that cannot be compared directly are discarded
                my_cols = ['correct_answer', 'correct_answer_chapters', 'debug_changed']
                pd.testing.assert_frame_equal(df_qa_existing.drop(my_cols, axis = 1), df_qa.drop(my_cols, axis = 1))
                pd.testing.assert_frame_equal(df_qa_debug_widespreadness_existing, df_qa_debug_widespreadness)
            df_qa = df_qa_existing
            df_qa_debug_widespreadness = df_qa_debug_widespreadness_existing
        else:
            # saving
            book_dirpath.mkdir(parents=True, exist_ok=True)
            export_list(book, book_filepath)
            df_book_groundtruth.to_parquet(df_book_groundtruth_filepath)
            df_qa.to_parquet(df_qa_filepath)
            df_qa_debug_widespreadness.to_parquet(df_qa_debug_widespreadness_filepath)

        # revert the apply(str) applied on column 'correct_answer_detailed' that is necessary to save into a parquet
        df_qa['correct_answer_detailed'] = [ast.literal_eval(x) for x in df_qa['correct_answer_detailed']]
        # re-apply conversion to set()
        df_qa['debug_changed'] = [set(x) for x in df_qa['debug_changed']]
        df_book_groundtruth['post_entities'] = [set(x) for x in df_book_groundtruth['post_entities']]
        # io end, also doing qa loading

        debug_all_generated_samples = d # for debugging only, get all information even for the samples at iteration that were not successful

        # additionally put the nb_paragraphs, position of each event feature, and style within the df_book_groundtruth (with care of the index)
        meta_event_info = [meta_events[x] for x in df_book_groundtruth['raw_generated_paragraph_idx']]
        df_book_groundtruth['nb_paragraphs'] = [x['nb_paragraphs'] for x in meta_event_info]
        df_book_groundtruth['style'] = [x['style'] for x in meta_event_info]
        df_book_groundtruth['idx_t'] = [x['idx_paragraph']['date'] for x in meta_event_info]
        df_book_groundtruth['idx_s'] = [x['idx_paragraph']['location'] for x in meta_event_info]
        df_book_groundtruth['idx_e'] = [x['idx_paragraph']['entity'] for x in meta_event_info]
        df_book_groundtruth['idx_c'] = [x['idx_paragraph']['content'] for x in meta_event_info]

        return book, df_book_groundtruth, df_qa, df_qa_debug_widespreadness, debug_all_generated_samples

    ## Related to `df_book_groundtruth`
    def get_df_book_groundtruth(self):
        return self.df_book_groundtruth
    
    ## Related to `df_qa`
    def get_df_qa(self):
        return self.df_qa
    
    ## Related to `df_qa_debug_widespreadness`
    def get_df_qa_debug_widespreadness(self):
        return self.df_qa_debug_widespreadness
    
    ## Related to `debug_all_generated_samples`
    def get_debug_all_generated_samples(self):
        return self.debug_all_generated_samples

    def pretty_print_debug_event_idx(self, event_idx = 0, width = 150):
        """
        **For debugging only**

        Given an *event index* from 0 to nb_events-1, 
        return the last iteration of the generated set of paragraphs if an iteration succeeded,
        or None if all the iteration failed.

        The output would constitutes a single chapter, but for now the indexing is different (and begins at 0)
        """
        pretty_print(event_idx, self.debug_all_generated_samples, width)
    
    def get_chapter(self, max_chapter):
        """
        Given chapter from 1 to max_chapter, 
        return a tuple containing (event, chapter_content) if the iteration succeeded,
        or None if the iteration failed.
        
        Returns:
            tuple: (event_list, generated_paragraph_text) or None if invalid
        """
        event_idx = max_chapter-1
        if event_idx >= len(self.debug_all_generated_samples):
            return None
            
        event_data = self.debug_all_generated_samples[event_idx]
        
        # Check if this sample is valid
        if not event_data.get('is_valid', False):
            return None
        
        # Extract the event and generated paragraph
        event = event_data.get('event', [])
        generated_paragraph = event_data.get('paragraphs', '')
        
        return (event, generated_paragraph)
    
    def get_text_up_to(self, max_chapter):
        """
        Given chapters from 1 to max_chapter,
        return concatenated chapter content from all valid chapters,
        or empty string if no valid chapters found.
        
        Returns:
            str: concatenated generated_paragraph_text from all valid chapters
        """
        return "".join(
            chapter_data[1] for i in range(1, max_chapter + 1)
            if (chapter_data := self.get_chapter(i)) is not None
        )
    def pretty_print_debug_event_iter_idx(self, event_idx = 0, iter_idx = "last", width = 150):
        """
        **For debugging only**

        Given an *event index* (from `0` to `nb_events-1`; or "last" for the last event), 
        given an iteration index (from `0` to the successful one, or to `itermax-1` if none succeed),
        return the generated set of paragraphs.

        The output would be a single chapter if successful, otherwise it is not used anymore.
        """
        pretty_print_before_post_processing(event_idx, iter_idx, self.prompt_parameters, self.model_parameters, self.data_folder, width)

    def invalid_debug_event_idx_func(self):
        d = self.debug_all_generated_samples
        possible_events = [e for e in range(len(d))]
        invalid_idxs = [e for e in possible_events if not d[e]['is_valid']]
        return invalid_idxs

    ## Related to `book`
    def get_book(self):
        return self.book
    
    def pretty_print_book(self):
        print(wrap_text(self.book))

    def pretty_print_book_chapter(self, chapter = 1, width = 150):
        # retrieve the original event_idx for this chapter
        event_idx = self.debug_mapping_chapter_idx_to_event_idx[chapter]

        # check that retrieving the chapter from the book itself gives the same as compared to retrieve the successful event_idx
        assert self.debug_all_generated_samples[event_idx]['paragraphs'] == self.split_chapters[chapter], "chapter extracted from book is not the same to the generated original sample"

        # use the event_idx function, since it is already done
        return self.pretty_print_debug_event_idx(event_idx, width)

    def __get_mapping_chapter_idx_to_event_idx(self):
        idx_chapters = book_indexing_func(self.debug_all_generated_samples, self.book_parameters)
        return {1+chapter: e for (chapter,e) in zip(range(len(idx_chapters)) , idx_chapters)}

    def nb_tokens(self):
        return count_tokens(self.book)
    
    def nb_chapters(self):
        return len(self.df_book_groundtruth['chapter'].values)
    
    ## Get relative positions
    def relative_position_within_paragraph(self):
        target_paragraph_t = [int(self.df_book_groundtruth['idx_t'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_s = [int(self.df_book_groundtruth['idx_s'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_e = [int(self.df_book_groundtruth['idx_e'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_c = [int(self.df_book_groundtruth['idx_c'][chapter]) for chapter in self.df_book_groundtruth['chapter']]

        def get_generated_chapter(chapter):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            return generated_paragraph
        
        cut_paragraphs_t = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_t)]
        cut_paragraphs_s = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_s)]
        cut_paragraphs_e = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_e)]
        cut_paragraphs_c = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_c)]

        def find_relative_position_paragraph_func(chapter, cut_paragraphs_t, cut_paragraphs_s, cut_paragraphs_e, cut_paragraphs_c):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            event_index = d[event_index]['event_idx']
            event = d[event_index]['event']
            pos_t_paragraph = find_relative_position(event[0].lower(), cut_paragraphs_t[chapter-1].lower())
            pos_s_paragraph = find_relative_position(event[1].lower(), cut_paragraphs_s[chapter-1].lower())
            pos_e_paragraph = find_relative_position(event[2].lower(), cut_paragraphs_e[chapter-1].lower())
            pos_cd_paragraph = find_relative_position(event[4].lower(), cut_paragraphs_c[chapter-1].lower())
            return pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_cd_paragraph

        relative_positions = [find_relative_position_paragraph_func(chapter, cut_paragraphs_t, cut_paragraphs_s, cut_paragraphs_e, cut_paragraphs_c) for chapter in self.df_book_groundtruth['chapter']]
        pos_t_paragraph = [x[0] for x in relative_positions]
        pos_s_paragraph = [x[1] for x in relative_positions]
        pos_e_paragraph = [x[2] for x in relative_positions]
        pos_c_paragraph = [x[3] for x in relative_positions]

        return pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_c_paragraph

    def relative_position_within_chapter(self):
        def find_relative_position_func(chapter):
            # retrieve the original event_idx for this chapter
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]

            # use the event_idx function, since it is already done
            d = self.debug_all_generated_samples

            generated_paragraph = d[event_index]['paragraphs']
            event_index = d[event_index]['event_idx']
            event = d[event_index]['event']

            pos_t = find_relative_position(event[0].lower(), generated_paragraph.lower())
            pos_s = find_relative_position(event[1].lower(), generated_paragraph.lower())
            pos_e = find_relative_position(event[2].lower(), generated_paragraph.lower())
            pos_cd = find_relative_position(event[4].lower(), generated_paragraph.lower())

            return pos_t, pos_s, pos_e, pos_cd
    
        relative_positions = [find_relative_position_func(chapter) for chapter in self.df_book_groundtruth['chapter']]
        pos_t = [x[0] for x in relative_positions]
        pos_s = [x[1] for x in relative_positions]
        pos_e = [x[2] for x in relative_positions]
        pos_c = [x[3] for x in relative_positions]
        return pos_t, pos_s, pos_e, pos_c # each of length 196 and between 0 and 1 (relative to the chapter)

    def relative_position_within_book(self):
        def get_generated_chapter(chapter):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            return generated_paragraph

        chapter_lengths = [len(get_generated_chapter(chapter)) for chapter in self.df_book_groundtruth['chapter']]
        position_of_the_begin_of_each_chapter = np.cumsum([0] + chapter_lengths)[:-1] # 0 for the first chapter

        nb_characters_book = sum(chapter_lengths)

        pos_t, pos_s, pos_e, pos_c = self.relative_position_within_chapter()

        absolute_position_within_the_chapter_t = [prop*lenchapter for (prop, lenchapter) in zip(pos_t, chapter_lengths)] # position within the chapter
        absolute_position_within_the_chapter_s = [prop*lenchapter for (prop, lenchapter) in zip(pos_s, chapter_lengths)]
        absolute_position_within_the_chapter_e = [prop*lenchapter for (prop, lenchapter) in zip(pos_e, chapter_lengths)]
        absolute_position_within_the_chapter_c = [prop*lenchapter for (prop, lenchapter) in zip(pos_c, chapter_lengths)]

        pos_t_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_t, position_of_the_begin_of_each_chapter)]
        pos_s_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_s, position_of_the_begin_of_each_chapter)]
        pos_e_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_e, position_of_the_begin_of_each_chapter)]
        pos_c_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_c, position_of_the_begin_of_each_chapter)]

        return pos_t_book, pos_s_book, pos_e_book, pos_c_book
    
    # For RAG:
    def chunk_paragraphs(self, input_str = "mouth.\n\nAs the\ndajsj", my_split = '\n'):
        """
        \n for paragraph split, \n\n\n for chapter split
        """
        # Split the string by one or more newline characters
        chunks = input_str.split(my_split)
        # Remove any empty strings resulting from consecutive newlines
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def chunk_book(self, split = 'chapter'):
        """
        split: either 'chapter' or 'paragraph'
        """
        if split == 'chapter':
            return self.chunk_paragraphs(self.book, '\n\n\n')
        elif split == 'paragraph':
            xss = [[f"Chapter {k}, Paragraph {idx+1}\n\n{x}" for idx, x in enumerate(self.chunk_paragraphs(v))] for k,v in self.split_chapters.items()]
            flat_list = [x for xs in xss for x in xs]
            return flat_list
        
    # For fine-tuning:
    def book_content_func(self, chapter_idx):
        all_questions_for_this_chapter = retrieve_dataframe_for_single_question(chapter_idx, self.df_book_groundtruth, unused_universe = None, changed = set(), existing_change = True)
        full_event_question_for_this_chapter = all_questions_for_this_chapter[all_questions_for_this_chapter['retrieval_type'] == "Full event details"].copy()
        full_event_question_for_this_chapter['correct_answer_long'] = self.split_chapters[chapter_idx]
        full_event_question_for_this_chapter = full_event_question_for_this_chapter[['chapter', 'question', 'correct_answer_long']]
        book_content = full_event_question_for_this_chapter.iloc[0]['correct_answer_long']
        return book_content
        
    def replace_template_by_full_books(self, finetuning_questions):
        idx_to_replace = finetuning_questions[finetuning_questions['retrieval_type']== "Full event details"].index
        finetuning_questions_with_full_chapters = finetuning_questions.copy()
        for i in idx_to_replace:
            idx_chapter = finetuning_questions.iloc[i]['debug_chapter'][0]
            full_chapter = self.book_content_func(idx_chapter)
            finetuning_questions_with_full_chapters.loc[i, 'correct_answer'] = full_chapter
            
        return finetuning_questions_with_full_chapters

    def book_name(self):
        return "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition"
    
    def fine_tuning_system_prompt(self):
        return f'You are an expert in memory tests regarding the fictional book "{self.book_name()}".'
    
    def format_list(self, input_set_or_list):
        # case of full chapter
        if isinstance(input_set_or_list, str):
            return input_set_or_list

        l = list(input_set_or_list)
        if len(l) == 0:
            return ""
        elif len(l) == 1:
            return l[0]
        else:
            return ", ".join(l[:-1]) + f", and {l[-1]}"
        
    def get_user_prompt_for_finetuning(self, question):
        preprompt = f'This question is about the book "{self.book_name()}". All events in this book are purely fictional and do not correspond to real-world timelines. Please answer based solely on the content of this fictional story.'
        res = f"{preprompt}\n\n Question: {question}"  # Fixed: \n instead of /n
        return res

    def process_questions_for_finetuning(self, finetuning_questions_df):
        """
        Process a DataFrame with questions and answers into fine-tuning format.
        
        Args:
            finetuning_questions_df: DataFrame with at minimum ['question'] and ['correct_answer'] columns
        
        Returns:
            DataFrame with added columns: 'system_prompt', 'user_prompt', 'assistant_answer'
        """
        # Make a copy to avoid modifying the original
        processed_questions = finetuning_questions_df.copy()
        
        # Add the required columns for fine-tuning
        processed_questions['system_prompt'] = self.fine_tuning_system_prompt()
        processed_questions['user_prompt'] = [self.get_user_prompt_for_finetuning(x) for x in processed_questions['question']]
        processed_questions['assistant_answer'] = [self.format_list(x) for x in processed_questions['correct_answer']]
        
        return processed_questions

    def get_fine_tuning_questions(self, answer_in_one_chapter_only=True):
        """
        Get fine-tuning questions from the original datasets (backward compatibility method).
        
        Args:
            answer_in_one_chapter_only: Whether to use one-chapter questions
        
        Returns:
            DataFrame with processed fine-tuning questions
        """
        if answer_in_one_chapter_only:
            finetuning_questions = self.finetuning_questions_one_chapter.copy()
        else:
            finetuning_questions = self.finetuning_questions.copy()

        return self.process_questions_for_finetuning(finetuning_questions)

    def get_json_single_line(self, system_prompt, user_prompt, assistant_answer):
        my_json_line = {"messages": [{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": assistant_answer}]}
        return my_json_line

    def build_fine_tuning_jsonl(self, finetuning_questions_df=None, answer_in_one_chapter_only=True):
        """
        Build fine-tuning JSONL from provided questions DataFrame or default datasets.
        
        Args:
            finetuning_questions_df: DataFrame with ['question'] and ['correct_answer'] columns.
                                    If None, uses default datasets based on answer_in_one_chapter_only
            answer_in_one_chapter_only: Whether to use one-chapter questions (only used if finetuning_questions_df is None)
        
        Returns:
            List of JSON objects for fine-tuning
        """
        if finetuning_questions_df is not None:
            # Use the provided DataFrame
            processed_questions = self.process_questions_for_finetuning(finetuning_questions_df)
        else:
            # Use default behavior for backward compatibility
            processed_questions = self.get_fine_tuning_questions(answer_in_one_chapter_only)
        
        res = [self.get_json_single_line(s, u, a) for (s, u, a) in zip(
            processed_questions['system_prompt'], 
            processed_questions['user_prompt'], 
            processed_questions['assistant_answer']
        )]
        
        shuffle(res)
        return res
    



    def load_commonsense_dataset(self, data_folder):
        """
        Load the commonsense dataset from parquet files.
        
        Args:
            data_folder: Base data folder path
            
        Returns:
            Dataset object containing the commonsense data, or None if not found
        """
        try:
            # Construct the path to commonsense data
            commonsense_data_dir = os.path.join(data_folder, "commonsense")
            download_commonsense(commonsense_data_dir)
            
            # Check if the directory exists
            if not os.path.exists(commonsense_data_dir):
                print(f"Warning: Commonsense data directory not found at {commonsense_data_dir}")
                return None
            
            # Define data files
            data_files = {}
            
            # Check for train file
            train_file = os.path.join(commonsense_data_dir, "train-00000-of-00001.parquet")
            if os.path.exists(train_file):
                data_files["train"] = train_file
            
            # Check for test file
            test_file = os.path.join(commonsense_data_dir, "test-00000-of-00001.parquet")
            if os.path.exists(test_file):
                data_files["test"] = test_file
            
            # Check for validation file (optional)
            validation_file = os.path.join(commonsense_data_dir, "validation-00000-of-00001.parquet")
            if os.path.exists(validation_file):
                data_files["validation"] = validation_file
            
            # If no data files found, return None
            if not data_files:
                print(f"Warning: No parquet files found in {commonsense_data_dir}")
                return None
            
            # Load the dataset
            print(f"Loading commonsense dataset from {commonsense_data_dir}")
            print(f"Found data files: {list(data_files.keys())}")
            
            ds = load_dataset("parquet", data_files=data_files)
            
            print(f"Successfully loaded commonsense dataset with splits: {list(ds.keys())}")
            
            # Print basic info about the dataset
            for split_name, split_data in ds.items():
                print(f"  {split_name}: {len(split_data)} examples")
                if len(split_data) > 0:
                    print(f"    Columns: {list(split_data.column_names)}")
            
            return ds
            
        except Exception as e:
            print(f"Error loading commonsense dataset: {e}")
            return None

    def get_commonsense_dataset(self):
        """
        Get the loaded commonsense dataset.
        
        Returns:
            The commonsense dataset or None if not loaded
        """
        return self.commonsense_dataset

    def get_commonsense_split(self, split_name="train"):
        """
        Get a specific split of the commonsense dataset.
        
        Args:
            split_name: Name of the split to retrieve ("train", "test", "validation")
            
        Returns:
            The requested split or None if not available
        """
        if self.commonsense_dataset is None:
            print("Commonsense dataset not loaded")
            return None
        
        if split_name not in self.commonsense_dataset:
            print(f"Split '{split_name}' not found. Available splits: {list(self.commonsense_dataset.keys())}")
            return None
        
        return self.commonsense_dataset[split_name]
    

    def llm_questions_generated(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: float = 0.1) -> str:
        """
        Generate questions using the configured LLM model with the given prompt.
        
        Args:
            prompt: The input prompt for question generation
            max_new_tokens: Maximum number of new tokens to generate (uses model_parameters default if None)
            temperature: Sampling temperature for generation
            
        Returns:
            Generated text response from the model
        """
               
        # Use max_new_tokens from model_parameters if not provided
        if max_new_tokens is None:
            max_new_tokens = self.model_parameters.get('max_new_tokens', 4096)
        
        # Initialize the models wrapper with current parameters
        config = SettingsWrapper(_env_file=self.env_file)
        models_wrapper = ModelsWrapper(
            model_name=self.model_parameters['model_name'],
            config=config,
            answering_parameters=self.model_parameters
        )
        
        # Prepare messages in the format expected by the model
        messages_for_model = [
            {"role": "user", "content": prompt}
        ]
        
        # Generate response using the models wrapper
        try:
            generated_text, reasoning_text = models_wrapper.generate(
                messages_for_model=messages_for_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            if self.verbose:
                print(f"Generated questions using {self.model_parameters['model_name']}")
                print(f"Input prompt length: {len(prompt)} characters")
                print(f"Generated response length: {len(generated_text)} characters")
                
            return generated_text
            
        except Exception as e:
            print(f"Error during question generation with {self.model_parameters['model_name']}: {e}")
            if self.verbose:
                print(f"Failed prompt: {prompt[:200]}...")  # Show first 200 chars for debugging
            return ""

    def parse_llm_response_to_dataframe(self,response):
        """
        Parse LLM JSON response string into a DataFrame compatible with build_fine_tuning_jsonl
        
        Args:
            response: JSON string response from LLM containing list of question-answer pairs
            
        Returns:
            pandas DataFrame with ['question', 'correct_answer'] columns
        """
        try:
            # Parse the JSON string
            questions_list = json.loads(response)
            
            # Convert to DataFrame format
            df_data = []
            for q_data in questions_list:
                df_data.append({
                    'question': q_data['question'],
                    'correct_answer': q_data['correct_answer']
                })
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            print(f"Parsed {len(df)} questions from LLM response")
            return df
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response[:200]}...")
            return pd.DataFrame(columns=['question', 'correct_answer'])
        except KeyError as e:
            print(f"Missing expected key in JSON: {e}")
            return pd.DataFrame(columns=['question', 'correct_answer']) 


    def llm_synthetic_text_generated(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: float = 0.1) -> str:
        """
        Generate synthetic rephrased text using the configured LLM model with the given prompt.
        
        Args:
            prompt: The input prompt for synthetic text generation
            max_new_tokens: Maximum number of new tokens to generate (uses model_parameters default if None)
            temperature: Sampling temperature for generation
            
        Returns:
            Generated text response from the model
        """
            
        # Use max_new_tokens from model_parameters if not provided
        if max_new_tokens is None:
            max_new_tokens = self.model_parameters.get('max_new_tokens', 4096)
        
        # Initialize the models wrapper with current parameters
        config = SettingsWrapper(_env_file=self.env_file)
        models_wrapper = ModelsWrapper(
            model_name=self.model_parameters['model_name'],
            config=config,
            answering_parameters=self.model_parameters
        )
        
        # Prepare messages in the format expected by the model
        messages_for_model = [
            {"role": "user", "content": prompt}
        ]
        
        # Generate response using the models wrapper
        try:
            generated_text, reasoning_text = models_wrapper.generate(
                messages_for_model=messages_for_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            if self.verbose:
                print(f"Generated synthetic text using {self.model_parameters['model_name']}")
                print(f"Input prompt length: {len(prompt)} characters")
                print(f"Generated response length: {len(generated_text)} characters")
                
            return generated_text
            
        except Exception as e:
            print(f"Error during synthetic text generation with {self.model_parameters['model_name']}: {e}")
            if self.verbose:
                print(f"Failed prompt: {prompt[:200]}...")  # Show first 200 chars for debugging
            return ""

    def parse_synthetic_text_response(self, response):
        """
        Parse LLM response containing list of rephrased texts for next-word prediction training
        
        Args:
            response: JSON string response from LLM containing list of rephrased text versions
            
        Returns:
            List of strings ready for next-word prediction fine-tuning
        """
        try:
            # Parse the JSON string to get list of rephrased versions
            rephrased_texts = json.loads(response)
            
            if self.verbose:
                print(f"Parsed {len(rephrased_texts)} rephrased versions from LLM response")
                for i, text in enumerate(rephrased_texts[:2]):  # Show first 2 examples
                    print(f"Version {i+1}: {text[:100]}...")
            
            return rephrased_texts
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response[:200]}...")
            return []
        except Exception as e:
            print(f"Unexpected error parsing response: {e}")
            return []