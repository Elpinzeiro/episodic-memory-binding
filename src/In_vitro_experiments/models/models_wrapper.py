# epbench/src/models/models_wrapper.py

import os
from typing import Dict, Any, Optional, Tuple, Union, List # <--- Add List here
from pathlib import Path
import torch # For torch_dtype and device_map

# Assuming these are available or will be installed
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import google.generativeai as genai
import requests


class ModelsWrapper:
    def __init__(self, model_name: str, config: Any, answering_parameters: Dict=None, lora_path=None):
        self.model_name = model_name
        self.config = config
        self.answering_parameters = answering_parameters

        self.client = None
        self.hf_model: Optional[PeftModel] = None # Or AutoModelForCausalLM
        self.hf_tokenizer: Optional[AutoTokenizer] = None
        self.is_local_hf_model = False # Flag to indicate if it's a local HF model
        
        
        if answering_parameters and lora_path is not None:
            base_model_name = self.answering_parameters.get('model_name')
            max_new_tokens_param = self.answering_parameters.get('max_new_tokens_generation', 1024) # Default for safety
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(lora_path)
                self.hf_tokenizer.model_max_length = 8192 # Set to Llama 3's context window

            except Exception:
                print(f"Warning: Could not load tokenizer from {lora_path}. Loading from base model: {base_model_name}")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            base_hf_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16, # IMPORTANT: Match the dtype used during fine-tuning
                device_map="auto" # Load model on available devices (e.g., GPU)
            )
            # Load LoRA adapters onto base model
            self.hf_model = PeftModel.from_pretrained(base_hf_model, lora_path)
            self.hf_model.eval() # Set model to evaluation mode

            # Ensure pad_token_id is set for both tokenizer and model
            if self.hf_tokenizer.pad_token_id is None:
                if self.hf_tokenizer.eos_token_id is not None:
                    self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id
                else: 
                    # Fallback if neither pad_token nor eos_token is defined
                    self.hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    # Resize embeddings only if adding a new token that wasn't there before
                    self.hf_model.resize_token_embeddings(len(self.hf_tokenizer)) 

            if self.hf_model.config.pad_token_id is None:
                self.hf_model.config.pad_token_id = self.hf_tokenizer.pad_token_id
                print(f"Setting model.config.pad_token_id to tokenizer.pad_token_id: {self.hf_tokenizer.pad_token_id}")
            
            self.is_local_hf_model = True
            print("ModelsWrapper: Local Hugging Face fine-tuned model loaded.")
    

        elif answering_parameters and self.answering_parameters.get('kind') == 'ftuning-manual':
            base_model_name = self.answering_parameters.get('model_name')
            max_new_tokens_param = self.answering_parameters.get('max_new_tokens_generation', 1024)
            
            # Default to 'lora' for backward compatibility
            ftuning_type = self.answering_parameters.get('ftuning_type', 'lora')
            adapter_path_str = answering_parameters.get('after_finetuning_path')
            # Check if we should load from a specific checkpoint
            infer_from_chkpt = self.answering_parameters.get('infer_from_chkpt', None)

            if infer_from_chkpt is not None:
                # Load directly from the specified checkpoint path
                model_path = Path(infer_from_chkpt)
                print(f"Load from checkpoint path: {model_path}")
            else:
                # Original logic for backward compatibility
                if ftuning_type == 'lora':
                    model_path = Path(adapter_path_str) / "lora_adapters"
                    print(f"Load from this path: {model_path}")
                elif ftuning_type == 'full':
                    model_path = Path(adapter_path_str) / "full_finetuning_model"
                    print(f"Load from this path: {model_path}")
                else:
                    raise ValueError(f"Invalid ftuning_type: {ftuning_type}. Must be 'lora' or 'full'")
                
            
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # Load tokenizer
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.hf_tokenizer.model_max_length = 8192
            except Exception:
                print(f"Warning: Could not load tokenizer from {model_path}. Loading from base model: {base_model_name}")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load model based on ftuning_type
            if ftuning_type == 'lora':
                base_hf_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                self.hf_model = PeftModel.from_pretrained(base_hf_model, model_path)
            else:  # full
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            self.hf_model.eval()
            
            # Ensure pad_token_id is set
            if self.hf_tokenizer.pad_token_id is None:
                if self.hf_tokenizer.eos_token_id is not None:
                    self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id
                else:
                    self.hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.hf_model.resize_token_embeddings(len(self.hf_tokenizer))
            
            if self.hf_model.config.pad_token_id is None:
                self.hf_model.config.pad_token_id = self.hf_tokenizer.pad_token_id
            
            self.is_local_hf_model = True

        else:
            # --- Initialize API Client ---
            if model_name.startswith('openrouter/'):
                # any OpenRouter model, e.g. openrouter/google/gemini-2.5-flash (key OPENROUTER_API_KEY in .env)
                from openai import OpenAI
                if not config.OPENROUTER_API_KEY:
                    raise ValueError("OPENROUTER_API_KEY not found in .env for an openrouter/ model.")
                self.client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
            elif "claude" in model_name:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            elif model_name.startswith('gpt'):
                from openai import OpenAI
                api_key = config.OPENAI_API_KEY
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in .env for GPT.")
                self.client = OpenAI(api_key=api_key)
            elif "gemini" in model_name:
                if not config.GOOGLE_API_KEY:
                    raise ValueError("GOOGLE_API_KEY not found in .env for Gemini.")
                genai.configure(api_key=config.GOOGLE_API_KEY, transport='rest')
                self.client = genai.GenerativeModel(model_name)
            else:
                raise ValueError(f"ModelsWrapper: Unsupported API model_name or missing 'kind' handling: {model_name}. "
                                 "If it's a local model, ensure 'kind' is 'ftuning-manual' and parameters are correct.")
            
            print(f"ModelsWrapper: Initialized API client for {model_name}.")

    def generate(self, *args, **kwargs):
        """
        Unified generate function that handles both old and new signatures:
        
        Old signature (API): generate(user_prompt, system_prompt, full_outputs, max_new_tokens, temperature, keep_reasoning)
        New signature (Local): generate(messages_for_model, max_new_tokens, temperature) -> Tuple[str, Optional[str]]
        """
        
        # Check if it's the old signature by looking at the first argument type
        if args and isinstance(args[0], str):
            # Old signature - delegate to generate_from_api
            return self.generate_from_api(*args, **kwargs)
        elif args and isinstance(args[0], list):
            # New signature - delegate to generate_from_local_finetuned
            return self.generate_from_local_finetuned(*args, **kwargs)
        else:
            # Handle keyword-only calls
            if 'messages_for_model' in kwargs:
                # New signature called with kwargs
                return self.generate_from_local_finetuned(
                    kwargs['messages_for_model'],
                    kwargs.get('max_new_tokens'),
                    kwargs.get('temperature', 0.1)
                )
            else:
                # Old signature called with kwargs - delegate to generate_from_api
                return self.generate_from_api(**kwargs)

    def generate_from_api(self, user_prompt: str = "Who are you?", system_prompt: str = "You are a content event generator assistant.", 
                        full_outputs = False, max_new_tokens: int = 256, temperature: float = 1.0, keep_reasoning = False):
        """Original API-based generate function"""
        
        reasoning = None
        token_param = {}
        if self.model_name.startswith('openrouter/'):
            outputs = self.client.chat.completions.create(
                model=self.model_name[len('openrouter/'):],
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            return outputs if full_outputs else outputs.choices[0].message.content
        if "gpt" in self.model_name: 
            if "gpt-5" in self.model_name:
                token_param['max_completion_tokens'] = max_new_tokens
            else:
                token_param['max_tokens'] = max_new_tokens           
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ],
                **token_param,  # Unpack the appropriate parameter
                temperature=temperature
            )

            if not full_outputs:
                outputs = outputs.choices[0].message.content
        
        elif "o1" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_prompt
                }
                ]
            )
            # max_completion_tokens = max_new_tokens,
            # temperature = temperature

            if not full_outputs:
                outputs = outputs.choices[0].message.content
                #print(outputs)
        elif "o3" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_prompt
                }
                ]
            )
            # max_completion_tokens = max_new_tokens,
            # temperature = temperature

            if not full_outputs:
                outputs = outputs.choices[0].message.content
                #print(outputs) 
        elif "claude" in self.model_name:
            outputs = self.client.messages.create(
                model=self.model_name,
                system = system_prompt, # different syntax compared to openai
                messages=[
                    {"role": "user", "content": user_prompt
                }
                ],
                max_tokens = max_new_tokens,
                temperature = temperature
            )

            if not full_outputs:
                outputs = outputs.content[0].text

        elif "gemini" in self.model_name:
        # Handle Gemini with system prompt properly
            if system_prompt:
                # Combine system prompt with user prompt for Gemini
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                combined_prompt = user_prompt
                
            try:
                # Generate content using the SDK
                response = self.client.generate_content(
                    combined_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                )
                
                if not full_outputs:
                    # Extract just the text content
                    outputs = response.text
                else:
                    outputs = response
                    
                return outputs
                
            except Exception as e:
                print(f"Gemini SDK error: {e}")
                raise e
        
        elif "llama" in self.model_name:
            outputs = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                data=json.dumps({
                    "model": "meta-llama/" + self.model_name,
                    "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ],
                    "provider": {"order": ["Fireworks"]} # working with large context, contrary to the others
                })
                )
            
            if not full_outputs:
                raw_string = outputs.text
                cleaned_string = raw_string.strip()
                print(cleaned_string)
                parsed_dict = json.loads(cleaned_string)
                if not 'choices' in parsed_dict:
                    print(parsed_dict)
                outputs = parsed_dict['choices'][0]['message']['content']

            ## below (replicate):
            #outputs = self.client.run(
            #    "meta/" + self.model_name, # "meta/llama-2-13b",
            #    input={
            #        #"top_k": 0,
            #        #"top_p": 0.95,
            #        "prompt": user_prompt,
            #        #"max_tokens": 512,
            #        # "temperature": temperature, # take the default temperature
            #        "system_prompt": system_prompt,
            #        #"length_penalty": 1,
            #        "max_new_tokens": max_new_tokens
            #        #"stop_sequences": "<|end_of_text|>,<|eot_id|>",
            #        #"prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            #        #"presence_penalty": 0,
            #        #"log_performance_metrics": False
            #    })
            #
            #if not full_outputs:
            #    outputs = "".join(outputs)
        elif "deepseek" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name, # deepseek-reasoner # deepseek-chat
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt
                }
                ],
                stream=False
            )

            '''
            outputs = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                data=json.dumps({
                    "model": "deepseek/" + self.model_name,
                    "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ],
                    "provider": {"order": ["Nebius", "Fireworks", "Kluster"], 
                                "ignore": ["Avian", "Together", "Novita", "DeepInfra", "Featherless", "DeepSeek"]},
                    "include_reasoning": True # Include the parameter for downstream processing
                })
            )
            '''

            #"provider": {, "ignore": ["Avian.io", "Together", "NovitaAI", "DeepInfra", "Featherless", "DeepSeek"]}, 
            # working with large context too

            print(outputs)

            if not full_outputs:
                # specific for deepseek, with objects, not list
                if "deepseek-reasoner" in self.model_name:
                    reasoning = outputs.choices[0].message.reasoning_content
                outputs = outputs.choices[0].message.content

            '''
            if not full_outputs:
                raw_string = outputs.text
                cleaned_string = raw_string.strip()
                print(cleaned_string)
                parsed_dict = json.loads(cleaned_string)
                if not 'choices' in parsed_dict:
                    print(parsed_dict)
                outputs = parsed_dict['choices'][0]['message']['content']
                reasoning = parsed_dict['choices'][0]['message']['reasoning']
            '''
        else:
            raise ValueError("there is no generate function for this model name")
        
        if keep_reasoning:
            return outputs, reasoning

        return outputs 

    def generate_from_local_finetuned(self, messages_for_model: List[Dict], max_new_tokens: int, temperature: float = 0.1) -> Tuple[str, Optional[str]]:
        """
        Generates text using either the loaded local HF model or the initialized API client.
        Args:
            messages_for_model: List of message dictionaries (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
        Returns:
            A tuple of (generated_text, reasoning_text). Reasoning might be None.
        """
        if self.is_local_hf_model:
            if self.hf_model is None or self.hf_tokenizer is None:
                raise RuntimeError("Hugging Face model or tokenizer not loaded correctly.")
            
            full_input_str = self.hf_tokenizer.apply_chat_template(
                messages_for_model, tokenize=False, add_generation_prompt=True
            )
            inputs = self.hf_tokenizer(
                full_input_str, return_tensors="pt", truncation=True, 
                max_length=self.hf_tokenizer.model_max_length
            ).to(self.hf_model.device)

            with torch.no_grad():
                generated_tokens = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0, # Only sample if temperature > 0
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id
                )
            
            generated_text = self.hf_tokenizer.decode(
                generated_tokens[0, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            return generated_text, None # No separate reasoning from simple HF generation typically

        elif self.client:
            print(f"ModelsWrapper: Performing API inference with {self.model_name}...")
            system_prompt = next((msg['content'] for msg in messages_for_model if msg['role'] == 'system'), "")
            user_prompt = next((msg['content'] for msg in messages_for_model if msg['role'] == 'user'), "")
            
            try:
                if self.model_name.startswith('claude'):
                    message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_new_tokens, # Use passed max_new_tokens
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    generated_text = message.content[0].text
                    return generated_text, None # Or extract reasoning if Claude format allows
                
                elif self.model_name.startswith('gpt'):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages_for_model, # Pass messages directly for OpenAI
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    generated_text = response.choices[0].message.content
                    return generated_text, None # Or extract reasoning if GPT format allows
                
                # Add other API clients here
                elif "gemini" in self.model_name:
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [{"parts": [{"text": user_prompt}]}],
                        "generationConfig": {"maxOutputTokens": max_new_tokens}
                    }
                    response = self.client.post(
                        f"{self.gemini_url}?key={self.google_api_key}",
                        headers=headers,
                        json=data
                    )
                    generated_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return generated_text, None
                else:
                    raise NotImplementedError(f"API client for {self.model_name} not implemented in generate method.")
            except Exception as e:
                print(f"Error during API generation with {self.model_name}: {e}")
                return "", None # Return empty on error
        else:
            raise RuntimeError("No model or API client initialized in ModelsWrapper.")