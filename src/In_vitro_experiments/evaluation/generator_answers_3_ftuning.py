import torch
import json
import gc
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, PeftModel, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
import datetime
import inspect
import pandas as pd
from typing import Union
from In_vitro_experiments.evaluation.generator_answers_1_prompting import commonsense_test, generate_answers_func, generate_evaluation_func, generate_chronological_func
from pathlib import Path
import glob
from torch.utils.data import DataLoader
from accelerate import Accelerator


def _tensorboard_dir_kwargs(tensorboard_log_dir):
    # transformers 5 removed TrainingArguments(logging_dir=...); it reads TENSORBOARD_LOGGING_DIR instead.
    if not tensorboard_log_dir:
        return {}
    if "logging_dir" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"logging_dir": tensorboard_log_dir}
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_log_dir)
    return {}



def upload_ftuning_input(client, ftuning_input_filepath):
    client.files.create(file=open(ftuning_input_filepath, "rb"), purpose="fine-tune")
    print(f"File {ftuning_input_filepath} uploaded")
    return 0

def retrieve_fileid(client, jsonl_filename):
    file_listing = client.files.list()
    print(file_listing)
    corresponding_ids =[x.id for x in file_listing if x.filename == jsonl_filename]
    if len(corresponding_ids) == 0:
        print(jsonl_filename)
        raise ValueError('file not found, please first upload it with the `answering_parameters["ftuning_need_upload"] = True` argument')
    if len(corresponding_ids) > 1:
        print('length of the matching jsonl should be exactly one?')
    return corresponding_ids[0]

def find_latest_checkpoint(checkpoint_dir):
        """Find the latest checkpoint directory"""
        checkpoint_dir = str(checkpoint_dir)  # Convert PosixPath to string
        
        # Extract config pattern and find highest chapter
        config_pattern = checkpoint_dir.rsplit('max_chapter_', 1)[0] + 'max_chapter_*'
        
        chapter_dirs = [d for d in glob.glob(config_pattern) if os.path.isdir(d)]
        
        if not chapter_dirs:
            return None
        
        # Filter out directories that don't have valid chapter numbers and sort
        valid_chapter_dirs = []
        for d in chapter_dirs:
            try:
                chapter_num = int(d.split('max_chapter_')[1])
                valid_chapter_dirs.append((chapter_num, d))
            except ValueError:
                # Skip directories with invalid chapter numbers
                continue
        
        if not valid_chapter_dirs:
            print("No valid chapter directories found")
            return None
        
        # Sort by chapter number (highest first)
        valid_chapter_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Find highest chapter that contains checkpoints
        for _, checkpoint_dir in valid_chapter_dirs:
            checkpoints = [cp for cp in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")) if os.path.isdir(cp)]
            if checkpoints:
                break
        else:
            return None
        
        # Ordina per numero di checkpoint
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[1]))
        
        latest_checkpoint = checkpoints[-1]
        print(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint

def setup_and_resume_lora_training(model_name, output_dir, input_dataset, answering_parameters,  answer_path,checkpoint_to_continue=None, tensorboard_log_dir=None):
    """
    Setup and resume LoRA training from checkpoint or start from scratch.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_dir (str): Directory to save checkpoints and final model
        input_dataset (Dataset): Training dataset
        answering_parameters (dict): Parameters including 'n_epochs'
        tensorboard_log_dir (str): Directory for tensorboard logs
        answer_path (str): Path for existing LoRA adapters (optional)
    
    Returns:
        dict: Training results and status
    """
    
    # Early check: if LoRA adapters already exist, skip everything
    if answer_path:
        lora_adapters_path = Path(answer_path) / "lora_adapters"
        if lora_adapters_path.exists() and any(lora_adapters_path.iterdir()):
            #print(f"LoRA adapters already exist at: {lora_adapters_path}, skipping all training")
            return {
                "status": "skipped",
                "training_status": "already_exists",
                "message": f"LoRA adapters already exist"               
            }
    else:
        print(f"LoRA adapters not found at: {lora_adapters_path},can't skipping all training")
        
    
    
    try:

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        if tensorboard_log_dir is not None:
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        # Find checkpoint and setup training
        latest_checkpoint = find_latest_checkpoint(output_dir)
        target_epochs = answering_parameters['n_epochs']
        max_chapter = answering_parameters.get('max_chapter', 'no_max_chapter')        
        
        if latest_checkpoint:           
            print(f"Will resume training until epoch {target_epochs}")
           # print(f"Loading tokenizer from saved model: {output_dir}")          
        else:
            print(checkpoint_to_continue)
            if checkpoint_to_continue is not None and os.path.exists(checkpoint_to_continue) and os.path.isdir(checkpoint_to_continue):
                latest_checkpoint = checkpoint_to_continue
                print(f"HEREEEE !!!!!  CONTINUE TRAINING FROM A GIVEN CHECKPOINT: {latest_checkpoint}")
            else:
                print(f"Starting fresh training for {target_epochs} epochs")
                print(f'Debug model name = {model_name} bs{answering_parameters.get('batch_size')},lr{answering_parameters.get('learning_rate_multiplier')},ep{target_epochs}')
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        model_size_match = re.search(r'(\d+)B', model_name.upper())

        if model_size_match:
            model_size_b = int(model_size_match.group(1))
            is_large_model = model_size_b >= 10 
            print(f"Detected {model_size_b}B model - {'LARGE' if is_large_model else 'SMALL'}")
        else:
            # Fallback: assume small model if we can't detect size
            is_large_model = False
            print(f"Could not detect model size from name '{model_name}', assuming SMALL model")
        
        if is_large_model:
            print("Using Accelerate for large model training...")
            return train_with_accelerate(input_dataset, answering_parameters, output_dir, 
                                        latest_checkpoint, target_epochs, max_chapter, 
                                        tokenizer, model_name, answer_path, tensorboard_log_dir)
        else:
            print("Using standard Trainer for small model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,           
            
        )
        
        # Setup LoRA model
        if latest_checkpoint:
            print(f"Loading existing LoRA adapters from checkpoint: {latest_checkpoint}")
            model = PeftModel.from_pretrained(base_model, latest_checkpoint)
        elif max_chapter == 1 or max_chapter == "no_max_chapter":
            print("Creating new LoRA configuration")
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(base_model, lora_config)
        else:
            print("Continuing training, adding a new chapter")

            previous_model = answering_parameters['previous_model']
            lora_path = str(answer_path)                                  
            lora_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', lora_path)
            lora_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', lora_path)
            lora_path = lora_path + "/lora_adapters"
            model = PeftModel.from_pretrained(base_model, lora_path)
        
        # Enable training mode for LoRA adapters
        model.train()
        for param in model.parameters():
            param.requires_grad = False
            
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Tokenize dataset
        max_sequence_length = answering_parameters['max_new_tokens']
        tokenized_dataset = input_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_sequence_length),
            batched=True,
            remove_columns=input_dataset.column_names
        )
         
        # Training arguments - hardcoded configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=target_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=answering_parameters.get('batch_size', 8),
            warmup_steps=100,
            logging_steps=10,
            save_steps=answering_parameters.get('save_steps', 50),  # Add this parameter
            save_total_limit=answering_parameters.get('save_total_limit', 500),  # Make this configurable too
            learning_rate=answering_parameters.get('learning_rate_multiplier', 'auto'),
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            report_to=["tensorboard"] if tensorboard_log_dir else None,
            **_tensorboard_dir_kwargs(tensorboard_log_dir),
            save_strategy=answering_parameters.get('save_strategy', 'epochs'),  # Make this configurable
            logging_strategy="steps",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Print training info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        
        # Start/Resume training
        if latest_checkpoint:
            #trainer.train(resume_from_checkpoint=latest_checkpoint) changed for GridSearch
            training_status = "resumed"
            trainer.train()
        else:
            trainer.train()
            training_status = "fresh_start"
        
        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        trainer.save_state()
        
        # Save only LoRA adapters to answer_path        
        if answer_path:
            os.makedirs(answer_path, exist_ok=True)
            
            # Save LoRA adapters in subfolder
            lora_adapters_path = Path(answer_path) / "lora_adapters"
            
            # Save LoRA adapters (we know they don't exist because of early check)
            model.save_pretrained(lora_adapters_path)
            print(f"LoRA adapters saved to: {lora_adapters_path}")
            
            # Save tokenizer for inference
            tokenizer.save_pretrained(lora_adapters_path)
            print(f"Tokenizer saved to: {lora_adapters_path}")
            
            # Save input dataset sample
            sample_data_path = Path(answer_path) / "sample_training_data.json"
            
            # Check if sample data file already exists
            if not sample_data_path.exists():
                sample_data_to_save = input_dataset.to_list()
                with open(sample_data_path, "w") as f:
                    json.dump(sample_data_to_save, f, indent=2)
                print(f"Sample training data saved to: {sample_data_path}")
            else:
                print(f"Sample training data already exists at: {sample_data_path}, skipping save")
                
            print(f"Training completed! Model saved to: {output_dir}")
        
        return {
            "status": "completed",
            "training_status": training_status,
            "final_epoch": target_epochs,
            "model_path": output_dir,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params
        }
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def tokenize_function_causal_text(examples, tokenizer, max_sequence_length):
    """
    Tokenizes plain text examples for causal language modeling (next-word prediction).
    No chat template - just direct text tokenization.
    """
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
        return_tensors="pt"
    )
    
    # For causal LM, labels = input_ids (model handles the shifting internally)
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()
    
    return tokenized_output

def tokenize_function(examples, tokenizer, max_sequence_length):
    """
    Tokenizes the input examples for training a Causal LM.
    Applies chat template, truncates, and pads.
    """
    formatted_chats = []
    for single_example_messages in examples["messages"]:
        if not isinstance(single_example_messages, list):
            print(f"ERROR: Expected 'messages' to be a list of dictionaries, but found type {type(single_example_messages)} for: {single_example_messages}")
            formatted_chat = "" # Fallback
        else:
            try:
                formatted_chat = tokenizer.apply_chat_template(
                    single_example_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"ERROR: Failed to apply chat template to messages: {single_example_messages}. Error: {e}")
                formatted_chat = ""
        formatted_chats.append(formatted_chat)

    tokenized_output = tokenizer(
        formatted_chats,
        truncation=True,
        padding="max_length", # Ensure consistent padding to max_length for map function
        max_length=max_sequence_length,
        return_tensors="pt"
    )
   
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()

    return tokenized_output


    
def setup_tensorboard_logging(lora_weights_pathname, run_name=None):
    """
    Set up TensorBoard logging directory using the existing LoRA weights path structure.
    
    Args:
        lora_weights_pathname: Path object from ftuning_input_filepath_func
        run_name: Optional custom run name, otherwise uses timestamp
    
    Returns:
        str: Path to TensorBoard log directory
    """
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"lora_finetuning_{timestamp}"
    
    tensorboard_log_dir = lora_weights_pathname / "tensorboard_logs" / run_name
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    
    
    return str(tensorboard_log_dir)


def cleanup_memory():
    """
    Cleans up model and trainer objects and clears CUDA cache.
    """
    # These variables need to be global for cleanup to work if not passed as args
    global model, trainer, tokenized_dataset, raw_dataset, tokenizer
    
    if 'trainer' in globals() and trainer is not None:
        del trainer
    if 'model' in globals() and model is not None:
        del model
    if 'tokenized_dataset' in globals():
        del tokenized_dataset
    if 'raw_dataset' in globals():
        del raw_dataset
    if 'tokenizer' in globals():
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        #print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        #print("GPU memory cache cleared after training and saving.")
    else:
        return
        #print("No CUDA GPU available. Skipping CUDA cache clearing.")
    #print("\nScript finished.")


def train_with_accelerate(input_dataset, answering_parameters, output_dir, 
                         latest_checkpoint, target_epochs, max_chapter, 
                         tokenizer, model_name, answer_path, tensorboard_log_dir):
    """Training function for large models using model parallelism"""   
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=answering_parameters.get('batch_size', 8),
        mixed_precision='fp16',
    )
    
    if accelerator.is_main_process:
        # Create explicit device map (matching working snippet)
        num_gpus = torch.cuda.device_count()
        # Add proper layer detection for different model sizes
        if "70b" in model_name.lower():
            layers = 80
        elif "8b" in model_name.lower() or "8B" in model_name:
            layers = 32
        elif "13b" in model_name.lower() or "13B" in model_name:
            layers = 40
        else:
            layers = 40  # fallback
        
        device_map = {}
        layers_per_gpu = layers // num_gpus
        remainder = layers % num_gpus
        
        # Distribute embed tokens
        device_map["model.embed_tokens"] = 0
        
        # Distribute layers across GPUs
        current_layer = 0
        for gpu in range(num_gpus):
            # Some GPUs get an extra layer if remainder > 0
            gpu_layers = layers_per_gpu + (1 if gpu < remainder else 0)
            
            for layer_offset in range(gpu_layers):
                if current_layer < layers:
                    device_map[f"model.layers.{current_layer}"] = gpu
                    current_layer += 1
        
        # Put final layers on last GPU
        device_map["model.norm"] = num_gpus - 1
        device_map["lm_head"] = num_gpus - 1
        
        print("📍 Device mapping:")
        gpu_counts = {}
        for name, gpu in device_map.items():
            gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        for gpu, count in gpu_counts.items():
            print(f"  GPU {gpu}: {count} components")
        
        # Load base model with device mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        
        # Verify actual distribution
        print("✅ Actual model distribution:")
        actual_distribution = {}
        for name, param in base_model.named_parameters():
            device = str(param.device)
            actual_distribution[device] = actual_distribution.get(device, 0) + 1
        
        total_params = sum(actual_distribution.values())
        for device, count in actual_distribution.items():
            percentage = (count / total_params) * 100
            print(f"  {device}: {count} parameters ({percentage:.1f}%)")
        
        # Check memory usage after loading
        print("💾 Memory usage after model loading:")
        for gpu in range(num_gpus):
            allocated = torch.cuda.memory_allocated(gpu) / 1e9
            reserved = torch.cuda.memory_reserved(gpu) / 1e9
            total = torch.cuda.get_device_properties(gpu).total_memory / 1e9
            print(f"  GPU {gpu}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        
        # Setup LoRA model (same logic as original)
        if latest_checkpoint:
            print(f"Loading existing LoRA adapters from checkpoint: {latest_checkpoint}")
            model = PeftModel.from_pretrained(base_model, latest_checkpoint)
        elif max_chapter == 1 or max_chapter == "no_max_chapter":
            print("Creating new LoRA configuration")
           
            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(base_model, lora_config)
        else:
            print("Continuing training, adding a new chapter")
            previous_model = answering_parameters['previous_model']
            lora_path = str(answer_path)
            lora_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', lora_path)
            lora_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', lora_path)
            lora_path = lora_path + "/lora_adapters"
            model = PeftModel.from_pretrained(base_model, lora_path)
        
        # Enable training mode (same as original)
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Tokenize dataset (same as original)
        max_sequence_length = answering_parameters.get('max_new_tokens_generation', 1024)
        tokenized_dataset = input_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_sequence_length),
            batched=True,
            remove_columns=input_dataset.column_names
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        batch_size = answering_parameters.get('batch_size', 8)
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
        
        # Setup optimizer and training parameters
        lr = answering_parameters.get('learning_rate_multiplier', 'auto')
        if lr == 'auto':
            lr = 1e-5
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        # Training parameters
        save_steps = 500
        logging_steps = 10
        
        # Calculate total steps
        total_steps = len(dataloader) * target_epochs
        print(f"Starting training: {total_steps} total steps, {target_epochs} epochs")
        
        # Manual training loop
        model.train()
        step = 0
        
        if latest_checkpoint:
            training_status = "resumed"
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            training_status = "fresh_start"
            print("Starting fresh training")
        
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_param_count:,}")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        for epoch in range(target_epochs):
            print(f"Epoch {epoch + 1}/{target_epochs}")
            
            for batch in dataloader:
                step += 1
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Logging
                if step % logging_steps == 0:
                    print(f"Step {step}: Loss = {loss.item():.4f}")
                    if writer:  # ADD THIS LINE
                        writer.add_scalar('train/loss', loss.item(), step)  # ADD THIS LINE
                
                # Save checkpoint
                if step % save_steps == 0:
                    checkpoint_dir = f"{output_dir}/checkpoint-{step}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"Checkpoint saved: {checkpoint_dir}")
        
        print("Training completed!")
        
        # Save final model (same as original)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapters (same as original)
        if answer_path:
            os.makedirs(answer_path, exist_ok=True)
            lora_adapters_path = Path(answer_path) / "lora_adapters"
            model.save_pretrained(lora_adapters_path)
            tokenizer.save_pretrained(lora_adapters_path)
            
            sample_data_path = Path(answer_path) / "sample_training_data.json"
            if not sample_data_path.exists():
                sample_data_to_save = input_dataset.to_list()
                with open(sample_data_path, "w") as f:
                    json.dump(sample_data_to_save, f, indent=2)
        
        return {
            "status": "completed",
            "training_status": training_status,
            "final_epoch": target_epochs,
            "model_path": output_dir,
            "trainable_parameters": trainable_param_count,
            "total_parameters": sum(p.numel() for p in model.parameters())
        }
    
    else:
        accelerator.wait_for_everyone()
        return {"status": "worker_process"}

def setup_and_resume_full_finetuning(model_name, output_dir, input_dataset, answering_parameters, answer_path, checkpoint_to_continue=None, tensorboard_log_dir=None):
    """
    Setup and resume full fine-tuning from checkpoint or start from scratch.
    """
    
    # Early check: if model already exists, skip everything
    if answer_path:
        full_model_path = Path(answer_path) / "full_finetuning_model"
        if full_model_path.exists() and any(full_model_path.iterdir()):
            return {
                "status": "skipped",
                "training_status": "already_exists",
                "message": "Full fine-tuned model already exists"               
            }
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        if tensorboard_log_dir is not None:
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        latest_checkpoint = find_latest_checkpoint(output_dir)
        target_epochs = answering_parameters['n_epochs']
        max_chapter = answering_parameters.get('max_chapter', 'no_max_chapter')
        
        if not latest_checkpoint and checkpoint_to_continue and os.path.exists(checkpoint_to_continue):
            print(f"CONTINUING FROM THIS CHECKPOINT:{checkpoint_to_continue}")
            latest_checkpoint = checkpoint_to_continue
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if latest_checkpoint:
            model = AutoModelForCausalLM.from_pretrained(
                latest_checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "resumed"
        elif max_chapter == 1 or max_chapter == "no_max_chapter":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "fresh_start"
        else:
            previous_model = answering_parameters['previous_model']
            previous_model_path = str(answer_path)
            previous_model_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', previous_model_path)
            previous_model_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', previous_model_path)
            previous_model_path = previous_model_path + "/full_finetuning_model"
            model = AutoModelForCausalLM.from_pretrained(
                previous_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "continued_from_previous_chapter"
        
        # Enable all parameters for training
        for param in model.parameters():
            param.requires_grad = True
        
        # Tokenize dataset
        max_sequence_length = answering_parameters['max_new_tokens']
        tokenized_dataset = input_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_sequence_length),
            batched=True,
            remove_columns=input_dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=target_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=answering_parameters.get('batch_size', 8),
            warmup_steps=100,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            learning_rate=answering_parameters.get('learning_rate_multiplier', 'auto'),
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            report_to=["tensorboard"] if tensorboard_log_dir else None,
            **_tensorboard_dir_kwargs(tensorboard_log_dir),
            save_strategy="steps",
            logging_strategy="steps",
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, data_collator=data_collator)
        
        trainer.train()
        
        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        trainer.save_state()
        
        # Save to answer_path
        if answer_path:
            os.makedirs(answer_path, exist_ok=True)
            full_model_path = Path(answer_path) / "full_finetuning_model"
            model.save_pretrained(full_model_path)
            tokenizer.save_pretrained(full_model_path)
            
            sample_data_path = Path(answer_path) / "sample_training_data.json"
            if not sample_data_path.exists():
                with open(sample_data_path, "w") as f:
                    json.dump(input_dataset.to_list(), f, indent=2)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            "status": "completed",
            "training_status": training_status,
            "final_epoch": target_epochs,
            "model_path": output_dir,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params
        }
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
### EWC implementation here:
    



def compute_fisher_matrix(model, dataset, tokenizer, max_sequence_length):
    """
    Compute Fisher Information Matrix on a dataset.
    
    Args:
        model: The trained model
        dataset: Dataset to compute Fisher on (current chapter's data)
        tokenizer: Tokenizer
        max_sequence_length: Max sequence length for tokenization
    
    Returns:
        fisher_dict: Dictionary mapping parameter names to Fisher values
    """
    print("Computing Fisher Information Matrix...")
    """Compute Fisher Information Matrix on a dataset."""
    print(f"\n{'='*60}")
    print("🔍 DEBUG: Starting Fisher Computation")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Max sequence length: {max_sequence_length}")
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'No device map'}")
    
    # Check GPU memory before
    import torch
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    model.eval()
    
    # Initialize Fisher dictionary with zeros on same device as parameters
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_sequence_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=1, 
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Compute Fisher by accumulating squared gradients
    num_samples = 0
    print(f"Processing {len(dataloader)} samples for Fisher computation...")
    
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(dataloader)} samples...")
        
        # Move batch to model's device (works with device_map="auto")
        if hasattr(model, 'device'):
            batch = {k: v.to(model.device) for k, v in batch.items()}
        
        model.zero_grad()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass to get gradients
        loss.backward()
        
        # Accumulate squared gradients (Fisher approximation)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        num_samples += 1
    
    # Average Fisher over samples
    for name in fisher_dict:
        fisher_dict[name] /= num_samples
    
    print(f"Fisher computation complete on {num_samples} samples")
    model.train()
    return fisher_dict


def save_ewc_state(model, fisher_dict, save_path):
    """
    Save EWC state (Fisher matrix and optimal parameters).
    
    Args:
        model: Current model
        fisher_dict: Fisher Information Matrix
        save_path: Directory to save EWC state
    """
    ewc_path = Path(save_path) / "ewc_state"
    os.makedirs(ewc_path, exist_ok=True)
    
    # Save Fisher matrix
    torch.save(fisher_dict, ewc_path / "fisher_dict.pt")
    
    # Save optimal parameters (current model's parameters)
    optimal_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            optimal_params[name] = param.data.clone()
    
    torch.save(optimal_params, ewc_path / "optimal_params.pt")
    
    print(f"EWC state saved to: {ewc_path}")


def load_ewc_state(load_path, model):
    """
    Load EWC state (Fisher matrix and optimal parameters).
    
    Args:
        load_path: Directory containing EWC state
        model: Current model (to ensure parameters are on correct devices)
    
    Returns:
        fisher_dict: Fisher Information Matrix
        optimal_params: Previous optimal parameters
    """
    ewc_path = Path(load_path) / "ewc_state"
    
    if not ewc_path.exists():
        print(f"No EWC state found at: {ewc_path}")
        return None, None
    
    print(f"Loading EWC state from: {ewc_path}")
    
    # Load Fisher and optimal params
    fisher_dict = torch.load(ewc_path / "fisher_dict.pt")
    optimal_params = torch.load(ewc_path / "optimal_params.pt")
    
    # Ensure Fisher and optimal params are on same devices as current model parameters
    # This is crucial for multi-GPU setups
    fisher_dict_corrected = {}
    optimal_params_corrected = {}
    
    for name, param in model.named_parameters():
        if name in fisher_dict:
            fisher_dict_corrected[name] = fisher_dict[name].to(param.device)
            optimal_params_corrected[name] = optimal_params[name].to(param.device)
    
    print(f"EWC state loaded successfully")
    return fisher_dict_corrected, optimal_params_corrected


class EWCTrainer(Trainer):
    """
    Custom Trainer that adds EWC penalty to the loss.
    """
    def __init__(self, *args, ewc_lambda=0, fisher_dict=None, optimal_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict
        self.optimal_params = optimal_params
        
        # Log EWC configuration
        if self.fisher_dict is not None:
            print(f"\n{'='*60}")
            print(f"EWC ENABLED")
            print(f"  Lambda: {self.ewc_lambda}")
            print(f"  Protecting {len(self.fisher_dict)} parameters from previous chapter")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"EWC DISABLED (First chapter or no previous state)")
            print(f"{'='*60}\n")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with EWC penalty.
        """
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add EWC penalty if we have Fisher matrix from previous chapter
        if self.fisher_dict is not None and self.ewc_lambda > 0:
            ewc_penalty = 0.0
            
            for name, param in model.named_parameters():
                if name in self.fisher_dict and param.requires_grad:
                    # Get Fisher and optimal param for this parameter
                    fisher = self.fisher_dict[name]
                    optimal = self.optimal_params[name]
                    
                    # Compute penalty: Fisher * (current - optimal)^2
                    # All tensors are already on correct device (multi-GPU safe!)
                    ewc_penalty += (fisher * (param - optimal) ** 2).sum()
            
            # Add weighted penalty to loss
            total_loss = loss + (self.ewc_lambda / 2.0) * ewc_penalty
            
            # Optional: log EWC penalty for monitoring
            if self.state.global_step % self.args.logging_steps == 0:
                print(f"Step {self.state.global_step}: "
                      f"CE_Loss={loss.item():.4f}, "
                      f"EWC_penalty={ewc_penalty.item():.4f}, "
                      f"Total_Loss={total_loss.item():.4f}")
            
            loss = total_loss
        
        return (loss, outputs) if return_outputs else loss


def setup_and_resume_ewc_training(
    model_name, 
    output_dir, 
    input_dataset, 
    answering_parameters, 
    answer_path,
    checkpoint_to_continue=None, 
    tensorboard_log_dir=None
):
    """
    Setup and resume EWC training from checkpoint or start from scratch.
    Implements Online EWC: Fisher computed only on previous chapter's data.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_dir (str): Directory to save checkpoints and final model
        input_dataset (Dataset): Current chapter's training dataset
        answering_parameters (dict): Parameters including:
            - 'n_epochs': number of epochs
            - 'ewc_lambda': EWC regularization strength (default: 5000)
            - 'max_chapter': current chapter number
            - 'previous_model': dict with previous chapter info (epochs, chapter)
            - 'batch_size': gradient accumulation steps
            - 'learning_rate_multiplier': learning rate
            - 'max_new_tokens': max sequence length
        answer_path (str): Path to save final model and EWC state
        checkpoint_to_continue (str): Path to checkpoint to continue from
        tensorboard_log_dir (str): Directory for tensorboard logs
    
    Returns:
        dict: Training results and status
    """
    
    # Early check: if model already exists, skip everything
    if answer_path:
        ewc_model_path = Path(answer_path) / "ewc_model"
        if ewc_model_path.exists() and any(ewc_model_path.iterdir()):
            return {
                "status": "skipped",
                "training_status": "already_exists",
                "message": "EWC model already exists"               
            }
    
    try:
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        if tensorboard_log_dir is not None:
            os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # Get training parameters
        latest_checkpoint = find_latest_checkpoint(output_dir)
        target_epochs = answering_parameters['n_epochs']
        max_chapter = answering_parameters.get('max_chapter', 'no_max_chapter')
        ewc_lambda = answering_parameters.get('ewc_lambda', 5000)  # Default: 5000
        
        # Handle checkpoint continuation
        if not latest_checkpoint and checkpoint_to_continue and os.path.exists(checkpoint_to_continue):
            print(f"CONTINUING FROM THIS CHECKPOINT: {checkpoint_to_continue}")
            latest_checkpoint = checkpoint_to_continue
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize Fisher and optimal params (will be loaded if not first chapter)
        fisher_dict = None
        optimal_params = None
        
        # Load model based on chapter
        if latest_checkpoint:
            # Resume from checkpoint within current chapter
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                latest_checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "resumed"
            
            # Load EWC state if exists (for chapters > 1)
            if max_chapter > 1:
                previous_model = answering_parameters.get('previous_model')
                if previous_model:
                    # Get previous chapter's answer_path
                    previous_answer_path = str(answer_path)
                    previous_answer_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', previous_answer_path)
                    previous_answer_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', previous_answer_path)
                    fisher_dict, optimal_params = load_ewc_state(previous_answer_path, model)
                else:
                    # Try loading from current path if previous_model not specified
                    fisher_dict, optimal_params = load_ewc_state(answer_path, model)
        
        elif max_chapter == 1 or max_chapter == "no_max_chapter":
            # First chapter - start fresh
            print(f"Starting fresh training for Chapter 1")
            print(f"No EWC penalty will be applied (first chapter)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "fresh_start"
        
        else:
            # Continuing from previous chapter - load previous model and EWC state
            print(f"Continuing to Chapter {max_chapter} with EWC protection")
            previous_model = answering_parameters['previous_model']
            previous_model_path = str(answer_path)
            previous_model_path = re.sub(r'epochs_\d+', f'epochs_{previous_model["epochs"]}', previous_model_path)
            previous_model_path = re.sub(r'max_chapter_\d+', f'max_chapter_{previous_model["chapter"]}', previous_model_path)
            previous_model_path = previous_model_path + "/ewc_model"
            
            print(f"Loading model from: {previous_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                previous_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            training_status = "continued_from_previous_chapter"
            
            # Load EWC state from previous chapter
            previous_answer_path = re.sub(r'/ewc_model$', '', previous_model_path)
            fisher_dict, optimal_params = load_ewc_state(previous_answer_path, model)
            
            if fisher_dict is None:
                print("WARNING: Could not load EWC state from previous chapter!")
                print(f"Expected location: {previous_answer_path}/ewc_state")
                print("Training will proceed without EWC penalty!")
        
        # Enable all parameters for training (full fine-tuning)
        for param in model.parameters():
            param.requires_grad = True
        
        # Tokenize dataset
        max_sequence_length = answering_parameters['max_new_tokens']
        tokenized_dataset = input_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_sequence_length),
            batched=True,
            remove_columns=input_dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=target_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=answering_parameters.get('batch_size', 8),
            warmup_steps=100,
            logging_steps=10,
            save_steps=answering_parameters.get('save_steps', 200),
            save_total_limit=answering_parameters.get('save_total_limit', 3),
            learning_rate=answering_parameters.get('learning_rate_multiplier', 2e-5),
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=True,
            report_to=["tensorboard"] if tensorboard_log_dir else None,
            **_tensorboard_dir_kwargs(tensorboard_log_dir),
            save_strategy=answering_parameters.get('save_strategy', 'epoch'),
            logging_strategy="steps",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Create EWC Trainer with Fisher matrix and optimal params from previous chapter
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            ewc_lambda=ewc_lambda,
            fisher_dict=fisher_dict,
            optimal_params=optimal_params,
        )
        
        # Print training info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"  Chapter: {max_chapter}")
        print(f"  Epochs: {target_epochs}")
        print(f"  EWC Lambda: {ewc_lambda}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Total params: {total_params:,}")
        print(f"{'='*60}\n")
        
        # Train
        trainer.train()
        
        # Save model to output_dir
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        trainer.save_state()
        
        # Compute Fisher matrix on CURRENT chapter's data for next chapter
        print("\n" + "="*60)
        print("POST-TRAINING: Computing Fisher for next chapter")
        print("="*60)
        current_fisher = compute_fisher_matrix(
            model, 
            input_dataset,  # Use current chapter's data
            tokenizer, 
            max_sequence_length
        )
        
        # Save to answer_path
        if answer_path:
            os.makedirs(answer_path, exist_ok=True)
            
            # Save model
            ewc_model_path = Path(answer_path) / "ewc_model"
            model.save_pretrained(ewc_model_path)
            tokenizer.save_pretrained(ewc_model_path)
            print(f"Model saved to: {ewc_model_path}")
            
            # Save EWC state (Fisher + optimal params for NEXT chapter)
            save_ewc_state(model, current_fisher, answer_path)
            
            # Save sample training data
            sample_data_path = Path(answer_path) / "sample_training_data.json"
            if not sample_data_path.exists():
                with open(sample_data_path, "w") as f:
                    json.dump(input_dataset.to_list(), f, indent=2)
                print(f"Sample training data saved to: {sample_data_path}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"  Final model: {output_dir}")
        print(f"  EWC state saved for next chapter")
        print(f"{'='*60}\n")
        
        return {
            "status": "completed",
            "training_status": training_status,
            "final_epoch": target_epochs,
            "model_path": output_dir,
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "ewc_lambda": ewc_lambda,
        }
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR DURING EWC TRAINING")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise