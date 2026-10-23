import anthropic
from openai import OpenAI
import json
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import random


def get_next_available_number(output_dir):
    """Find the next available number for output files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 1  # Start from 1
    
    # Get all JSON files and extract numbers
    existing_numbers = []
    for file in output_dir.glob("*.json"):
        try:
            num = int(file.stem)
            existing_numbers.append(num)
        except ValueError:
            continue
    
    if not existing_numbers:
        return 1
    
    return max(existing_numbers) + 1


def find_prompt_file(prompts_dir):
    """Find a random available .md file in the prompts directory."""
    prompts_dir = Path(prompts_dir)
    
    # Get all .md files starting with "prompt_"
    md_files = list(prompts_dir.glob("prompt_*.md"))
    
    if not md_files:
        return None
    
    # Sort by name and pick a random one
    md_files.sort()
    i = random.randint(0, len(md_files) - 1)  # Fixed: was len(md_files) which could be out of bounds
    
    return md_files[i]


def parse_prompt_with_opus(prompt_path, output_path):
    """
    Read a prompt from a file, send it to Claude Opus 4.5 with web search enabled,
    and save the response.
    
    Args:
        prompt_path: Path to the prompt file (.md)
        output_path: Path to save the JSON response
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Initialize the Anthropic client
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    # Read the prompt from the file
    print(f"Reading prompt from: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    print(f"Prompt length: {len(prompt_text)} characters")
    
    # Make the API call to Claude Opus 4.5 with web search enabled
    print("Calling Claude Opus 4.5 with web search enabled...")
    
    try:
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=12000,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search"
                }
            ]
        )
        
        # Extract the response
        response_text = message.content[-1].text  # Generally the last block, but may vary
        response_json = json.loads(response_text)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the response to JSON
        print(f"Saving response to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, ensure_ascii=False)
        
        print("✓ Success!")
        print(f"  Input tokens: {message.usage.input_tokens}")
        print(f"  Output tokens: {message.usage.output_tokens}")
        
        return True
        
    except anthropic.APIError as e:
        print(f"API Error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Response was not valid JSON. Saving raw response...")
        # Save raw response if it's not valid JSON
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"raw_response": response_text}, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def parse_prompt_with_gpt5(prompt_path, output_path):
    """
    Read a prompt from a file, send it to GPT-5 with web search, and save the response.
    
    Args:
        prompt_path: Path to the prompt file (.md)
        output_path: Path to save the JSON response
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Read the prompt from the file
    print(f"Reading prompt from: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    print(f"Prompt length: {len(prompt_text)} characters")
    
    # Make the API call to GPT-5 with web search enabled
    print("Calling GPT-5 with web search enabled...")
    
    try:
        response = client.responses.create(
            model="gpt-5",
            tools=[{"type": "web_search"}],
            input=prompt_text
        )
        
        # Extract the response
        response_text = response.output_text
        response_json = json.loads(response_text)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the response to JSON
        print(f"Saving response to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, ensure_ascii=False)
        
        print("✓ Success!")
        if hasattr(response, 'usage'):
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"API Error: {e}")
        # Try to save raw response if available
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"error": str(e), "raw_response": response_text if 'response_text' in locals() else None}, f, indent=2, ensure_ascii=False)
        except:
            pass
        return False
    

def move_processed_file(prompt_path, executed_dir):
    """Move the processed prompt file to the executed_claude directory."""
    executed_dir = Path(executed_dir)
    executed_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_path = Path(prompt_path)
    destination = executed_dir / prompt_path.name
    
    print(f"Moving {prompt_path.name} to executed_claude folder...")
    shutil.move(str(prompt_path), str(destination))
    print(f"✓ Moved to: {destination}")


def main():
    load_dotenv()
    
    # Define base paths - UPDATE THESE TO YOUR LOCAL PATHS and MODEL USED
    base_dir = Path("./generated_prompts")
    prompts_dir = base_dir
    output_dir = base_dir / "test_cases_claude"
    executed_dir = base_dir / "executed_claude"
    
    print("=" * 60)
    print("Autonomous Claude Prompt Processor")
    print("=" * 60)
    
    # Find the next available prompt file
    prompt_path = find_prompt_file(prompts_dir)
    
    if prompt_path is None:
        print("❌ No prompt files found in the generated_prompts folder!")
        print("Looking for files matching pattern: prompt_*.md")
        exit(1)
    
    print(f"\n📄 Found prompt file: {prompt_path.name}")
    
    # Get the next available number
    next_num = get_next_available_number(output_dir)
    output_path = output_dir / f"{next_num}.json"
    
    print(f"💾 Will save as: {next_num}.json")
    print()
    
    # Process the prompt
    success = parse_prompt_with_opus(prompt_path, output_path)
    
    # If successful, move the file
    if success:
        print()
        move_processed_file(prompt_path, executed_dir)
        print("\n" + "=" * 60)
        print("✅ All done! Ready for next prompt.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Processing failed. File not moved.")
        print("=" * 60)
        exit(1)


if __name__ == '__main__':
    for i in range(10):  # Set to desired number of iterations
        main()