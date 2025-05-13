import torch
import torch.nn as nn
# import bitsandbytes as bnb # Keep if you were using quantization
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from transformers import pipeline, BitsAndBytesConfig # Keep if needed
import argparse
from rank_bm25 import BM25Okapi
# --- RL Imports ---
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
# --- End RL Imports ---
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM # No longer needed for SFT
import transformers # Still needed for utility data collator potentially, or args if reused
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training # Keep prepare_model_for_kbit_training if using quantization

# --- Add reward calculation library ---
from rewards import reward_func
# ---

parser = argparse.ArgumentParser(description="Parser for LoRA RFT") # Changed description
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
# --- Keep existing args ---
parser.add_argument('--batch_size', type=int, default=16) # This will now be PPO batch size
parser.add_argument('--k', type=int, default=0)
# parser.add_argument('--max_step', type=int, default=5000) # Less relevant for PPO user-specific training, use epochs
# FIXME: cut_off may need larger
parser.add_argument('--cut_off', type=int, default=512) # PPO often uses shorter sequences, adjust if needed
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.7) # Generation temperature
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--task_lora', type=str, default='./ckpt/movie_tagging/k0-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt') # Initial LoRA weights
parser.add_argument('--access_token', type=str, default=None)
# --- Add RFT specific args ---
parser.add_argument('--ppo_epochs', type=int, default=4, help="Number of optimisation epochs per PPO batch")
parser.add_argument('--lr', type=float, default=1.41e-5, help="Learning rate for PPO")
parser.add_argument('--init_kl_coef', type=float, default=0.2, help="Initial KL penalty coefficient")
parser.add_argument('--adap_kl_ctrl', type=bool, default=True, help="Use adaptive KL control")
# FIXME: target_kl larger or smaller?
parser.add_argument('--target_kl', type=float, default=0.02, help="Target KL value for adaptive control") # Adjusted from 0.1, common value is higher
parser.add_argument('--mini_batch_size', type=int, default=4, help="PPO mini batch size")
parser.add_argument('--max_new_tokens_gen', type=int, default=50, help="Max new tokens to generate during RFT") # Control generation length
parser.add_argument('--gradient_accumulation_steps', type=int, default=1) # Add gradient accumulation if needed

args = parser.parse_args()
model_name = args.model_name
task_name = args.task_name
ppo_batch_size = args.batch_size # Reuse batch_size arg for PPO batch size
k = args.k
cutoff_len = args.cut_off
add_eos_token = False # Usually handled by generation config
max_epoch = args.max_epoch # Epochs over user profile data

# --- Quantization Config (Keep if you were using it) ---
# bnb_config = ...

# --- Tokenizer Setup (Same as before) ---
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
tokenizer.eos_token = "</s>"
# IMPORTANT: Set pad token USED by the model. For Llama usually not needed or set explicitly
# Check model config if PAD is defined. If not, potentially use EOS as PAD.
# Let's assume the base model handles padding or we set it to EOS if needed.
if tokenizer.pad_token is None:
    print("Tokenizer has no pad token, setting it to EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    print(f"Using existing pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")


# --- Load Base Model (Same as before, potentially without quantization for RFT simplicity first) ---
# Consider loading without quantization initially for easier debugging of RFT.
# Add quantization back later if needed and compatible with trl/value head.
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config, # Add back if needed
    local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, # Or float16/float32
    pad_token_id=tokenizer.pad_token_id, # Ensure model knows pad token id
    token=args.access_token

)


# --- Model Config (Same as before) ---
# base_model.config.use_cache = False # Will be handled by PPO / generation config
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id

# --- Prepare for k-bit training (Keep if using quantization) ---
# base_model.gradient_checkpointing_enable()
# base_model = prepare_model_for_kbit_training(base_model)


# --- Load initial Task LoRA weights ---
# Load the adapter onto the base model *before* creating the Value Head model
print(f"Loading initial LoRA adapter from: {args.task_lora}")
# Ensure the base model loaded here is the same architecture as the one used for task_lora training
# We are *not* setting is_trainable=False here, as we want to train it further with PPO
try:
    base_model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora)
    print("Successfully loaded initial LoRA adapter.")
    # Merge the initial adapter? Optional, might simplify state management.
    # If merged, the 'base_model' now incorporates the task LoRA.
    # If not merged, the PeftModel structure is kept. Let's try *not* merging first.
    base_model = base_model.merge_and_unload()
    print("Merged initial LoRA adapter.")
except Exception as e:
    print(f"Could not load initial LoRA adapter from {args.task_lora}. Error: {e}")
    print("Proceeding with base model weights only for LoRA target modules.")
    # If loading fails, we will initialize a *new* LoRA adapter below based on peft_config

# --- LoRA Config for PPO fine-tuning ---
# We define the LoRA config again to ensure it's applied correctly,
# especially if loading the initial adapter failed or if we want to re-apply it
# to potentially different modules or with different settings for PPO.
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"], # Ensure these match your initial LoRA if continuing training
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

ppo_config = PPOConfig(
    # ---- 通用 -----------------------------------------------------------
    learning_rate       = args.lr,             # 比 SFT 通常再小 5–10×
    batch_size          = ppo_batch_size,       # 一次 rollout 采样多少 (= prompt 数)
    mini_batch_size     = max(1, ppo_batch_size // 2),
    num_ppo_epochs      = args.ppo_epochs,                # ← 旧字段 ppo_epochs
    gradient_accumulation_steps = 1,        # 继续沿用你的设置
    remove_unused_columns       = False,

    # ---- KL 相关 --------------------------------------------------------
    kl_coef             = args.init_kl_coef,              # ← 旧字段 kl_penalty
    # target KL 如需硬阈值，可在 trainer init 再加 early_stopping_kl_threshold
    kl_estimator        = "k1",             # 默认即可

    # ---- PPO & Advantage ------------------------------------------------
    cliprange           = 0.2,
    vf_coef             = 1.0,              # value‑loss 系数
    cliprange_value     = 0.2,
    gamma               = 1.0,              # γ = 1 即无折扣（常见做法）
    lam                 = 0.95,             # λ for GAE
    whiten_rewards      = False,            # rule‑based reward 通常不需要
)
# --- Generation Kwargs for PPO ---
# Control how responses are generated during PPO training
generation_kwargs = {
    "top_k": 10,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": args.max_new_tokens_gen, # Control output length during RL
    "temperature": args.temperature,
}
output_length_sampler = LengthSampler(generation_kwargs["max_new_tokens"] // 2, generation_kwargs["max_new_tokens"]) # Sample length dynamically


# --- Data Loading (Same as before) ---
with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
    test_data = json.load(f) # This seems to be the user data source

format_flag = False
if args.task_name == "movie_tagging":
    extract_article = extract_movie
    format_flag = True
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
    format_flag = True
# ... (rest of the elif conditions for extract_article)
elif args.task_name == "tweet_paraphrase":
    extract_article = extract_tweet_paraphrasing

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

if args.add_profile:
    with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)

# --- Helper to extract reference completion ---
def extract_reference(full_prompt, prompt):
    # Simple extraction assuming completion follows prompt immediately
    if full_prompt.startswith(prompt):
        return full_prompt[len(prompt):].strip()
    else:
        # Fallback or more robust extraction needed if structure varies
        print(f"Warning: Prompt not found at the start of full_prompt.")
        # Try to find the target based on task structure if needed
        # For now, return empty string if basic extraction fails
        return ""


# --- Training Loop (Replaces SFT Trainer) ---
from datasets import Dataset # Keep Dataset import

# Evaluation variables (same as before)
pred_all = []
# actual = [] # 'actual' wasn't used in the original eval loop snippet


# --- Main Loop Over Users ---
for user_idx, user_data in enumerate(tqdm(test_data, desc="Processing Users")):
    print(f"\n--- Processing User {user_idx} ---")

    # --- 1. Prepare Model for User ---
    # Create a fresh value head model for each user, starting from the potentially task-LoRA-loaded base model
    # This ensures the value head is trained per user along with the LoRA adapter
    print("Creating model with Value Head...")
    value_head_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, # This is either the original base or base+initial LoRA
        peft_config=peft_config, # Apply LoRA config HERE
        torch_dtype=torch.bfloat16, # Match base model dtype
        # token=args.access_token # May not be needed if base_model is already loaded
        # Add quantization config here if AutoModelForCausalLMWithValueHead supports it and you need it
    )
    value_head_model.generation_config = base_model.generation_config # Add this line
    print_trainable_parameters(value_head_model) # Check LoRA parameters are trainable

    # Create a reference model for KL divergence calculation
    # This model should *not* be trained during PPO
    print("Creating reference model...")
    ref_model = create_reference_model(value_head_model)
    ref_model.eval() # Set ref model to eval mode


    # --- 2. Prepare User Data for PPO ---
    user_profile_data = []
    profile_str = ""
    if args.add_profile:
        profile_str = test_profile[user_idx]['output'] + "\n" # Get user profile string

    history_context = [] # Store history strings for BM25/retrieval for this user

    # Build history context first if k > 0
    if k > 0 and format_flag:
        visible_history_list = user_data['profile']
        for p in visible_history_list:
             # Limit history token length
            for key, value in p.items():
                p[key] = get_first_k_tokens(str(value), 128) # Shorter limit for history items
            history_context.append(prompt_template[args.task_name]['retrieval_history'].format(**p))

    # Prepare training data points (prompts and references)
    for idx, q in enumerate(user_data['profile']):
        # Limit input token length
        q_limited = {key: get_first_k_tokens(str(value), 768) for key, value in q.items()}

        # Base prompt and full prompt (input + expected output)
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q_limited)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q_limited)

        # Add retrieval context if k > 0
        retrieval_str = ""
        if k > 0 and idx > 0 and format_flag and history_context:
            # Use history built up *before* this item
            current_history = history_context[:idx]
            if current_history:
                tokenized_corpus = [doc.split(" ") for doc in current_history]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q_limited).split(' ')
                retrieved_history = bm25.get_top_n(tokenized_query, current_history, n=args.k)
                retrieval_str = "".join(retrieved_history) + "\n"

        # Add profile and retrieval strings
        final_prompt = profile_str + retrieval_str + prompt

        # Extract reference completion
        reference_completion = extract_reference(full_prompt, prompt) # Get the yu_i part

        if not reference_completion:
            print(f"Skipping data point {idx} due to empty reference completion.")
            continue

        user_profile_data.append(
            {
                "query": final_prompt, # This is the input xu_i for generation
                "reference": reference_completion # This is the target yu_i for reward
            }
        )

    if not user_profile_data:
        print(f"Skipping User {user_idx} due to no valid training data points.")
        continue

    # Convert to Dataset
    user_dataset = Dataset.from_list(user_profile_data)

    def tokenize_query(element):
        # FIXME: padding=False???
        tokens = tokenizer(element["query"], truncation=True, max_length=cutoff_len, padding=False)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], 
                "reference": element["reference"]}

    tokenized_dataset = user_dataset.map(tokenize_query, batched=False)
    tokenized_dataset.set_format(type="torch") # Ensure output is PyTorch tensors

    # --- 3. Instantiate PPO Trainer for User ---
    # Need a data collator for padding during batching by PPOTrainer
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=base_model,
        ref_model=None,
        reward_model=RewardModel(task_name=args.task_name),
        value_model=value_head_model,
        train_dataset=tokenized_dataset, # Provide dataset for potential internal shuffling/sampling
        data_collator=data_collator, # Used for padding batches
    )

    ppo_trainer.train()

    # --- 5. Save User-Specific LoRA Adapter ---
    print(f"Saving user-specific LoRA adapter for User {user_idx}...")
    # The PPOTrainer modifies the 'model' in place. Save its LoRA adapter weights.
    # We save the PEFT adapter, not the whole model with value head.
    output_name = "./ckpt/{}/k{}-{}-{}-OPPU_LoRA_RFT-User{}".format(
        args.task_name, args.k, args.task_name, model_name.split('/')[-1], user_idx
    )
    ppo_trainer.model.save_pretrained(output_name) # Saves LoRA adapters + value head if configured
    # If you only want the LoRA adapter:
    # ppo_trainer.model.base_model.save_pretrained(output_name) # Saves only the PEFT adapter layers

    print(f"User {user_idx} LoRA adapter saved to {output_name}")

    # --- 6. Evaluation for the Current User ---
    # Prepare model for inference: Use the trained LoRA adapter.
    # Need to load the base model *without* value head and apply the *just trained* adapter.
    print(f"Evaluating User {user_idx}...")

    inference_base_model = base_model
    # Load the user-specific adapter trained with PPO
    inference_model = PeftModel.from_pretrained(
        inference_base_model,
        output_name, # Path where the adapter was saved
        is_trainable=False # Set to false for inference
    )
    # Optional: Merge for potentially faster inference
    # inference_model = inference_model.merge_and_unload()

    inference_model.eval() # Set to evaluation mode
    if hasattr(inference_model, 'config'):
         inference_model.config.use_cache = True # Enable caching for generation

    # Prepare test questions for this user (similar logic to SFT version)
    test_question_list = []
    question_id_list = []
    test_history_context = [] # History for eval may differ slightly if needed

    # Rebuild history context for evaluation (using all profile data)
    if k > 0 and format_flag:
        eval_history_list = user_data['profile'] # Use full profile for eval context
        for p in eval_history_list:
            p_limited = {key: get_first_k_tokens(str(value), 128) for key, value in p.items()}
            test_history_context.append(prompt_template[args.task_name]['retrieval_history'].format(**p_limited))


    for q_eval in user_data['query']: # Iterate through test queries for the user
        q_eval_limited = q_eval # Assume query 'input' is already reasonably sized, or apply get_first_k_tokens
        
        # Extract article/key info from the query input
        if args.task_name == 'citation':
            test_question = q_eval_limited['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt_base = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)
            retrieval_query_text = test_article # Use article for retrieval query
        else:
            test_question = q_eval_limited['input']
            test_article = extract_article(test_question)
            if test_article is None: # Handle case where extraction fails
                 print(f"Warning: Could not extract article/key info for query ID {q_eval['id']}. Using full input.")
                 test_article = test_question # Fallback
            test_prompt_base = prompt_template[args.task_name]['prompt'].format(test_article)
            retrieval_query_text = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article)


        # Add retrieval context for evaluation
        retrieval_str_eval = ""
        if k > 0 and format_flag and test_history_context:
            tokenized_corpus_eval = [doc.split(" ") for doc in test_history_context]
            bm25_eval = BM25Okapi(tokenized_corpus_eval)
            tokenized_query_eval = retrieval_query_text.split(" ")
            retrieved_history_eval = bm25_eval.get_top_n(tokenized_query_eval, test_history_context, n=args.k)
            retrieval_str_eval = "".join(retrieved_history_eval) + "\n"

        # Add profile context for evaluation
        profile_str_eval = ""
        if args.add_profile:
            profile_str_eval = test_profile[user_idx]['output'] + "\n"

        # Combine components for the final evaluation prompt
        final_test_prompt = profile_str_eval + retrieval_str_eval + test_prompt_base

        test_question_list.append(final_test_prompt)
        question_id_list.append(q_eval['id'])

    # Generate predictions for the user's test queries
    test_batch_list = split_batch(test_question_list, 4) # Use smaller batch size for inference if needed
    user_out_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_batch_list, desc=f"User {user_idx} Inference", leave=False)):
            sentences = batch
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=cutoff_len, return_token_type_ids=False)
            inputs = {k: v.to(inference_model.device) for k, v in inputs.items()}

            # Use standard generate, not ppo_trainer.generate
            # with torch.autocast(device_type="cuda"): # Use if inference_model uses bfloat16/float16
            outputs = inference_model.generate(
                    **inputs,
                    # Generation parameters similar to PPO or specific eval settings
                    do_sample=True,
                    top_k=10,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, # Ensure pad token is set
                    max_new_tokens=200 # Max length for evaluation outputs
            )

            # Decode outputs, excluding the prompt part
            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Remove prompt from generated output
            cleaned_outputs = []
            for i in range(len(out_sentence)):
                 # Be careful with tokenization differences; find the prompt reliably
                 prompt_length = len(tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True))
                 # Simple slicing (might be inaccurate if special tokens affect length)
                 # A more robust way is to use the input length from generate output if available
                 # generated_part = tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                 # Fallback using string replacement (less reliable)
                 output_full = out_sentence[i]
                 prompt_text = sentences[i] # The original prompt string
                 if output_full.startswith(prompt_text):
                      cleaned_outputs.append(output_full[len(prompt_text):].strip())
                 else:
                      # If prompt isn't exactly at start, maybe just return full output or log warning
                      print(f"Warning: Prompt not found at start of output for QID {question_id_list[len(user_out_list) + i]}. Using full output.")
                      cleaned_outputs.append(output_full.strip()) # Keep full output as fallback

            user_out_list.extend(cleaned_outputs)


    # Store predictions for the user
    for i in range(len(user_out_list)):
        pred_all.append({
            "id": question_id_list[i],
            "output": user_out_list[i]
            })
        # print(f"QID: {question_id_list[i]}, Pred: {user_out_list[i]}") # Optional: print predictions

    # --- Clean up memory for next user ---
    del model, ref_model, ppo_trainer, inference_model, inference_base_model
    torch.cuda.empty_cache()


# --- Save final combined results (Same as before) ---
output_file = {
    'task': name2taskid[args.task_name],
    'golds': pred_all, # Note: Renamed from 'golds', maybe rename file too? 'predictions' might be better.
    'model': model_name,
}

if args.add_profile:
    with open('./output/{}/output-OPPU-k{}-{}-{}-profile.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
        json.dump(output_file, f, indent=4)
else:
    with open('./output/{}/output-OPPU_RL-k{}-{}-{}.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
        json.dump(output_file, f, indent=4)
print("Finished RFT training and evaluation.")