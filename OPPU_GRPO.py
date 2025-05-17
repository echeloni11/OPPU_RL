"""RFT fine‑tuning with TRL v0.8 GRPOTrainer + custom reward
----------------------------------------------------------------
This script is a **drop‑in replacement** for the old PPO version.
Only the parts that had to change are rewritten; everything else
(loader utils, evaluation, etc.) can be copied from the previous
file unchanged.

Major changes
-------------
1. **PPOTrainer ➜ GRPOTrainer**
2. **Value‑head / Advantage network removed** – GRPO does not use
   a learned value head, so we no longer wrap the base model with
   `AutoModelForCausalLMWithValueHead` nor create a reference value
   model.  Advantages are computed directly from the group‑normalised
   rewards produced by `reward_func`.
3. **reward_func signature** – GRPO calls the function with
   `(prompts, completions, completion_ids, …extras)` and expects a
   list/np.ndarray/torch.Tensor with **one scalar per completion**.
4. **Dataset format** – must contain a column **"prompt"**.  Any
   additional columns are forwarded to the reward function.
5. **Config object** – we build a small `GRPOConfig` instead of
   `PPOConfig`.

Usage
-----
```bash
python rft_grpo.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --task_name movie_tagging \
  --batch_size 8 \
  --k 2 \
  --max_epoch 2 \
  --temperature 0.7 \
  --add_profile
```
The CLI flags mirror the old script; irrelevant PPO‑specific ones
are dropped.
"""

# ‑‑ Imports -----------------------------------------------------------------
import os, json, argparse, warnings, textwrap
from typing import List

import torch
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

from trl import GRPOTrainer, GRPOConfig

# ---------- Local helpers ---------------------------------------------------
from utils import (
    split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid,
    extract_citation_title, extract_option, extract_movie, extract_news_cat,
    extract_news_headline, extract_product_review, extract_scholarly_title,
    extract_tweet_paraphrasing,
)
from rank_bm25 import BM25Okapi

# ---- Reward ----------------------------------------------------------------
from rewards import get_reward_func_for_task



# ‑‑ CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LoRA RFT with GRPO")
parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--task_name", default="movie_tagging")
parser.add_argument("--batch_size", type=int, default=4,
                    help="per‑device prompt batch size (== num groups)")
parser.add_argument("--k", type=int, default=0,
                    help="#history snippets to retrieve")
parser.add_argument("--cut_off", type=int, default=2048,
                    help="max prompt tokens")
parser.add_argument("--max_new_tokens_gen", type=int, default=64,
                    help="max tokens to generate in RL step")
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--task_lora", default=None,
                    help="path to initial task LoRA adapter (optional)")
parser.add_argument("--add_profile", action="store_true", help="add profile to the prompt")
parser.add_argument("--num_users", type=int, default=None, help="number of users to train")
args = parser.parse_args()

# ‑‑ Tokeniser & Model -------------------------------------------------------
print("Loading tokenizer …")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading initial base model (will be reloaded per user)...")
# Load the model initially to check setup, but it will be reloaded in the loop
initial_base_model_load_check = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
initial_base_model_load_check.config.pad_token_id = tokenizer.pad_token_id

# Apply **existing** task adapter ONCE if provided, to create the common starting point
# This merged model state will be recreated inside the loop for training
if args.task_lora:
    print(f"Loading initial task LoRA from {args.task_lora} and merging...")
    initial_base_model_load_check = PeftModel.from_pretrained(initial_base_model_load_check, args.task_lora)
    initial_base_model_load_check = initial_base_model_load_check.merge_and_unload()
    print("Initial task LoRA merged into the base model state for reference.")

# Clean up the initial check model to save memory before the loop
del initial_base_model_load_check
torch.cuda.empty_cache()

# # Define *new* adapter config (used per user)
# peft_cfg = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "v_proj"],
#     task_type="CAUSAL_LM",
# )

# # ‑‑ Prepare dataset (one user at a time) ------------------------------------
# with open(f"./data/{args.task_name}/user_top_100_history.json", "r") as fh:
#     users_data = json.load(fh)
if args.add_profile:
    with open(f"./data/{args.task_name}/profile_user_100.json", "r") as fh:
        profiles = json.load(fh)

with open(f"./data/{args.task_name}/user_top_100_history.json", 'r') as f:
    users_data = json.load(f)

format_flag = False
if args.task_name == "movie_tagging":
    extract_article = extract_movie
    format_flag = True
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
    format_flag = True
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
    format_flag = True
elif args.task_name == "product_rating":
    extract_article = extract_product_review
    format_flag = True
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
    format_flag = True
elif args.task_name == "tweet_paraphrase":
    extract_article = extract_tweet_paraphrasing

# --- Load the reward function ---
reward_func = get_reward_func_for_task(args.task_name)

pred_all = []
actual = []
train_data = []

prompt_template = json.load(open("./prompt/prompt.json"))

# ➜ iterate over *each* user and train an adapter ---------------------------
for uid, user in enumerate(tqdm(users_data, desc="users")):
    if args.num_users is not None and uid >= args.num_users:
        break

    print(f"\n### User {uid}")

    # ---- Reload base model for this user's training ----
    print(f"Reloading base model state for User {uid} training...")
    current_model_for_training = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    current_model_for_training.config.pad_token_id = tokenizer.pad_token_id

    # Re-apply initial task adapter if it was used, creating the starting point
    model_to_train_for_user = current_model_for_training # Start with the base model
    peft_config_for_trainer = None # Default to creating a new LoRA as per original peft_cfg

    if args.task_lora:
        print(f"Loading task LoRA from {args.task_lora} for User {uid} to fine-tune further...")
        try:
            model_to_train_for_user = PeftModel.from_pretrained(current_model_for_training, args.task_lora)
            # DO NOT MERGE. model_to_train_for_user is now a PeftModel with task_lora adapters.
            peft_config_for_trainer = None # Signal to GRPOTrainer to use existing adapters
            print(f"Successfully loaded task LoRA. Will fine-tune existing adapters for User {uid}.")
        except Exception as e:
            print(f"Warning: Failed to load task_lora from {args.task_lora} for user {uid}. Error: {e}. Will proceed to train a new LoRA if peft_cfg is defined.")
            # Fallback to using current_model_for_training (base model) and peft_cfg if task_lora loading fails
            model_to_train_for_user = current_model_for_training


    # (Optional: Prepare for k-bit training if needed, GRPOTrainer might handle this with peft_config)
    # current_model_for_training = prepare_model_for_kbit_training(current_model_for_training)


    # ---- build per‑user prompt/reference list -----------------------------
    rows = []
    history_context = []
    if args.k > 0:
        for rec in user["profile"]:
            trimmed = {k: get_first_k_tokens(str(v), 128) for k, v in rec.items()}
            history_context.append(
                prompt_template[args.task_name]["retrieval_history"].format(**trimmed)
            )

    profile_prefix = (profiles[uid]["output"] + "\n") if args.add_profile else ""

    for idx, q in enumerate(user["profile"]):
        q_trim = {k: get_first_k_tokens(str(v), 768) for k, v in q.items()}
        prompt_core = prompt_template[args.task_name]["OPPU_input"].format(**q_trim)
        full_prompt  = prompt_template[args.task_name]["OPPU_full"].format(**q_trim)

        # FIXME: reference would be different for tweet_paraphrase task
        # for tweet_paraphrase, prompt_core here is "tweet:"
        # and full_prompt is "tweet: {text}"
        # we add half of the words in the text into prompt_core
        # and use the other half as reference
        if args.task_name == "tweet_paraphrase":
            half_words = len(full_prompt.split()) // 2
            prompt_core += " " + " ".join(full_prompt.split()[:half_words])
            reference = " ".join(full_prompt.split()[half_words:])
        else:
            reference = full_prompt[len(prompt_core):].strip()

        # # retrieval (optional)
        # retrieval = ""
        # if args.k > 0 and idx > 0 and history_context:
        #     bm25 = BM25Okapi([h.split() for h in history_context[:idx]])
        #     token_q = prompt_template[args.task_name]["retrieval_query"].format(**q_trim).split()
        #     retrieval = "".join(bm25.get_top_n(token_q, history_context[:idx], n=args.k)) + "\n"

        # final_prompt = profile_prefix + retrieval + prompt_core
        if reference:
            rows.append({"prompt": prompt_core, "reference": reference})

    if not rows:
        warnings.warn(f"User {uid} skipped – no reference completions found")
        continue
    
    # TODO: need to check the correct format of prompt and reference
    ds = Dataset.from_list(rows)

    # ---- Tokenise only the prompt (no labels) -----------------------------
    # def _tok(x):
    #     tok = tokenizer(x["prompt"], max_length=args.cut_off, truncation=True)
    #     return {**tok, "reference": x["reference"]}

    # ds = ds.map(_tok, remove_columns=["prompt"], num_proc=4)

    ### It's not necessary to tokenize the prompt and reference here because GRPO will handle tokenization.

    # ---- GRPO config -------------------------------------------------------
    gcfg = GRPOConfig(
        per_device_train_batch_size=args.batch_size,
        output_dir=f"./ckpt/{args.task_name}/u{uid}",
        max_prompt_length=args.cut_off,
        max_completion_length=args.max_new_tokens_gen,
        num_generations=4,         # FIXME: for classification task, it should be 1
                                   # maybe for generation it's larger
        num_iterations=args.max_epoch,
        temperature=args.temperature,
        beta=1,                 # KL term (0 = no KL)
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=0,
        # for sampling
        top_k=10 if args.task_name in ["citation", "news_categorize", "product_rating"] else None
    )

    trainer = GRPOTrainer(
        model=model_to_train_for_user, # Pass the potentially PeftModel model
        reward_funcs=reward_func,
        args=gcfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config_for_trainer, # Will be None if task_lora was loaded, else original peft_cfg
    )

    trainer.train()

    # --- 5. Save User-Specific LoRA Adapter ---
    print(f"Saving user-specific LoRA adapter for User {uid}...")
    # The PPOTrainer modifies the 'model' in place. Save its LoRA adapter weights.
    # We save the PEFT adapter, not the whole model with value head.
    output_name = "./ckpt/{}/k{}-{}-{}-OPPU_LoRA_RFT-User{}".format(
        args.task_name, args.k, args.task_name, args.model_name.split('/')[-1], uid
    )
    trainer.model.save_pretrained(output_name) # Saves LoRA adapters + value head if configured
    # If you only want the LoRA adapter:
    # ppo_trainer.model.base_model.save_pretrained(output_name) # Saves only the PEFT adapter layers

    print(f"User {uid} LoRA adapter saved to {output_name}")

    # trainer.save_model(gcfg.output_dir)

    # --- Clean up training resources ---
    del trainer
    del current_model_for_training
    torch.cuda.empty_cache()

    # (Optional) evaluation block can be pasted unchanged from the old file
    # torch.cuda.empty_cache() # Already done above

# --- 6. Evaluation for the Current User ---
    # Prepare model for inference: Use the trained LoRA adapter.
    # Need to load the base model *without* value head and apply the *just trained* adapter.
    print(f"\nEvaluating User {uid}...")

    # ---- Reload base model state for this user's evaluation ----
    print("Reloading base model for evaluation...")
    inference_base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16, # Or inference dtype
        device_map="auto",
    )
    inference_base_model.config.pad_token_id = tokenizer.pad_token_id
    # NOTE: No need to merge the initial task_lora here.
    # The user-specific adapter was trained *on top* of that state (if it existed),
    # so loading the user-specific adapter implicitly includes the initial adaptation.

    # Load the user-specific adapter trained with GRPO
    print(f"Loading user-specific adapter from {output_name} for evaluation...")
    inference_model = PeftModel.from_pretrained(
        inference_base_model,
        output_name, # Path where the adapter was saved
        is_trainable=False # Set to false for inference
    )
    inference_model.eval() # Set to evaluation mode
    if hasattr(inference_model, 'config'):
         inference_model.config.use_cache = True # Enable caching for generation

    # Prepare test questions for this user (similar logic to SFT version)
    test_question_list = []
    question_id_list = []
    test_history_context = [] # History for eval may differ slightly if needed

    # Rebuild history context for evaluation (using all profile data)
    if args.k > 0 and format_flag:
        eval_history_list = user['profile'] # Use full profile for eval context
        for p in eval_history_list:
            p_limited = {key: get_first_k_tokens(str(value), 128) for key, value in p.items()}
            test_history_context.append(prompt_template[args.task_name]['retrieval_history'].format(**p_limited))


    for q_eval in user['query']: # Iterate through test queries for the user
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
        if args.k > 0 and format_flag and test_history_context:
            tokenized_corpus_eval = [doc.split(" ") for doc in test_history_context]
            bm25_eval = BM25Okapi(tokenized_corpus_eval)
            tokenized_query_eval = retrieval_query_text.split(" ")
            retrieved_history_eval = bm25_eval.get_top_n(tokenized_query_eval, test_history_context, n=args.k)
            retrieval_str_eval = "".join(retrieved_history_eval) + "\n"

        # Add profile context for evaluation
        profile_str_eval = ""
        if args.add_profile:
            profile_str_eval = profiles[uid]['output'] + "\n"

        # Combine components for the final evaluation prompt
        final_test_prompt = profile_str_eval + retrieval_str_eval + test_prompt_base

        test_question_list.append(final_test_prompt)
        question_id_list.append(q_eval['id'])

    # Generate predictions for the user's test queries
    test_batch_list = split_batch(test_question_list, 4) # Use smaller batch size for inference if needed
    user_out_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_batch_list, desc=f"User {uid} Inference", leave=False)):
            sentences = batch
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=args.cut_off, return_token_type_ids=False) # Use args.cut_off
            inputs = {k: v.to(inference_model.device) for k, v in inputs.items()}

            # Use standard generate, not ppo_trainer.generate
            # with torch.autocast(device_type="cuda"): # Use if inference_model uses bfloat16/float16
            outputs = inference_model.generate(
                    **inputs,
                    # Generation parameters similar to PPO or specific eval settings
                    do_sample=True,
                    top_k=5,
                    temperature=args.temperature,
                    top_p=0.2,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, # Ensure pad token is set
                    max_new_tokens=32 # Max length for evaluation outputs
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

    # --- Clean up evaluation resources ---
    del inference_model
    del inference_base_model
    torch.cuda.empty_cache()

    # Store predictions for the user
    for i in range(len(user_out_list)):
        pred_all.append({
            "id": question_id_list[i],
            "output": user_out_list[i]
            })
        # print(f"QID: {question_id_list[i]}, Pred: {user_out_list[i]}") # Optional: print predictions

# --- Save final combined results (Same as before) ---
output_file = {
    'task': name2taskid[args.task_name],
    'golds': pred_all, # Note: Renamed from 'golds', maybe rename file too? 'predictions' might be better.
    'model': args.model_name,
}

import os
output_dir = './output_final/{}/'.format(args.task_name)
os.makedirs(output_dir, exist_ok=True)

if args.add_profile:
    with open('{}/output-OPPU-k{}-{}-{}-profile.json'.format(output_dir, args.k, args.task_name, args.model_name.split('/')[-1]), 'w') as f: # Use args.task_name and args.model_name
        json.dump(output_file, f, indent=4)
else:
    with open('{}/output-OPPU_RL-k{}-{}-{}.json'.format(output_dir, args.k, args.task_name, args.model_name.split('/')[-1]), 'w') as f: # Use args.task_name and args.model_name
        json.dump(output_file, f, indent=4)
print("Finished RFT training and evaluation.")
