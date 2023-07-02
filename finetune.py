"""
    llama-4b trainer with support of Stanford Alpaca-like JSON datasets (short for SAD)
    Intended to use with https://github.com/johnsmith0031/alpaca_lora_4bit

    SAD structure:
    [
        {
            "instruction": "Give null hypothesis",
            "input": "6 subjects were given a drug (treatment group) and an additional 6 subjects a placebo (control group).",
            "output": "Drug is equivalent of placebo"
        },
        {
            "instruction": "What does RNA stand for?",
            "input": "",
            "output": "RNA stands for ribonucleic acid."
        }
    ]
"""
import time

import transformers.models.llama.modeling_llama

# Early load config to replace attn if needed
from arg_parser import get_config
ft_config = get_config()

from monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_gptq_lora_model
replace_peft_model_with_gptq_lora_model()

from monkeypatch.llama_attn_hijack_xformers import hijack_llama_attention
hijack_llama_attention()


# def replace_llama_rope_with_scaled_rope():
# transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = ScaledRotaryEmbedding



if ft_config.flash_attention:
    from monkeypatch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
elif ft_config.xformers:
    from monkeypatch.llama_attn_hijack_xformers import hijack_llama_attention
    hijack_llama_attention()
    
    
# Call the rope replace function here
from monkeypatch.llama_rope_scaled_monkey_patch import replace_llama_rope_with_scaled_rope
replace_llama_rope_with_scaled_rope()
    

import autograd_4bit
if ft_config.backend.lower() == 'triton':
    autograd_4bit.switch_backend_to('triton')
else:
    autograd_4bit.switch_backend_to('cuda')



import sys
import os

import peft
import peft.tuners.lora

import wandb
import torch
import transformers
from autograd_4bit import load_llama_model_4bit_low_ram
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel, set_peft_model_state_dict

import matmul_utils_4bit
matmul_utils_4bit.faster = False


"""
Backs up and restores a settings file to Dropbox.
This is an example app for API v2.
Copied from https://github.com/dropbox/dropbox-sdk-python/blob/main/example/back-up-and-restore/backup-and-restore-example.py
"""
if ft_config.dropbox_token:
    import dropbox
    from dropbox.files import WriteMode
    from dropbox.exceptions import ApiError, AuthError

    TOKEN = ft_config.dropbox_token
    if ft_config.name:
        BACKUPPATH = "/" + ft_config.name + "/"
    else:
        BACKUPPATH = '/dump/'

    # Uploads contents of LOCALFILE to Dropbox
    def backup(LOCALFILE):
        with open(LOCALFILE, 'rb') as f:
            # We use WriteMode=overwrite to make sure that the settings in the file
            # are changed on upload
            print("Uploading " + LOCALFILE + " to Dropbox as " + BACKUPPATH + "...")
            try:
                dbx.files_upload(f.read(), BACKUPPATH, mode=WriteMode('overwrite'))
            except ApiError as err:
                # This checks for the specific error where a user doesn't have
                # enough Dropbox space quota to upload this file
                if (err.error.is_path() and
                        err.error.get_path().reason.is_insufficient_space()):
                    sys.exit("ERROR: Cannot back up; insufficient space.")
                elif err.user_message_text:
                    print(err.user_message_text)
                    sys.exit()
                else:
                    print(err)
                    sys.exit()

    # Restore the local and Dropbox files to a certain revision
    def restore(rev=None):
        # Restore the file on Dropbox to a certain revision
        print("Restoring " + BACKUPPATH + " to revision " + rev + " on Dropbox...")
        dbx.files_restore(BACKUPPATH, rev)

        # Download the specific revision of the file at BACKUPPATH to LOCALFILE
        print("Downloading current " + BACKUPPATH + " from Dropbox, overwriting " + LOCALFILE + "...")
        dbx.files_download_to_file(LOCALFILE, BACKUPPATH, rev)

    # Look at all of the available revisions on Dropbox, and return the oldest one
    def select_revision():
        # Get the revisions for a file (and sort by the datetime object, "server_modified")
        print("Finding available revisions on Dropbox...")
        entries = dbx.files_list_revisions(BACKUPPATH, limit=30).entries
        revisions = sorted(entries, key=lambda entry: entry.server_modified)

        for revision in revisions:
            print(revision.rev, revision.server_modified)

        # Return the oldest revision (first entry, because revisions was sorted oldest:newest)
        return revisions[0].rev


    class CustomTrainer(transformers.Trainer):
        def _tune_save_checkpoint(self):
            if not self.use_tune_checkpoints:
                return
            with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
                output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                self.save_model(output_dir, _internal_call=True)
                if self.args.should_save:
                    self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                print(output_dir)
                backup(output_dir + "/adapter_config.json")
                backup(output_dir + "/adapter_model.bin")
                backup(output_dir + "/optimizer.pt")
                backup(output_dir + "/rng_state.pth")
                backup(output_dir + "/scheduler.pt")
                backup(output_dir + "/trainer_state.json")
                backup(output_dir + "/training_args.bin")

    if len(TOKEN) == 0:
        sys.exit("ERROR: Looks like you didn't add your access token. "
            "Open up backup-and-restore-example.py in a text editor and "
            "paste in your token in line 14.")

    # Create an instance of a Dropbox class, which can make requests to the API.
    print("Creating a Dropbox backup object...")
    with dropbox.Dropbox(TOKEN) as dbx:

        # Check that the access token is valid
        try:
            dbx.users_get_current_account()
        except AuthError:
            sys.exit("ERROR: Invalid access token; try re-generating an "
                "access token from the app console on the web.")


        print("Done!")


# ! Config
import train_data

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

if ft_config.mbatch_size > ft_config.batch_size:
    raise Exception('batch_size need to be larger than mbatch_size.')

offloadfolder = "./offload"

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                  ft_config.llama_q4_config_dir+ft_config.llama_q4_model,
                                                  device_map=ft_config.device_map,
                                                  groupsize=ft_config.groupsize,
                                                  is_v1_model=ft_config.v1)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="all",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    device_map = ft_config.device_map
    if ft_config.ddp:
        device_map = {'': 0}
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {'': 0}
    print('Device map for lora:', device_map)
    
    
    if ft_config.resume_checkpoint:
       model = PeftModel.from_pretrained(
          model, 
          ft_config.resume_checkpoint,
          ft_config.lora_apply_dir, 
          device_map=device_map,
          torch_dtype=torch.float32, 
          is_trainable=True)
       ft_config.resume_checkpoint = False
    else:
       model = PeftModel.from_pretrained(
          model, 
          ft_config.lora_apply_dir, 
          device_map=device_map, 
          torch_dtype=torch.float32, 
          is_trainable=True)
    
    print(ft_config.lora_apply_dir, 'loaded')


# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if '4bit' in str(type(m)):
        if m.is_v1_model:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    data = None
    if ft_config.ds_type == "txt" and not ft_config.skip:
        #### LLaMa
        data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "alpaca" and not ft_config.skip:
        #### Stanford Alpaca-like Data
        data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
        print(data)
        print("Hi")
    elif ft_config.ds_type == "gpt4all" and not ft_config.skip:
        #### GPT4All Data
        data = train_data.TrainGPT4All(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
    elif ft_config.ds_type == "bluemoon" and not ft_config.skip:
        #### Blue Moon Data
        data = train_data.TrainBlueMoon(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)        
    else:
        raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data(thd=ft_config.txt_row_thd, use_eos_token=ft_config.use_eos_token)
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
        
        
    if data.train_data is None:
        print("Training data is None. Check your data loading/preprocessing.")
        # Handle error here
    if data.val_data is None:
        print("Validation data is None. Check your data loading/preprocessing.")
        # Handle error here
    print("Training data: ", data.train_data)
    print("Validation data: ", data.val_data)
    print("Training data shape: ", data.train_data.shape)
    print("Validation data shape: ", data.val_data.shape)

        
    # Count eval count for wandb
    if ft_config.val_set_size > 0:
        eval_count = 50
        eval_steps = max(
            ft_config.logging_steps, (len(data.train_data) + len(data.val_data)) // (eval_count*ft_config.mbatch_size)
        )
        print(f"Run eval every {eval_steps} steps")
    else:
        eval_steps = 3000

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=ft_config.mbatch_size,
        gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
        warmup_steps=ft_config.warmup_steps,
        optim="adamw_torch",
        num_train_epochs=ft_config.epochs,
        learning_rate=ft_config.lr,
        fp16=True,
        logging_steps=ft_config.logging_steps,
        evaluation_strategy="steps" if eval_steps != 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if eval_steps != 0 else None,
        save_steps=ft_config.save_steps,
        output_dir=ft_config.lora_out_dir,
        save_total_limit=ft_config.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ft_config.ddp else None,
        weight_decay=ft_config.weight_decay,
        adam_beta1=ft_config.adam_beta1,
        adam_beta2=ft_config.adam_beta2,
        adam_epsilon=ft_config.adam_epsilon
    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=data.train_data,
        eval_dataset=data.val_data,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    # Set Model dict
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Set Verbose
    if ft_config.verbose:
        transformers.logging.set_verbosity_info()

    # Run Trainer
    with wandb.init(project=ft_config.project) as run:
        runName = run.name
        if ft_config.resume_checkpoint:
            print('Resuming from {} ...'.format(ft_config.resume_checkpoint))
            state_dict_peft = torch.load(os.path.join(ft_config.resume_checkpoint, 'llama7b-gptq-4bit-128g.safetensors'), map_location='cpu')
            set_peft_model_state_dict(model, state_dict_peft)
            trainer.train(ft_config.resume_checkpoint)
        else:
            trainer.train()

    # Restore old model state dict
    model.state_dict = old_state_dict

    print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')

while True:
    # your script here

    time.sleep(3600)  # pause for one hour
