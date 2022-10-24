# binary-HuggingFace
## (Scripts for fine-tuning Hugging Face models on binary tasks)
Disclaimer: it looks like Hugging Face may be changing how `load_metric` works soon, so this script may stop working soon. If that happens, please add it as an issue and I will try to fix it as soon as possible!

## Create your datasets
First, put your datasets into the following format:
The first column should be called "sentence1", and should be string values
If you have a dataset of sentence pairs (e.g. [MRPC](https://paperswithcode.com/dataset/mrpc)), then your next column should be called "sentence2", and should contain string values.
Your final column should be called "label", and should contain integer values.

Don't have any NaN (or empty) values in your dataset.

If your dataset has single quotes in the sentences, I recommend using the `deal_with_single_quotes.py` script in order to save your dataset with double quotes surrounding the sentences.

To test to see if your dataset is in the right format, run `find_bad_lines.py` on your dataset, which will point out lines that aren't in the right format.

You should create 2-3 datasets: a train dataset, a validation dataset (optional), and a test dataset.
If you don't create a validation dataset, **use your test dataset as your validation dataset**.

## Set up Weights and Biases
Create a (free) account at https://wandb.ai/ in order to track training.
(If you don't want to do this, delete all the lines having to do with `wandb` in `main.py`.)

Set the project and entity in the `main.py` script. (Usually, the project is the project name, and the entity is the team for the project.)
```
WANDB_PROJECT = "wandb_project"
WANDB_ENTITY = "wandb_entity"
```

## Run the Code
### Train
In order to train, use the following command:
`python main.py --train_file TRAIN.CSV --validation_file VALIDATION.CSV --test_file TEST.CSV --do_train --model_name YOUR_MODEL_NAME --overwrite_output_dir --eval_steps 100 --evaluation_strategy steps --save_strategy no --report_to wandb --run_name RUN_NAME --output_dir RUN_NAME`

TRAIN.CSV should be your train dataset
VALIDATION.CSV should be your validation (or test if you don't have validation) dataset
TEST.CSV should be your test dataset
YOUR_MODEL_NAME should be the name of the model you're fine tuning (e.g. [`roberta-large`](https://huggingface.co/roberta-large))
RUN_NAME should be the name of this trained model on Weights and Biases (and when it saves folders for data about this run).

This will train a model. Every 100 steps, it will test the model on the validation dataset.
If you want to only save the best model, add `--save_total_limit 2` (see [this](https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442) for details), otherwise, it will save a checkpoint every 100 steps of training. (At this point, I'm not sure if "best" refers to "lowest loss" or "highest accuracy on the validation dataset". Please add an issue if you have a source which clarifies this.)
By saving a checkpoint each time it evaluates on the dataset, you can pick out the best model after training, which will be stored inside the checkpoint folder for that checkpoint.

You can also add in a bunch of hyperparameters for your model. To do so, just set the appropriate flags. When you first run the program, it should display all the values it's currently using that can be set. I've included a recent list below, but this may change over time.

### Test
Run the exact same script as for train, but replace `--do_train` with `--do_predict`, and `YOUR_MODEL_NAME` with the folder containing the checkpoint you want to use. This will run a prediction on your test dataset. You should find the results in the folder for the model that you did the prediction with. The `get_metrics.py` script should automatically be run, giving you accuracy, F1 score, etc., all of which will be uploaded to Weights and Biases. You should also get the predictions themselves uploaded as an artifact to weights and biases.

## Recent Hyperparameters
(These are not necessarily the defaults, they are just copy/pasted from an earlier run of this script.)
```
_n_gpu=1,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.98,
adam_epsilon=1e-06,
auto_find_batch_size=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=True,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=500,
evaluation_strategy=steps,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=2e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=-1,
log_level=passive,
log_level_replica=passive,
log_on_each_node=True,
logging_dir=roberta_test7/runs/Oct16_02-42-59_d723a2c568d1,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=steps,
lr_scheduler_type=linear,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_hf,
output_dir=roberta_test7,
overwrite_output_dir=True,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=8,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['wandb'],
resume_from_checkpoint=None,
run_name=roberta_test7,
save_on_each_node=False,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=30000,
weight_decay=0.01,
xpu_backend=None
```

To use these, put two dashes before the name and then a `=` (if setting it to a number) or space (if setting it to a string) afterwards, followed immediately by what you want to set it to. For example `--output_dir myNewOutputDir` or `--learning_rate=2e-05`.