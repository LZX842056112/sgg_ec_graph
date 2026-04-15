- **output_dir** (`str`, *optional*, defaults to `"trainer_output"`) — The output directory where the model predictions and checkpoints will be written.

- **overwrite_output_dir** (`bool`, *optional*, defaults to `False`) — If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory.

- **eval_strategy** (`str` or [IntervalStrategy](https://hf-mirror.com/docs/transformers/v4.56.1/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"no"`) — The evaluation strategy to adopt during training. Possible values are:`"no"`: No evaluation is done during training.`"steps"`: Evaluation is done (and logged) every `eval_steps`.`"epoch"`: Evaluation is done at the end of each epoch.

- **per_device_train_batch_size** (`int`, *optional*, defaults to 8) — The batch size *per device*. The **global batch size** is computed as: `per_device_train_batch_size * number_of_devices` in multi-GPU or distributed setups.

- **per_device_eval_batch_size** (`int`, *optional*, defaults to 8) — The batch size per device accelerator core/CPU for evaluation.

- **gradient_accumulation_steps** (`int`, *optional*, defaults to 1) — Number of updates steps to accumulate the gradients for, before performing a backward/update pass.When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

- **eval_accumulation_steps** (`int`, *optional*) — Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, the whole predictions are accumulated on the device accelerator before being moved to the CPU (faster but requires more memory).

- **learning_rate** (`float`, *optional*, defaults to 5e-5) — The initial learning rate for `AdamW` optimizer.

- **num_train_epochs(`float`,** *optional*, defaults to 3.0) — Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).

- **max_steps** (`int`, *optional*, defaults to -1) — If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until `max_steps` is reached.

- **lr_scheduler_type** (`str` or [SchedulerType](https://hf-mirror.com/docs/transformers/v4.56.1/en/main_classes/optimizer_schedules#transformers.SchedulerType), *optional*, defaults to `"linear"`) — The scheduler type to use. See the documentation of [SchedulerType](https://hf-mirror.com/docs/transformers/v4.56.1/en/main_classes/optimizer_schedules#transformers.SchedulerType) for all possible values.

- **lr_scheduler_kwargs** (‘dict’, *optional*, defaults to {}) — The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values.

- **warmup_ratio** (`float`, *optional*, defaults to 0.0) — Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.

- **warmup_steps** (`int`, *optional*, defaults to 0) — Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.

- **logging_dir** (`str`, *optional*) — [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME\***.

- **logging_strategy** (`str` or [IntervalStrategy](https://hf-mirror.com/docs/transformers/v4.56.1/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"steps"`) — The logging strategy to adopt during training. Possible values are:`"no"`: No logging is done during training.`"epoch"`: Logging is done at the end of each epoch.`"steps"`: Logging is done every `logging_steps`.

- **logging_steps** (`int` or `float`, *optional*, defaults to 500) — Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.

- **save_strategy** (`str` or `SaveStrategy`, *optional*, defaults to `"steps"`) — The checkpoint save strategy to adopt during training. Possible values are:`"no"`: No save is done during training.`"epoch"`: Save is done at the end of each epoch.`"steps"`: Save is done every `save_steps`.`"best"`: Save is done whenever a new `best_metric` is achieved.If `"epoch"` or `"steps"` is chosen, saving will also be performed at the very end of training, always.

- **save_steps** (`int` or `float`, *optional*, defaults to 500) — Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.

- **save_total_limit** (`int`, *optional*) — If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. When `load_best_model_at_end` is enabled, the “best” checkpoint according to `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two checkpoints are saved: the last one and the best one (if they are different).

- **use_cpu** (`bool`, *optional*, defaults to `False`) — Whether or not to use cpu. If set to False, we will use cuda or mps device if available.

- **bf16** (`bool`, *optional*, defaults to `False`) — Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.

  > 参考资料：https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

- **fp16** (`bool`, *optional*, defaults to `False`) — Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.

- **eval_steps** (`int` or `float`, *optional*) — Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.

- **load_best_model_at_end** (`bool`, *optional*, defaults to `False`) — Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved. See [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit) for more.When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in the case it is “steps”, `save_steps` must be a round multiple of `eval_steps`.

- **metric_for_best_model** (`str`, *optional*) — Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`.If not specified, this will default to `"loss"` when either `load_best_model_at_end == True` or `lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU` (to use the evaluation loss).If you set this value, `greater_is_better` will default to `True` unless the name ends with “loss”. Don’t forget to set it to `False` if your metric is better when lower.

- **greater_is_better** (`bool`, *optional*) — Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models should have a greater metric or not. Will default to:`True` if `metric_for_best_model` is set to a value that doesn’t end in `"loss"`.`False` if `metric_for_best_model` is not set, or set to a value that ends in `"loss"`.

- **resume_from_checkpoint** (`str`, *optional*) — The path to a folder with a valid checkpoint for your model. This argument is not directly used by [Trainer](https://hf-mirror.com/docs/transformers/v4.56.1/en/main_classes/trainer#transformers.Trainer), it’s intended to be used by your training/evaluation scripts instead. See the [example scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.