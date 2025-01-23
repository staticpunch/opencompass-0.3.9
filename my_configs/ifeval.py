from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets

datasets = ([]
    + ifeval_datasets
)

from opencompass.models import HuggingFacewithChatTemplate

model_configs = [
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_03", "merge_#03"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_02", "merge_#02"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "model-it"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-math", "expert-math"),
]

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        tokenizer_path=model_configs[i][0],
        # tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(temperature=0),
    ) for i in range(len(model_configs))
]