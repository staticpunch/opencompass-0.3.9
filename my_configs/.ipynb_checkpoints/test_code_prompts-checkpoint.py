from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate
    
with read_base():
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_4a6eef import humaneval_datasets as datasets_4a6eef
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_6d1cc2 import humaneval_datasets as datasets_6d1cc2
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets as datasets_a82cae 
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_d2537e import humaneval_datasets as datasets_d2537e
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_fd5822 import humaneval_datasets as datasets_fd5822
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_ff7054 import humaneval_datasets as datasets_ff7054
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets as datasets_8e312c
    from opencompass.configs.datasets.humaneval.humaneval_gen_66a7f4 import humaneval_datasets as datasets_66a7f4

dataset_configs = [
    (datasets_4a6eef, "4a6eef"),
    (datasets_6d1cc2, "6d1cc2"),
    (datasets_a82cae, "a82cae"),
    (datasets_d2537e, "d2537e"),
    (datasets_fd5822, "fd5822"),
    (datasets_ff7054, "ff7054"),
    (datasets_8e312c, "8e312c"),
    (datasets_66a7f4, "66a7f4"),
]

for config in dataset_configs:
    print("before:", config[0][0]["abbr"])
    config[0][0]["abbr"] = config[1]
    print("after:", config[0][0]["abbr"])

datasets = sum([dataset for dataset, name in dataset_configs], [])

configs = [
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code/checkpoint-153", "code100k-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code-50k", "code50k-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-evolcode-66k/checkpoint-84", "code66k-3b"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "instruct-3b"),
    
]

model_configs = configs

models = [
    dict(
        # type=HuggingFacewithChatTemplate,
        type=VLLMwithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        # tokenizer_path=model_configs[i][0],
        # tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1, ),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(temperature=0),
        # model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
    ) for i in range(len(model_configs))
]