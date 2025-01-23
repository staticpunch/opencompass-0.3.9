from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    # from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen_cdbebf import mmlu_pro_datasets
    # from opencompass.configs.datasets.MathBench.mathbench_gen import mathbench_datasets
    # from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets

datasets = ([]
    # + gsm8k_datasets[:] 
    # + mathbench_datasets_en
    # + mmlu_math
    + humaneval_datasets[:] 
    + mbpp_datasets[:]
    # + mmlu_pro_datasets[:]
    # + triviaqa_datasets
    # + simpleqa_datasets
    
)

configs_0 = [
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/baselines/task_arithmetic", "ta-3b"),
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/baselines/ties_topK_0.1", "ties-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code/checkpoint-153", "code100k-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code-50k", "code50k-3b"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "instruct-3b"),
    
]

model_configs = configs_0

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