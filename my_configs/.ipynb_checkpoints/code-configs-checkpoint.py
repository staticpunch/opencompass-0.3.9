from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen import LCB_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

datasets = ([]
    + humaneval_datasets[:] 
    + mbpp_datasets[:]
    + LCB_datasets
    + ifeval_datasets[:] 
    + mmlu_datasets[:]
    + ARC_e_datasets
    + ARC_c_datasets   
)

configs_3b_ties = [
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-3b/ties_weight_0.1", "code-ties-0.1-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-3b/ties_weight_0.2", "code-ties-0.2-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-3b/ties_weight_0.3", "code-ties-0.3-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-3b/ties_weight_0.4", "code-ties-0.4-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-3b/ties_weight_0.5", "code-ties-0.5-3b"),
]

configs_3b_auto = [
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-smol", "instruct-3b"),
    ("/mnt/md0data/hiennm/logits/models/smol-experts/llama-3.2-3b-expert-code", "code-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-3b/merge-code-3b-u19", "IC-3b-u19"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-3b/merge-code-3b-u28", "IC-3b-u28"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-3b/merge-code-3b-u37", "IC-3b-u37"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-3b/merge-code-3b-u46", "IC-3b-u46"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-3b/merge-code-3b-u55", "IC-3b-u55"),
]

configs_8b_ties = [
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-8b/ties_weight_0.1", "code-ties-0.1-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-8b/ties_weight_0.2", "code-ties-0.2-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-8b/ties_weight_0.3", "code-ties-0.3-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-8b/ties_weight_0.4", "code-ties-0.4-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-code-8b/ties_weight_0.5", "code-ties-0.5-8b"),
]

configs_8b_auto = [
    ("/mnt/md0data/hiennm/logits/models/llama-3.1-8b-smol", "instruct-8b"),
    ("/mnt/md0data/hiennm/logits/models/smol-experts/llama-3.1-8b-expert-code", "code-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-8b/merge-code-8b-u19", "IC-8b-u19"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-8b/merge-code-8b-u28", "IC-8b-u28"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-8b/merge-code-8b-u37", "IC-8b-u37"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-8b/merge-code-8b-u46", "IC-8b-u46"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-code-8b/merge-code-8b-u55", "IC-8b-u55"),
]

# model_configs = configs_8b_ties + configs_8b_auto
model_configs = configs_3b_ties + configs_3b_auto

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        max_out_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1, ),
        stop_words=['<|end_of_text|>', '<|eot_id|>', '00000'],
        generation_kwargs=dict(temperature=0),
        # model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
    ) for i in range(len(model_configs))
]