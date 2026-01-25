import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

if __name__ == "__main__":
    from rich import print as rprint
    import torch.distributed as dist
    from transformers import AutoConfig
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    hf_config = AutoConfig.from_pretrained(model_path)
    print("--- hf_config ---")
    rprint(hf_config)

    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    torch.cuda.set_device(0)
        
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(hf_config)
    print("--- model ---")
    rprint(model)