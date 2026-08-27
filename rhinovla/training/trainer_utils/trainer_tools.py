"""trainer_tools.py — optimizer param-group construction, module freezing, and checkpoint load/save helpers."""

import re
import json
import numpy as np
import torch

from accelerate.logging import get_logger

logger = get_logger(__name__)


# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        elif "=" in arg:
            normalized.append(arg)
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    param_groups = []
    special_freeze_modes = {"rhino_train_merged_action_expert_io"}

    for freeze_path in freeze_patterns:
        if freeze_path in special_freeze_modes:
            # Special modes are applied in freeze_backbones() and do not map to a real module path.
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
        except AttributeError:
            print(f"⚠️ module path `{module_name}` not found in vla")
            continue
        # Only optimize parameters that remain trainable after freeze_backbones().
        params = [p for p in module.parameters() if p.requires_grad]
        if params:  # only add param group if there are trainable parameters
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)

    # assign base learning rate to the remaining unused trainable parameters
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in used_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from PIL import Image


def resize_images(images, target_size=(256, 256), resample=Image.BICUBIC):
    """
    Recursively resize all images in the nested list.

    Aligned with Qwen3-VL official preprocessing style:
    - Default target 256x256 (was 224x224; forced 224 path removed).
    - BICUBIC resample (PIL default in recent versions, made explicit).

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :param resample: PIL resample method, default BICUBIC.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):
        return images.resize(target_size, resample=resample)
    elif isinstance(images, list):
        return [resize_images(img, target_size, resample) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def configure_torch_compile(model, enabled=False):
        """Optionally compile only the Action Expert forward call.

        Keeping the module object itself intact is important: replacing the
        module with an ``OptimizedModule`` changes checkpoint namespaces to
        ``_orig_mod.*`` and makes strict load/resume unnecessarily fragile.
        Qwen, the vision tower, data loading, and checkpoint I/O stay eager.
        """
        if not enabled:
            return model
        if not hasattr(model, "action_expert"):
            raise ValueError("action_expert is required for trainer.compile_action_expert")
        model.action_expert.forward = torch.compile(model.action_expert.forward, dynamic=True)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("Enabled torch.compile for action_expert.forward", flush=True)
        return model

    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen, action_expert",
            it means to freeze model.qwen and model.action_expert.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            # Special handling for partial freeze modes
            if "rhino_train_merged_action_expert_io" in patterns:
                train_count = 0
                prefixes = ("action_expert.",)
                for name, param in model.named_parameters():
                    if name.startswith(prefixes):
                        param.requires_grad = True
                        train_count += param.numel()
                frozen.append(
                    "rhino_train_merged_action_expert_io "
                    f"(train={train_count/1e6:.1f}M)"
                )
                patterns.remove("rhino_train_merged_action_expert_io")

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        # 检查分布式环境是否已初始化，未初始化时跳过 rank 检查
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass  # 分布式未初始化时直接在主进程打印
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def _adapt_checkpoint_state_dict(target_state_dict, checkpoint):
        """Adapt plain checkpoint keys to the target model.

        The release checkpoint is expected to be a merged, plain PyTorch state_dict.
        The only compatibility kept here is `.base.` <-> plain Linear key
        mapping, which is harmless for already-plain checkpoints and avoids
        carrying unrelated instance-adapter code.
        """
        target_keys = set(target_state_dict.keys())
        adapted = {}
        stats = {
            "plain_to_base": 0,
            "base_to_plain": 0,
            "legacy_module_key": 0,
            "dropped_unexpected": 0,
        }

        for key, value in checkpoint.items():
            original_key = key
            if key.startswith("qwen_vl_interface."):
                key = "qwen." + key[len("qwen_vl_interface.") :]
            elif key.startswith("action_io."):
                key = "action_expert.io." + key[len("action_io.") :]
            elif key.startswith("action_expert.layers.") or key.startswith("action_expert.norm."):
                key = "action_expert.transformer." + key[len("action_expert.") :]
            elif key.startswith("state_mask_proj."):
                key = "action_expert.state_mask_proj." + key[len("state_mask_proj.") :]
            elif key.startswith("action_mask_proj."):
                key = "action_expert.action_mask_proj." + key[len("action_mask_proj.") :]
            if key != original_key:
                stats["legacy_module_key"] += 1

            if key in target_keys:
                adapted[key] = value
                continue

            if ".base." in key:
                plain_key = key.replace(".base.", ".")
                if plain_key in target_keys:
                    adapted.setdefault(plain_key, value)
                    stats["base_to_plain"] += 1
                    continue

            if "." in key:
                prefix, suffix = key.rsplit(".", 1)
                base_key = f"{prefix}.base.{suffix}"
                if base_key in target_keys:
                    adapted.setdefault(base_key, value)
                    stats["plain_to_base"] += 1
                    continue

            stats["dropped_unexpected"] += 1

        return adapted, stats

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """Load a merged checkpoint into the model."""
        if not checkpoint_path:
            return []
        if (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        if reload_modules:
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                attrs = path.split(".")
                module = model
                try:
                    for module_name in attrs:
                        module = getattr(module, module_name)
                except AttributeError:
                    raise RuntimeError(f"strict checkpoint load failed: cannot find module path '{path}'")

                prefix = path + "."
                sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                if not sub_state_dict:
                    raise RuntimeError(f"strict checkpoint load failed: parameters not found in checkpoint for '{path}'")
                adapted_sub_state_dict, adapt_stats = TrainerUtils._adapt_checkpoint_state_dict(
                    module.state_dict(),
                    sub_state_dict,
                )
                TrainerUtils._raise_on_unexpected_checkpoint_tensors(adapt_stats, path)
                module.load_state_dict(adapted_sub_state_dict, strict=True)
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"✅ parameters loaded to module '{path}'")
                    TrainerUtils._print_load_adapt_stats(adapt_stats)
            return model

        try:
            adapted_checkpoint, adapt_stats = TrainerUtils._adapt_checkpoint_state_dict(
                model.state_dict(),
                checkpoint,
            )
            TrainerUtils._raise_on_unexpected_checkpoint_tensors(adapt_stats, "<full_model>")
            model.load_state_dict(adapted_checkpoint, strict=True)
            if (not dist.is_initialized() or dist.get_rank() == 0):
                print("✅ loaded <full_model> model parameters")
                TrainerUtils._print_load_adapt_stats(adapt_stats)
        except Exception as e:
            raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def _raise_on_unexpected_checkpoint_tensors(adapt_stats, label: str):
        dropped = int(adapt_stats.get("dropped_unexpected", 0) or 0)
        if dropped:
            raise RuntimeError(
                f"strict checkpoint load failed for {label}: "
                f"{dropped} unexpected checkpoint tensor(s)"
            )

    @staticmethod
    def _print_load_adapt_stats(adapt_stats):
        if adapt_stats["plain_to_base"]:
            print(f"✅ remapped {adapt_stats['plain_to_base']} plain Linear params into .base.* weights")
        if adapt_stats["base_to_plain"]:
            print(f"✅ remapped {adapt_stats['base_to_plain']} .base.* params into plain Linear weights")
        if adapt_stats.get("legacy_module_key"):
            print(f"✅ remapped {adapt_stats['legacy_module_key']} legacy module keys to RhinoVLA keys")

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # 获取所有符合命名规则，支持 .pt 和 .safetensors
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) 
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # 确保是文件
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # 提取步数并排序
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # 按步数排序，获取最新的 checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")
