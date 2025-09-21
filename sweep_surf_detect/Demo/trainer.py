import sys

sys.path.append("../point-cept")
sys.path.append("../base-trainer/")

import os
import torch

from sweep_surf_detect.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = os.environ["HOME"] + "/chLi/Dataset/"
    assert dataset_root_folder_path is not None
    print(dataset_root_folder_path)

    batch_size = 2
    accum_iter = 32
    num_workers = 16
    model_file_path = "./output/v1/model_last.pth"
    model_file_path = None
    weights_only = False
    device = "auto"
    dtype = torch.float32
    warm_step_num = 100
    finetune_step_num = -1
    lr = 1e-4
    lr_batch_size = 64
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 1
    use_amp = True
    quick_test = False

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        model_file_path,
        weights_only,
        device,
        dtype,
        warm_step_num,
        finetune_step_num,
        lr,
        lr_batch_size,
        ema_start_step,
        ema_decay_init,
        ema_decay,
        save_result_folder_path,
        save_log_folder_path,
        best_model_metric_name,
        is_metric_lower_better,
        sample_results_freq,
        use_amp,
        quick_test,
    )

    trainer.train()
    return True
