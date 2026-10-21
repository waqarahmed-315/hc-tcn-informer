"""Train and evaluate an Informer or TCN-Informer from an explicit JSON configuration.

Example settings are provided for a quick demonstration; they are NOT paper settings.
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from exp.exp_informer import Exp_Informer
from models.initialization import load_initialization_config

REQUIRED_KEYS = ['model', 'init_mode', 'init_config_path', 'experiment_label', 'study_tags', 'run_id', 'data', 'root_path', 'data_path', 'features', 'target', 'freq', 'detail_freq', 'seq_len', 'label_len', 'pred_len', 'enc_in', 'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor', 'dropout', 'attn', 'embed', 'activation', 'distil', 'mix', 'use_revin', 'tcn_channels', 'tcn_kernel_size', 'tcn_dropout', 'fusion_type', 'learning_rate', 'lr_min', 'weight_decay', 'train_epochs', 'batch_size', 'patience', 'warmup_epochs', 'lradj', 'grad_clip', 'seed', 'num_workers', 'use_gpu', 'gpu', 'results_root', 'checkpoints', 'init_cache', 'mape_eps', 'grad_diag_steps', 'save_prediction_samples', 'keep_checkpoint']


def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true", "yes", "y"):
        return True
    if str(value).lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/example_training.json",
                   help="JSON training configuration; example settings are NOT paper settings")
    p.add_argument("--init_config", default=None,
                   help="JSON with system coefficients and normalization/scaling settings, required for chaotic modes")
    p.add_argument("--model", choices=("informer", "tcn_informer"))
    p.add_argument("--init_mode", choices=("standard", "xavier", "he", "chaotic", "hyperchaotic"))
    p.add_argument("--data")
    p.add_argument("--data_path")
    p.add_argument("--pred_len", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--use_gpu", type=str2bool)
    opt = p.parse_args(argv)
    with open(opt.config, encoding="utf-8") as f:
        config = json.load(f)
    missing = sorted(set(REQUIRED_KEYS).difference(config))
    if missing:
        p.error("Missing required training configuration fields: " + ", ".join(missing))
    config.pop("_comment", None)
    for k in ("model", "init_mode", "data", "data_path", "pred_len", "seed", "use_gpu"):
        override = getattr(opt, k)
        if override is not None:
            config[k] = override
    if opt.init_config is not None:
        config["init_config_path"] = opt.init_config
    if config["model"] not in ("informer", "tcn_informer"):
        p.error("model must be informer or tcn_informer")
    if config["init_mode"] not in ("standard", "xavier", "he", "chaotic", "hyperchaotic"):
        p.error("Invalid init_mode")
    if config["init_mode"] in ("chaotic", "hyperchaotic"):
        # Fail early and clearly if numerical settings have not been supplied.
        load_initialization_config(config["init_config_path"])
    if config["init_mode"] == "standard" and "example_training" in os.path.basename(opt.config):
        print("NOTE: using illustrative configuration, NOT a reproduction of paper results.")
    return argparse.Namespace(**config)


def build_setting(args):
    if args.run_id:
        return args.run_id
    return (f"{args.data}_pl{args.pred_len}_seed{args.seed}_{args.model}_{args.init_mode}"
            f"_rv{int(args.use_revin)}_fus{args.fusion_type}")


def main(argv=None):
    args = get_args(argv)
    args.setting = build_setting(args)
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    exp = Exp_Informer(args)
    exp.train(args.setting)
    exp.test(args.setting)


if __name__ == "__main__":
    main()
