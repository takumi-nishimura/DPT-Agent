import argparse
import os
import random
from datetime import datetime

import torch
import yaml

try:  # Prefer upstream dependency name, fallback to pyyaml-include
    from yamlinclude import YamlIncludeConstructor
except ImportError:
    from yaml_include.constructor import Constructor as YamlIncludeConstructor  # type: ignore

from coop_marl.utils import Dotdict
from coop_marl.utils.logger import get_logger, pblock
from coop_marl.utils.utils import update_existing_keys
from llms.get_llm_output import valid_models
from llms.get_llm_output_act import valid_models as valid_models_act

def _adapt_yaml_constructor(constructor_cls):
    if hasattr(constructor_cls, "add_to_loader_class"):
        return constructor_cls

    class _Adapter(constructor_cls):  # type: ignore[misc]
        @classmethod
        def add_to_loader_class(cls, loader_class):
            yaml.add_constructor("!inc", cls(), Loader=loader_class)

    return _Adapter


YamlIncludeConstructor = _adapt_yaml_constructor(YamlIncludeConstructor)
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

DEF_CONFIG = "def_config"


def save_yaml(conf, path):
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(conf, f, default_flow_style=False)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=argparse.FileType(mode="r"),
        default="config/algs/play/overcooked.yaml",
    )
    parser.add_argument(
        "--env_config_file",
        type=argparse.FileType(mode="r"),
        default="config/envs/overcooked.yaml",
    )
    parser.add_argument("--config", default={}, type=yaml.load)
    parser.add_argument(
        "--env_config",
        default={"kernel_gamma": 0.5, "render": 1, "save_folder": "./logs/play"},
        type=yaml.load,
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--control_agent", default=None, type=str)
    parser.add_argument("--send_message", default=False, type=bool)
    parser.add_argument("--receive_message", default=False, type=bool)
    parser.add_argument("--infer_human", action="store_true")
    parser.add_argument(
        "--model",
        "-m",
        default="4o-mini",
        type=str,
        choices=valid_models,
    )
    parser.add_argument("--fsm", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--display", "-d", action="store_true")
    return parser


def create_parser_act():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=argparse.FileType(mode="r"),
        default="config/algs/play/overcooked.yaml",
    )
    parser.add_argument(
        "--env_config_file",
        type=argparse.FileType(mode="r"),
        default="config/envs/overcooked_single_agent_exp1.yaml",
    )
    parser.add_argument("--config", default={}, type=yaml.load)
    parser.add_argument(
        "--env_config",
        default={"kernel_gamma": 0.5, "render": 1, "save_folder": "./logs/play"},
        type=yaml.load,
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--control_agent", default=None, type=str)
    parser.add_argument(
        "--model",
        "-m",
        default="gemma2:27b",
        type=str,
        choices=valid_models_act,
    )
    parser.add_argument("--fsm", action="store_true")
    parser.add_argument("--display", "-d", action="store_true")
    return parser


def create_parser_biased_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=argparse.FileType(mode="r"),
        default="config/algs/play/overcooked.yaml",
    )
    parser.add_argument(
        "--env_config_file",
        type=argparse.FileType(mode="r"),
        default="config/envs/overcooked.yaml",
    )
    parser.add_argument("--config", default={}, type=yaml.load)
    parser.add_argument(
        "--env_config",
        default={"kernel_gamma": 0.5, "render": 1, "save_folder": "./logs/play"},
        type=yaml.load,
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--control_agent", default=None, type=str)
    parser.add_argument("--send_message", default=False, type=bool)
    parser.add_argument("--receive_message", default=False, type=bool)
    parser.add_argument("--infer_human", default=False, action="store_true")
    parser.add_argument(
        "--model",
        "-m",
        default="4o-mini",
        type=str,
        choices=valid_models,
    )
    parser.add_argument("--biased_agent", "-ba", required=True, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 0: beef, 1: lettuce, 2: assemble_serve
    # ... more
    parser.add_argument("--fsm", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--display", "-d", action="store_true")
    return parser


def get_def_conf(data, init_call=False):
    if DEF_CONFIG not in data:
        if init_call:
            return {}
        return data
    cur_level = {k: v for k, v in data.items() if k != DEF_CONFIG}
    next_level = get_def_conf(data[DEF_CONFIG])
    next_level.update(cur_level)
    return next_level


def parse_nested_yaml(yaml):
    def_conf = get_def_conf(yaml, True)
    conf = {k: v for k, v in yaml.items() if k != DEF_CONFIG}
    if def_conf is not None:
        def_conf.update(conf)
        conf = def_conf
    return conf


def parse_args(parser):
    # print("111")
    args = parser.parse_args()
    # print("aaa")
    data = yaml.load(args.config_file, Loader=yaml.FullLoader)
    conf = parse_nested_yaml(data)
    # print("222")

    env_conf = yaml.load(args.env_config_file, Loader=yaml.FullLoader)

    # replace the config params with hparams from console args
    unused_param = [None, None]
    conf_names = ["config", "env config"]
    for i, (cli_conf, yaml_conf, text) in enumerate(zip([args.config, args.env_config], [conf, env_conf], conf_names)):
        yaml_conf, unused_param[i] = update_existing_keys(yaml_conf, cli_conf)
    # print("333")

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(args.run_name) > 0:
        run_name = args.run_name
    conf["run_name"] = run_name

    if not getattr(conf, "save_dir", ""):
        env_folder = f'{env_conf["name"]}_{env_conf["mode"]}' if "mode" in env_conf else f'{env_conf["name"]}'
        save_folder = conf["save_folder"]
        conf["save_dir"] = f'{save_folder}/{env_folder}/{conf["algo_name"]}/{run_name}'
    logger = get_logger(log_dir=conf["save_dir"], debug=conf["debug"])
    [logger.info(pblock(unused_param[i], f"Unused {conf_names[i]} parameters")) for i in range(2)]
    if args.seed == -1:
        args.seed = random.randint(1, int(2**31 - 1))

    if conf["use_gpu"]:
        conf["device"] = "cuda"

    if conf["training_device"] == "cuda":
        if not torch.cuda.is_available():
            logger.info("CUDA is not available, using CPU for training instead.")
            conf["training_device"] = "cpu"

    delattr(args, "env_config_file")
    delattr(args, "config_file")
    [save_yaml(c, f'{conf["save_dir"]}/{name}') for c, name in zip([conf, env_conf], ["conf", "env_conf"])]
    return [Dotdict(x) for x in [vars(args), conf, env_conf]] + [conf["trainer"]]
