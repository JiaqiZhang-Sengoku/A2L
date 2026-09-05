"""Small CLI and provenance helpers for single-domain and compound/open A2L."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

from .project_paths import PROJECT_ROOT, relative_path, resolve_path

DEFAULTS = dict(
    dataset="Domain1", epoch=30, warmup_epochs=10, lr=5e-4, batch_size=8,
    seed=42, active_num=2, model_ema_rate=0.98, pseudo_label_threshold=0.75,
    temperature=0.10, hard_topk_pixels=512, cal_margin_scale=0.20,
    beta_sca=1.0, lambda_cal=1.0, num_workers=2, gpu="0",
    data_dir="Data", model_file="Checkpoints/Source/source_model.pth.tar",
    output_root="Outputs",
)


def experiment_domains(task):
    """Fixed original protocols: adaptation domains, C tests, and O tests."""
    if task == "Domain4":
        return ["Domain2", "Domain1"], ["Domain2", "Domain1"], ["Domain4"]
    return [task], [task], []


def parse_args(argv=None, *, default_dataset=None):
    parser = argparse.ArgumentParser(description="A2L: UPA → PAUH → LGMC, including compound/open Domain4",
                                     allow_abbrev=False)
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--skip-test", action="store_true",
                        help="train without opening test images or labels (development runs)")
    parser.add_argument("--validation-split",
                        help="fixed held-out TRAIN names for development; requires --skip-test")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.set_defaults(active_num=None, dataset=default_dataset or DEFAULTS["dataset"])
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--config")
    config_path, _ = probe.parse_known_args(argv)
    values = {}
    if config_path.config:
        try:
            values = json.loads(resolve_path(config_path.config).read_text(encoding="utf-8"))
            if not isinstance(values, dict) or set(values) - DEFAULTS.keys():
                raise ValueError("configuration contains unsupported keys")
            for key, value in values.items():
                expected = type(DEFAULTS[key])
                if isinstance(value, bool) or not isinstance(value, (int, float) if expected is float else expected):
                    raise ValueError(f"invalid type for {key}")
        except (OSError, ValueError) as error:
            parser.error(str(error))
    args = parser.parse_args(argv, namespace=argparse.Namespace(**values))
    if any(not math.isfinite(getattr(args, key)) for key, value in DEFAULTS.items()
           if isinstance(value, float)):
        parser.error("floating-point parameters must be finite")
    if args.validation_split and not args.skip_test:
        parser.error("--validation-split requires --skip-test; do not tune on test")
    if args.dataset not in ("Domain1", "Domain2", "Domain4"):
        parser.error("choose Domain1, Domain2, or the compound/open Domain4 experiment")
    if args.validation_split and args.dataset == "Domain4":
        parser.error("--validation-split currently reports single-domain development only; "
                     "run the full Domain4 compound/open experiment without this option")
    if args.active_num is None:
        args.active_num = {"Domain1": 2, "Domain2": 5, "Domain4": 7}[args.dataset]
    if not 0 < args.warmup_epochs < args.epoch:
        parser.error("require 0 < warmup-epochs < epoch so both training stages execute")
    if args.batch_size < 2 or args.active_num < 2 or args.num_workers < 0:
        parser.error("batch-size and active-num must be >= 2; num-workers must be >= 0")
    if args.lr <= 0 or args.temperature <= 0 or args.hard_topk_pixels < 1:
        parser.error("lr, temperature and hard-topk-pixels must be positive")
    if not 0 <= args.model_ema_rate < 1 or not 0.5 < args.pseudo_label_threshold < 1:
        parser.error("invalid EMA rate or pseudo-label threshold")
    if min(args.beta_sca, args.lambda_cal, args.cal_margin_scale) < 0:
        parser.error("loss weights and margin must be nonnegative")
    if not args.gpu.isdigit():
        parser.error("gpu must be a single nonnegative device index")
    return args


def parse_domain_args(domain, argv=None):
    """Fix the domain; D1/D2 use validation, D4 retains compound/open tests."""
    if domain not in ("Domain1", "Domain2", "Domain4"):
        raise ValueError("domain entrypoints support Domain1, Domain2, or Domain4 only")
    arguments = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--config")
    probe.add_argument("--validation-split")
    selected, _ = probe.parse_known_args(arguments)
    if selected.config is None:
        arguments = ["--config", f"Config/A2L_{domain}.json", *arguments]
    if domain != "Domain4":
        if selected.validation_split is None:
            arguments = ["--validation-split", "Config/validation_split.json", *arguments]
        arguments = ["--skip-test", *arguments]
    args = parse_args(arguments, default_dataset=domain)
    if args.dataset != domain:
        parser = argparse.ArgumentParser(prog=f"A2L_{domain}.py", allow_abbrev=False)
        parser.error(f"this entry requires dataset={domain}; effective dataset={args.dataset}. "
                     "Use the matching domain entry or correct --config/--dataset.")
    if domain != "Domain4" and not args.validation_split:
        parser = argparse.ArgumentParser(prog=f"A2L_{domain}.py", allow_abbrev=False)
        parser.error("domain entrypoints require a nonempty independent --validation-split")
    return args


def effective_config(args):
    config = {key: getattr(args, key) for key in DEFAULTS}
    for key in ("data_dir", "model_file", "output_root"):
        config[key] = relative_path(config[key])
    return config


def apply_validation_split(args, train_set, weak_set, validation_set):
    """Apply a pre-registered manifest, without opening any images or masks."""
    manifest = json.loads(resolve_path(args.validation_split).read_text(encoding="utf-8"))
    adapt_domains, _, _ = experiment_domains(args.dataset)
    names = [name for domain in adapt_domains
             for name in manifest["domains"][domain]["validation_names"]]
    held_out = set(names)
    available = {row["img_name"] for row in train_set.image_list}
    if len(names) != len(held_out) or not held_out or not held_out < available:
        raise ValueError("validation names must be unique and a proper subset of TRAIN")
    for dataset in (train_set, weak_set):
        dataset.image_list = [row for row in dataset.image_list if row["img_name"] not in held_out]
    validation_set.image_list = [row for row in validation_set.image_list if row["img_name"] in held_out]
    if {row["img_name"] for row in validation_set.image_list} != held_out:
        raise ValueError("validation dataset does not match the fixed manifest")
    return manifest


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def create_run(args, train_set, test_sets, validation_set=None):
    import torch
    output = (resolve_path(args.output_dir) if args.output_dir else
              resolve_path(args.output_root) / args.dataset / datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    output.mkdir(parents=True, exist_ok=False)
    config = effective_config(args)
    source = {"path": str(resolve_path(args.model_file)), "sha256": sha256_file(resolve_path(args.model_file))}
    code_paths = [PROJECT_ROOT / name for name in
                  ("A2L.py", "A2L_Domain1.py", "A2L_Domain2.py", "A2L_Domain4.py")]
    for name in ("Dataloaders", "Networks", "Utils", "Config"):
        code_paths += sorted(p for p in (PROJECT_ROOT / name).rglob("*") if p.suffix in (".py", ".json"))
    metadata = dict(created_at=datetime.now().astimezone().isoformat(), command=sys.argv,
                    source_checkpoint=source, final_evaluation_model="student", seed=args.seed,
                    annotation_budget=args.active_num, test_evaluated=not args.skip_test,
                    code_sha256={str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in code_paths})
    adapt, compound, opened = experiment_domains(args.dataset)
    metadata.update(adaptation_domains=adapt, compound_evaluation_domains=compound,
                    open_evaluation_domains=opened, selection_pool="joint; no per-domain quotas")
    validation_count = len(validation_set) if validation_set is not None else 0
    metadata.update(validation_annotations=validation_count,
                    per_run_training_annotations=args.active_num,
                    per_run_total_annotations=args.active_num + validation_count,
                    selection_rule=(f"best independent-validation student in epochs {args.warmup_epochs + 1}-{args.epoch}; "
                                    "maximum mean Dice, then minimum valid ASSD, then earliest epoch"
                                    if validation_set is not None else "final epoch student; no validation selection"),
                    test_evaluated=False, status="running")
    if args.validation_split:
        metadata["validation_split"] = dict(path=str(resolve_path(args.validation_split)),
                                           sha256=sha256_file(resolve_path(args.validation_split)))
    def records(dataset):
        return [dict(name=row["img_name"], image_sha256=sha256_file(row["image"]))
                for row in dataset.image_list] if dataset is not None else []
    manifest = dict(train=records(train_set), test={name: records(dataset) for name, dataset in test_sets.items()},
                    validation=records(validation_set),
                    masks_hashed=False)
    if validation_set is not None:
        manifest["validation_mask_sha256"] = {row["img_name"]: sha256_file(row["label"])
                                              for row in validation_set.image_list}
    environment = dict(python=sys.version, platform=platform.platform(),
                       packages={p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "scipy", "Pillow", "medpy")},
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for name, value in (("config", config), ("run_metadata", metadata),
                        ("dataset_manifest", manifest), ("environment", environment)):
        write_json(output / (name + ".json"), value)
    return output, metadata


def checkpoint_payload(model, args, metadata, role, *, epoch=None):
    if epoch is not None and (type(epoch) is not int or not 1 <= epoch <= args.epoch):
        raise ValueError("checkpoint epoch must be an actual completed epoch within the training schedule")
    return dict(model_state_dict=model.state_dict(), config=effective_config(args), seed=args.seed,
                epoch=args.epoch if epoch is None else epoch, model_role=role,
                source_checkpoint=metadata["source_checkpoint"])


def load_checkpoint(model, path):
    import torch
    saved = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
