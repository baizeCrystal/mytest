import argparse
import os
import sys

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from datasets import StudentActionDataset
from models import PhaseContrastActionErrorModel


def parse_args():
    parser = argparse.ArgumentParser(description="Build correct-action phase prototypes")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--action_slot_repo", type=str, default=os.path.join(os.path.dirname(PROJECT_DIR), "Action-slot"))

    parser.add_argument("--num_actions", type=int, required=True)
    parser.add_argument("--num_error_classes", type=int, required=True)
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="x3d", choices=["x3d", "action_slot", "simple_cnn"])
    parser.add_argument("--no_pretrained_backbone", action="store_true")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--normalize", type=str, default="pytorchvideo", choices=["pytorchvideo", "imagenet", "none"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_videos(videos, device):
    return [frame.to(device, dtype=torch.float32, non_blocking=True) for frame in videos]


def main():
    args = parse_args()
    args.device = resolve_device(args.device)

    dataset = StudentActionDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        split="val",
        num_error_classes=args.num_error_classes,
        seq_len=args.seq_len,
        sampling_rate=args.sampling_rate,
        image_size=args.image_size,
        resize_size=args.resize_size,
        normalize=args.normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
    )

    model = PhaseContrastActionErrorModel(
        num_actions=args.num_actions,
        num_error_classes=args.num_error_classes,
        num_phases=args.num_phases,
        feature_dim=args.feature_dim,
        backbone=args.backbone,
        action_slot_repo=args.action_slot_repo,
        pretrained_backbone=not args.no_pretrained_backbone,
        learnable_prototypes=True,
    ).to(args.device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    sums = torch.zeros(args.num_actions, args.num_phases, args.feature_dim, device=args.device)
    counts = torch.zeros(args.num_actions, device=args.device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Build prototypes"):
            is_correct = batch["is_correct"].to(args.device)
            if not is_correct.any():
                continue

            videos = move_videos(batch["videos"], args.device)
            action_id = batch["action_id"].to(args.device)
            with autocast(enabled=args.amp and args.device.type == "cuda"):
                outputs = model(videos, action_id)
            phase_features = outputs["phase_features"]

            for action in action_id[is_correct].unique():
                mask = (action_id == action) & is_correct
                sums[action] += phase_features[mask].sum(dim=0)
                counts[action] += mask.sum()

    missing = counts == 0
    counts = counts.clamp_min(1.0)
    prototypes = sums / counts[:, None, None]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes.detach().cpu(),
            "counts": counts.detach().cpu(),
            "missing_actions": torch.where(missing.cpu())[0],
            "num_actions": args.num_actions,
            "num_phases": args.num_phases,
            "feature_dim": args.feature_dim,
        },
        args.output,
    )
    print(f"Saved prototypes to {args.output}")
    if missing.any():
        print(f"WARNING: missing correct samples for actions: {torch.where(missing.cpu())[0].tolist()}")


if __name__ == "__main__":
    main()
