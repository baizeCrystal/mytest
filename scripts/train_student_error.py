import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from datasets import StudentActionDataset
from models import PhaseContrastActionErrorModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train phase-contrast student action error recognizer")

    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--action_slot_repo", type=str, default=os.path.join(os.path.dirname(PROJECT_DIR), "Action-slot"))
    parser.add_argument("--logdir", type=str, default="student_error_log")

    parser.add_argument("--num_actions", type=int, required=True)
    parser.add_argument("--num_error_classes", type=int, required=True)
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="x3d", choices=["x3d", "action_slot", "simple_cnn"])
    parser.add_argument(
        "--model_variant",
        type=str,
        default="kinematic_chain",
        choices=["kinematic_chain", "part_slot"],
    )
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--num_part_slots", type=int, default=7)
    parser.add_argument(
        "--part_slot_preset",
        type=str,
        default="full_body_7",
        choices=["lower_body_6", "full_body_7", "custom"],
    )
    parser.add_argument("--part_slot_prior_strength", type=float, default=2.5)
    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--skeleton_root", type=str, default="")
    parser.add_argument("--skeleton_layout", type=str, default="coco17")
    parser.add_argument("--skeleton_num_joints", type=int, default=17)
    parser.add_argument("--skeleton_input_dim", type=int, default=3)
    parser.add_argument("--skeleton_required", action="store_true")
    parser.add_argument("--phase_temperature", type=float, default=0.08)
    parser.add_argument("--phase_min_duration_ratio", type=float, default=0.08)

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--normalize", type=str, default="pytorchvideo", choices=["pytorchvideo", "imagenet", "none"])
    parser.add_argument("--disable_hflip", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--phase_duration_weight", type=float, default=0.05)
    parser.add_argument("--kinematic_length_weight", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--train_last_blocks", type=int, default=2)
    parser.add_argument("--clip_grad_norm", type=float, default=10.0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--part_slot_visualize", action="store_true")
    parser.add_argument("--part_slot_vis_every", type=int, default=1)
    parser.add_argument("--part_slot_vis_samples", type=int, default=4)
    parser.add_argument("--part_slot_vis_size", type=int, default=160)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return GradScaler(enabled=enabled)


def amp_autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return autocast(enabled=enabled)


def has_nonfinite_gradients(model) -> bool:
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return True
    return False


def get_normalization_stats(normalize: str):
    if normalize == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if normalize == "none":
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    return [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_part_slot_names(model):
    module = unwrap_model(model)
    aggregator = getattr(module, "part_aggregator", None)
    if aggregator is None:
        return []
    if hasattr(aggregator, "get_part_slot_names"):
        return aggregator.get_part_slot_names()
    return [f"slot_{idx:02d}" for idx in range(getattr(aggregator, "num_part_slots", 0))]


def should_export_part_slot_visualizations(args, model, train: bool, epoch: int) -> bool:
    if train or not args.part_slot_visualize:
        return False
    if getattr(unwrap_model(model), "part_aggregator", None) is None:
        return False
    if args.part_slot_vis_samples <= 0 or args.part_slot_vis_every <= 0:
        return False
    if (epoch + 1) % args.part_slot_vis_every != 0:
        return False
    return bool(get_part_slot_names(model))


def tensor_to_uint8_image(frame: torch.Tensor, normalize: str) -> np.ndarray:
    mean, std = get_normalization_stats(normalize)
    mean = torch.tensor(mean, dtype=frame.dtype, device=frame.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=frame.dtype, device=frame.device).view(3, 1, 1)
    image = frame.detach().cpu() * std.cpu() + mean.cpu()
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.uint8(np.round(image * 255.0))


def overlay_attention_map(image: np.ndarray, attention: torch.Tensor) -> Image.Image:
    attn = attention.detach().float().cpu()
    attn = attn - attn.min()
    if float(attn.max()) > 0:
        attn = attn / attn.max()
    heat = attn.numpy()[..., None]
    image_f = image.astype(np.float32)
    highlight = np.array([255.0, 72.0, 72.0], dtype=np.float32)
    alpha = 0.60 * heat
    overlaid = image_f * (1.0 - alpha) + highlight * alpha
    return Image.fromarray(np.uint8(np.clip(overlaid, 0.0, 255.0)))


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def save_part_slot_visualization(sample_id, videos, phase_weights, part_attn, part_names, normalize, output_path, cell_size):
    num_phases = int(phase_weights.shape[0])
    num_parts = int(part_attn.shape[1])
    if len(part_names) != num_parts:
        part_names = [f"slot_{idx:02d}" for idx in range(num_parts)]

    time_coords = torch.arange(phase_weights.shape[-1], dtype=phase_weights.dtype)
    centers = (phase_weights * time_coords[None]).sum(dim=-1) / phase_weights.sum(dim=-1).clamp_min(1e-6)
    frame_indices = centers.round().long().tolist()
    left_margin = 132
    top_margin = 48
    title_margin = 28
    canvas = Image.new(
        "RGB",
        (left_margin + num_phases * cell_size, top_margin + num_parts * cell_size + title_margin),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"sample={sample_id}", fill=(0, 0, 0))

    for phase_idx in range(num_phases):
        frame_idx = int(frame_indices[phase_idx])
        draw.text(
            (left_margin + phase_idx * cell_size + 8, title_margin),
            f"phase_{phase_idx + 1} | t={frame_idx}",
            fill=(0, 0, 0),
        )

    for part_idx, part_name in enumerate(part_names):
        draw.text((12, top_margin + part_idx * cell_size + cell_size // 2 - 8), part_name, fill=(0, 0, 0))

    for phase_idx in range(num_phases):
        frame_idx = int(frame_indices[phase_idx])
        frame = tensor_to_uint8_image(videos[frame_idx], normalize)
        for part_idx in range(num_parts):
            attn_map = part_attn[phase_idx, part_idx, frame_idx]
            attn_map = F.interpolate(
                attn_map[None, None],
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            overlay = overlay_attention_map(frame, attn_map).resize((cell_size, cell_size), resample=Image.BILINEAR)
            canvas.paste(
                overlay,
                (left_margin + phase_idx * cell_size, top_margin + part_idx * cell_size + title_margin),
            )

    canvas.save(output_path)


def export_part_slot_visualizations(model, batch, outputs, args, epoch: int, exported: int) -> int:
    if exported >= args.part_slot_vis_samples:
        return exported

    part_names = get_part_slot_names(model)
    if not part_names:
        return exported

    export_dir = os.path.join(args.logdir, "part_slot_vis", f"epoch_{epoch + 1:03d}")
    os.makedirs(export_dir, exist_ok=True)

    batch_size = len(batch["id"])
    remaining = args.part_slot_vis_samples - exported
    save_count = min(batch_size, remaining)
    for sample_idx in range(save_count):
        sample_id = batch["id"][sample_idx]
        sample_videos = [frame[sample_idx] for frame in batch["videos"]]
        sample_phase_weights = outputs["phase_weights"][sample_idx].detach().cpu()
        sample_part_attn = outputs["part_attn"][sample_idx].detach().cpu()
        output_name = f"{exported + sample_idx + 1:02d}_{_sanitize_name(sample_id)}.png"
        save_part_slot_visualization(
            sample_id=sample_id,
            videos=sample_videos,
            phase_weights=sample_phase_weights,
            part_attn=sample_part_attn,
            part_names=part_names,
            normalize=args.normalize,
            output_path=os.path.join(export_dir, output_name),
            cell_size=args.part_slot_vis_size,
        )
    return exported + save_count


def move_videos(videos, device):
    return [frame.to(device, dtype=torch.float32, non_blocking=True) for frame in videos]


def move_skeleton(skeleton, device):
    if skeleton is None:
        return None
    return skeleton.to(device, dtype=torch.float32, non_blocking=True)


def build_dataset(args, manifest, split):
    skeleton_enabled = bool(args.use_skeleton and args.model_variant == "kinematic_chain")
    return StudentActionDataset(
        manifest_path=manifest,
        data_root=args.data_root,
        split=split,
        num_error_classes=args.num_error_classes,
        seq_len=args.seq_len,
        sampling_rate=args.sampling_rate,
        image_size=args.image_size,
        resize_size=args.resize_size,
        normalize=args.normalize,
        enable_hflip=(
            (split == "train")
            and (not args.disable_hflip)
            and (not skeleton_enabled)
            and args.model_variant not in {"part_slot", "kinematic_chain"}
        ),
        use_skeleton=skeleton_enabled,
        skeleton_root=args.skeleton_root,
        skeleton_layout=args.skeleton_layout,
        skeleton_num_joints=args.skeleton_num_joints,
        skeleton_input_dim=args.skeleton_input_dim,
        skeleton_required=bool(args.skeleton_required and skeleton_enabled),
    )


def build_model(args):
    skeleton_enabled = bool(args.use_skeleton and args.model_variant == "kinematic_chain")
    return PhaseContrastActionErrorModel(
        num_actions=args.num_actions,
        num_error_classes=args.num_error_classes,
        num_phases=args.num_phases,
        feature_dim=args.feature_dim,
        backbone=args.backbone,
        action_slot_repo=args.action_slot_repo,
        freeze_backbone=args.freeze_backbone,
        train_last_blocks=args.train_last_blocks,
        phase_temperature=args.phase_temperature,
        phase_min_duration_ratio=args.phase_min_duration_ratio,
        pretrained_backbone=not args.no_pretrained_backbone,
        model_variant=args.model_variant,
        num_part_slots=args.num_part_slots,
        part_slot_preset=args.part_slot_preset,
        part_slot_prior_strength=args.part_slot_prior_strength,
        use_skeleton=skeleton_enabled,
        skeleton_num_joints=args.skeleton_num_joints,
        skeleton_input_dim=args.skeleton_input_dim,
        skeleton_layout=args.skeleton_layout,
    )


def compute_metrics(targets, scores):
    targets = np.asarray(targets, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    valid = targets.sum(axis=0) > 0
    if valid.any():
        m_ap = average_precision_score(targets[:, valid], scores[:, valid], average="macro")
    else:
        m_ap = 0.0
    preds = scores >= 0.5
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return {"mAP": float(m_ap), "macro_f1": float(f1)}


def run_epoch(model, loader, optimizer, scaler, args, train=True, epoch=0):
    model.train(train)
    losses = []
    all_targets = []
    all_scores = []
    exported_visualizations = 0
    skipped_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "Train" if train else "Val"
    export_visuals = should_export_part_slot_visualizations(args, model, train=train, epoch=epoch)

    with context:
        for batch in tqdm(loader, desc=desc):
            videos = move_videos(batch["videos"], args.device)
            skeleton = move_skeleton(batch.get("skeleton"), args.device)
            has_skeleton = batch.get("has_skeleton")
            if has_skeleton is not None:
                has_skeleton = has_skeleton.to(args.device, non_blocking=True)
            error = batch["error"].to(args.device, dtype=torch.float32, non_blocking=True)
            action_id = batch["action_id"].to(args.device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with amp_autocast(args.device, enabled=args.amp and args.device.type == "cuda"):
                outputs = model(videos, action_id, skeleton=skeleton, has_skeleton=has_skeleton)
                loss_dict = model.compute_losses(
                    outputs,
                    error_targets=error,
                    phase_duration_weight=args.phase_duration_weight,
                    kinematic_length_weight=args.kinematic_length_weight,
                )
                loss = loss_dict["total"]

            logits = outputs["logits"]
            if not torch.isfinite(loss).all() or not torch.isfinite(logits).all():
                skipped_batches += 1
                sample_ids = batch["id"][:3]
                print(
                    f"[Warn] skipped non-finite {desc.lower()} batch at epoch {epoch + 1} "
                    f"ids={list(sample_ids)}"
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if has_nonfinite_gradients(model):
                    skipped_batches += 1
                    sample_ids = batch["id"][:3]
                    print(
                        f"[Warn] skipped non-finite gradient batch at epoch {epoch + 1} "
                        f"ids={list(sample_ids)}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            losses.append(float(loss.item()))
            all_targets.append(error.detach().cpu().numpy())
            batch_scores = torch.sigmoid(logits)
            batch_scores = torch.nan_to_num(batch_scores, nan=0.0, posinf=1.0, neginf=0.0)
            all_scores.append(batch_scores.detach().cpu().numpy())

            if export_visuals and exported_visualizations < args.part_slot_vis_samples:
                exported_visualizations = export_part_slot_visualizations(
                    model=model,
                    batch=batch,
                    outputs=outputs,
                    args=args,
                    epoch=epoch,
                    exported=exported_visualizations,
                )

    if not losses:
        if train:
            raise RuntimeError(f"No valid batches completed in {desc.lower()} epoch {epoch + 1}")
        print(f"[Warn] no valid {desc.lower()} batches completed in epoch {epoch + 1}")
        return {
            "loss": float("nan"),
            "mAP": float("nan"),
            "macro_f1": float("nan"),
            "skipped_batches": int(skipped_batches),
        }

    targets = np.concatenate(all_targets, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    metrics = compute_metrics(targets, scores)
    metrics["loss"] = float(np.mean(losses))
    metrics["skipped_batches"] = int(skipped_batches)
    return metrics


def save_checkpoint(path, epoch, model, optimizer, best_map):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mAP": best_map,
        },
        path,
    )


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.logdir, exist_ok=True)

    train_set = build_dataset(args, args.train_manifest, "train")
    val_set = build_dataset(args, args.val_manifest, "val")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
    )

    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )
    scaler = build_grad_scaler(args.device, enabled=args.amp and args.device.type == "cuda")

    metrics_path = os.path.join(args.logdir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_mAP,train_f1,val_loss,val_mAP,val_f1,best_mAP\n")

    best_map = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} | best_mAP={best_map:.4f}")
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, args, train=True, epoch=epoch)
        print(
            f"[Train] loss={train_metrics['loss']:.4f} "
            f"mAP={train_metrics['mAP']:.4f} f1={train_metrics['macro_f1']:.4f}"
        )

        val_metrics = {"loss": float("nan"), "mAP": float("nan"), "macro_f1": float("nan")}
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, args, train=False, epoch=epoch)
            print(
                f"[Val]   loss={val_metrics['loss']:.4f} "
                f"mAP={val_metrics['mAP']:.4f} f1={val_metrics['macro_f1']:.4f}"
            )
            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                save_checkpoint(
                    os.path.join(args.logdir, "best_model.pth"),
                    epoch,
                    model,
                    optimizer,
                    best_map,
                )
                print(f"Saved best checkpoint with mAP={best_map:.4f}")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},{train_metrics['loss']:.6f},{train_metrics['mAP']:.6f},"
                f"{train_metrics['macro_f1']:.6f},{val_metrics['loss']:.6f},"
                f"{val_metrics['mAP']:.6f},{val_metrics['macro_f1']:.6f},{best_map:.6f}\n"
            )


if __name__ == "__main__":
    main()
