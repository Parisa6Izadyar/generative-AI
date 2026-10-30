"""Resume training from completed epochs, including optimizer and RNG state."""

import json
from pathlib import Path

import torch


def prepare_progress(output, config, data_manifest, source_hashes, resume, overwrite):
    progress = Path(output) / "progress"
    manifest = {"config": config, "data": data_manifest, "source_sha256": source_hashes}
    path = progress / "run.json"
    if resume:
        if not path.exists():
            raise ValueError("No saved training progress exists in this output directory.")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Resume requires unchanged configuration, data and source files.")
    else:
        progress.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for name in ("judge.pt", "vae.pt", "gan.pt"):
                (progress / name).unlink(missing_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return progress


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_stage(name, models, loader, train_function, epochs, device, progress, resume):
    path = progress / f"{name}.pt"
    start_epoch, history, optimizer_state = 0, [], None
    if resume and path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        start_epoch, history = saved["completed_epochs"], saved["history"]
        if len(history) != start_epoch or not 0 <= start_epoch <= epochs:
            raise ValueError(f"Invalid saved epoch count for {name}.")
        for key, model in models.items():
            model.load_state_dict(saved["models"][key])
        optimizer_state = saved["optimizer"]
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        loader.generator.set_state(saved["loader_rng"])
        print(f"Resume {name}: {start_epoch}/{epochs} epochs complete", flush=True)

    def record_epoch(completed, optimizer, history):
        optimizer_state = ({key: item.state_dict() for key, item in optimizer.items()}
                           if isinstance(optimizer, dict) else optimizer.state_dict())
        save_checkpoint(path, {
            "completed_epochs": completed, "history": history,
            "models": {key: model.state_dict() for key, model in models.items()},
            "optimizer": optimizer_state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "loader_rng": loader.generator.get_state(),
        })

    return train_function(*models.values(), loader, epochs, device, start_epoch=start_epoch,
                          history=history, optimizer_state=optimizer_state, on_epoch_end=record_epoch)
