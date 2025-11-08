from subprocess import check_output
from datetime import datetime
import torch
from src.config.run import RunCfg

def _get_git_commit_short_id():
    try:
        commit_id = check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        return commit_id
    except Exception as e:
        return f"unknown (error: {e})"


def finalise(config: RunCfg) -> RunCfg:
    from src.config.io import load_cfg
    from src.utils.utils import get_palette_size
    from pathlib import Path
    from dataclasses import replace

    # finalise the num_visible channels based on training mode and palette size
    if config.training.mode == "discrete":
        palsize = get_palette_size(Path(config.target.target_image))
        num_visible = palsize - 1 # -1 because we are removing the alpha channel
    elif config.training.mode == "continuous":
        num_visible = 3 # RGB

    final = replace(config, model=replace(config.model, num_visible=num_visible))
    final = replace(final, model=replace(final.model, num_channels=final.model.num_visible + final.model.num_hidden + 1)) # +1 for alpha channel

    # now set the device
    print(f"Configuring device: {config.compute.requested_device}")

    if config.compute.requested_device == 'auto':
        if torch.backends.mps.is_available():
            dvc = torch.device("mps")
            print("MPS device found.")
        elif torch.cuda.is_available():
            dvc = torch.device("cuda")
            print("CUDA device found.")
        else:
            dvc = torch.device("cpu")
            print("No GPU found, using CPU.")
    elif config.compute.device_name == 'mps' and not torch.backends.mps.is_available():
        print ("Metal Performance Shaders (MPS) requested but not available, falling back to CPU.")
        dvc = torch.device('cpu')
    elif config.compute.device_name == 'cuda' and not torch.cuda.is_available():
        print ("CUDA requested but not available, falling back to CPU.")
        dvc = torch.device('cpu')
    else:
        dvc = torch.device(dvc)
    
    # update the config with the actual device
    final = replace(final, compute=replace(final.compute, device=dvc))

    # now set the reuslts output dir
    now = datetime.now()
    run_output_dir = Path(
        config.results.output_dir,
        f"{now:%Y-%m-%d}/{now:%H%M%S}_{config.experiment.name}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    final = replace(final, results=replace(final.results, output_dir=str(run_output_dir)))

    # set the experiment version to the current git commit short id
    final = replace(final, experiment=replace(final.experiment, version_used=_get_git_commit_short_id()))

    # TODO save the configuration

    return final