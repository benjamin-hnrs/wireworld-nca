from pathlib import Path
from matplotlib.pyplot import step
import torch
from src.logging.logger import get_logger, init_logger, set_context
from src.training.metrics import exact_rgb_accuracy, mse_similarity, palette_snapped_exact_rgb_accuracy
from src.utils.utils import nchw_display, nchw_zoom, rgba_tensor_to_pil, state_to_rgba, to_rgba
from src.visualisation.channel_grid import channel_headers, channel_strip
from src.visualisation.nca_video import generate_nca_video
from src.visualisation.pool_video import generate_pool_video

class TrainerCallback:
    def on_train_start(self, trainer): pass
    def on_train_step_start(self, trainer, step): pass
    
    def on_rollout_start(self, trainer, train_step): pass
    def on_rollout_step_start(self, trainer, train_step, ca_step, x): pass
    def on_rollout_step_end(self, trainer, train_step, ca_step, x): pass
    def on_rollout_end(self, trainer, train_step, ca_step, x): pass
    
    def on_train_step_end(self, trainer, step, pool, x0, x, loss): pass
    def on_train_end(self, trainer, step): pass


class VisualisationCallback(TrainerCallback):
    @torch.no_grad()
    def on_train_start(self, trainer):
        # save target image
        if trainer.config.training.mode == "discrete":
            rgba = state_to_rgba(trainer.target, trainer.palette, trainer.config.model.num_visible, trainer.config.model.alive_threshold, alpha_mode='clamped')
        elif trainer.config.training.mode == "continuous":
            rgba = to_rgba(trainer.target)
        # TODO does this need to call the logger?
        rgba_tensor_to_pil(rgba).save(get_logger().images_dir / "target.png")

        # preview the target image
        if trainer.config.misc.preview_target_image:
            nchw_display(nchw_zoom(rgba, 2), fmt="png")

        # save a header for the rollout channel image
        if trainer.config.training.mode == "discrete":
            img = channel_headers(
                num_channels=trainer.model.num_channels,
                num_visible=trainer.model.num_visible,
                width=trainer.target.shape[2],
                palette=trainer.palette,
                spacing=trainer.config.results.channel_strip.spacing,
                font_size=trainer.config.results.channel_strip.font_size
            )
            get_logger().save_image(img, get_logger().images_dir / "channels/ch_.png")

    def on_rollout_step_end(self, trainer, train_step, ca_step, x):
        # produce an image of the channels at this step
        if (trainer.config.results.channel_strip.enabled and
                train_step % trainer.config.results.channel_strip.interval == 0 and
                train_step == trainer.config.results.channel_strip.for_train_step
            ):
            img = channel_strip(x[0], cmap_name=trainer.config.results.channel_strip.cmap_name)
            get_logger().save_image(img, get_logger().images_dir / f"channels/ch_{train_step}_{ca_step:05d}.png")


    def on_train_step_end(self, trainer, step, pool, x0, x, loss):
        # log pool images
        if (
            step % trainer.config.results.pool_images.interval == 0
            and trainer.config.training.pool_size > 0
            and trainer.config.results.pool_images.enabled
        ):
            with torch.no_grad():
                get_logger().save_pool_image(pool, discrete=trainer.mode=="discrete", num_visible=trainer.model.num_visible, palette=trainer.palette, alive_threshold=trainer.config.model.alive_threshold)

        # log batch images
        if (
            trainer.config.results.batch_images.enabled
            and step % trainer.config.results.batch_images.interval == 0
        ):
            get_logger().save_batch_image(
                x0,
                x,
                step,
                discrete=trainer.mode == "discrete",
                num_visible=trainer.model.num_visible if trainer.model.num_visible > 0 else None,
                palette=trainer.palette if trainer.model.num_visible > 0 else None,
                alive_threshold=trainer.config.model.alive_threshold
            )

    @torch.no_grad()
    def on_train_end(self, trainer, step):
        if trainer.config.results.pool_images.video:
            generate_pool_video(
                input_dir=Path(trainer.config.results.output_dir) / "images/pool",
                output_path=Path(trainer.config.results.output_dir) / "videos/pool.mp4",
                pattern="*_pool.png",
                fps=trainer.config.results.pool_images.fps,
                scale=1.0,
            )
        if trainer.config.results.channel_strip.enabled:
            from src.visualisation.channel_grid import generate_channel_grid

            generate_channel_grid(
                input_dir=Path(trainer.config.results.output_dir) / "images/channels",
                output_path=Path(trainer.config.results.output_dir) / "images/channels/grid.png",
            )
        if trainer.config.results.ca_rollout.enabled:
            # First try to use the best checkpoint, fallback to latest if not available
            checkpoints_dir = Path(trainer.config.results.output_dir) / "checkpoints"

                # Fallback to latest checkpoint
            checkpoints = list(checkpoints_dir.glob("checkpoint_*.pth"))
            if checkpoints:
                checkpoint_to_use = max(checkpoints, key=lambda x: x.stat().st_mtime)
                is_discrete = trainer.config.training.mode == "discrete"
                if is_discrete:
                    # trainer = cast(DiscreteTrainer, trainer)
                    p = trainer.palette
                else:  # gnca
                    p = None
                generate_nca_video(
                    config=trainer.config,
                    checkpoint_path=checkpoint_to_use,
                    output_path=Path(trainer.config.results.output_dir) / "videos/nca.mp4",
                    grid_size=trainer.config.target.target_size
                    + 2 * trainer.config.target.target_padding,
                    num_channels=trainer.config.model.num_channels,
                    n_steps=trainer.config.results.ca_rollout.ca_steps,
                    fps=trainer.config.results.ca_rollout.fps,
                    scale=4,
                    discrete=is_discrete,
                    palette=p
                )
            else:
                print(f"No checkpoints found in {checkpoints_dir}, skipping video generation")


class LoggingCallback(TrainerCallback):
    def on_train_start(self, trainer):
        log = init_logger(trainer.config.results.output_dir, run_id=str(trainer.config.results.output_dir))
        
        # bit hacky, but oh well.
        set_context(
            target_image=str(trainer.config.target.target_image),
            target_size=str(trainer.config.target.target_size),
            target_padding=str(trainer.config.target.target_padding),
            training_mode=str(trainer.config.training.mode),
            hidden_channels=str(trainer.config.model.num_hidden),
            lr=str(trainer.config.training.learning_rate),
            batch_size=str(trainer.config.training.batch_size),
            pool_size=str(trainer.config.training.pool_size),
            seed_in_batch=str(trainer.config.training.replace_with_seed),
            damage_in_batch=str(trainer.config.training.num_to_damage),
            rollout_min=str(trainer.config.training.rollout.min),
            rollout_max=str(trainer.config.training.rollout.max),
            step_size_warmup_steps=str(trainer.config.training.rollout.warmup_steps),
            starting_step_size=str(trainer.config.training.rollout.starting_step_size),
            milestones=str(trainer.config.training.scheduler.milestones),
            decay_factor=str(trainer.config.training.scheduler.decay_factor),
            comment=str(trainer.config.experiment.comment),
            phase="train",
        )
        log.enable_tensorboard(flush_secs=5)


    def on_rollout_start(self, trainer, train_step):
        set_context(phase="train/rollout")

    def on_rollout_step_start(self, trainer, train_step, ca_step, x):
        set_context(ca_step=ca_step)

    def on_rollout_end(self, trainer, train_step, ca_step, x):
        set_context(phase="train")

    def on_train_step_end(self, trainer, step, pool, x0, x, loss):
        set_context(step=step)
        if step % trainer.config.results.metrics.interval == 0:
            grad_norm = sum(p.grad.norm().item() for p in trainer.model.parameters() if p.grad is not None)

            # # TODO: why have I only been calculating metrics on the first sample in the batch?
            # if trainer.config.training.pool_size > 0:
            #     pred = x[:1]
            # else:
            #     pred = x
            pred = x
            tgt = trainer.target.expand(pred.shape[0], -1, -1, -1) # expand (virtual repeat)

            if trainer.config.training.mode == "discrete":
                x_rgba = state_to_rgba(pred, trainer.palette, trainer.model.num_visible, trainer.config.model.alive_threshold, alpha_mode='clamped')
                y_rgba = state_to_rgba(tgt, trainer.palette, trainer.model.num_visible, trainer.config.model.alive_threshold, alpha_mode='clamped')
            else:
                x_rgba = to_rgba(pred)
                y_rgba = to_rgba(tgt)

            mse = mse_similarity(x_rgba, y_rgba)
            acc_rgb = exact_rgb_accuracy(x_rgba, y_rgba, use_uint8_compare=True)
            acc_snap_rgb = palette_snapped_exact_rgb_accuracy(x_rgba, y_rgba, trainer.palette)

            get_logger().log_metrics(loss=loss, mse=mse, acc=acc_rgb, acc_snap=acc_snap_rgb, grad_norm=grad_norm)

            # hacky
            if trainer.config.training.early_stopping.enabled:
                if trainer.config.training.early_stopping.measure == "mse":
                    trainer.recent_es_measures.append(mse)
                elif trainer.config.training.early_stopping.measure == "loss":
                    trainer.recent_es_measures.append(loss)
                elif trainer.config.training.early_stopping.measure == "accuracy":
                    trainer.recent_es_measures.append(acc_rgb)
                elif trainer.config.training.early_stopping.measure == "accuracy_snap":
                    trainer.recent_es_measures.append(acc_snap_rgb)


    def on_train_end(self, trainer, step):
        log = get_logger()
        log.close()


class CheckpointCallback(TrainerCallback):
    def on_train_step_end(self, trainer, step, pool, s0, x, loss):
        if step % trainer.config.results.model_checkpoint.interval == 0:
            get_logger().save_model_checkpoint(trainer.model, step)

    def on_train_end(self, trainer, step):
        get_logger().save_model_checkpoint(trainer.model, step)