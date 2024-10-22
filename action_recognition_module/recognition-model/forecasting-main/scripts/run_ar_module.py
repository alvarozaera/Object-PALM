import pprint

from ego4d_forecasting.utils import logging
import pytorch_lightning
import torch

from ego4d_forecasting.tasks.long_term_anticipation import LongTermAnticipationTask
from ego4d_forecasting.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

logger = logging.get_logger(__name__)




def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    if cfg.DATA.TASK == "long_term_anticipation":
        TaskType = LongTermAnticipationTask
    else:
        raise NotImplementedError(f"Task {cfg.DATA.TASK} not implemented.")

    task = TaskType(cfg)

    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH
    if len(ckp_path) > 0 or cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":

        if cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":
            
            # Load slowfast weights into backbone submodule
            ckpt = torch.load(
                cfg.DATA.CHECKPOINT_MODULE_FILE_PATH,
                map_location=lambda storage, loc: storage,
            )

            def remove_first_module(key):
                return ".".join(key.split(".")[1:])


            key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

            state_dict = {
                remove_first_module(k): v
                for k, v in ckpt[key].items()
                #if "head" not in k
            }

            if hasattr(task.model, 'backbone'):
                backbone = task.model.backbone
            else:
                backbone = task.model

            missing_keys, unexpected_keys = backbone.load_state_dict(
                state_dict, strict=False
            )

        else:
            # Load all child modules except for "head" if CHECKPOINT_LOAD_MODEL_HEAD is
            # False.

            pretrained = TaskType.load_from_checkpoint(ckp_path)
            state_dict_for_child_module = {
                child_name: child_state_dict.state_dict()
                for child_name, child_state_dict in pretrained.model.named_children()
            }
            for child_name, child_module in task.model.named_children():
                if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
                    continue

                logger.info(f"Loading in {child_name}")
                state_dict = state_dict_for_child_module[child_name]
                missing_keys, unexpected_keys = child_module.load_state_dict(state_dict)
                assert len(missing_keys) + len(unexpected_keys) == 0 



    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="max", save_last=True, save_top_k=1
    )

    checkpoint_callback_extra = None
    if hasattr(task, "checkpoint_metric_extra"):
        checkpoint_callback_extra = ModelCheckpoint(
        monitor=task.checkpoint_metric_extra, mode="max", save_last=True, save_top_k=1
    )
      
    if cfg.ENABLE_LOGGING:
        if checkpoint_callback_extra is not None:
            args = {"callbacks": [LearningRateMonitor(), checkpoint_callback, checkpoint_callback_extra]}
        else:
            args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        if checkpoint_callback_extra is not None:
            args = {"logger": False, "callbacks": [checkpoint_callback, checkpoint_callback_extra]}
        else:
            args = {"logger": False, "callbacks": checkpoint_callback}
    

    trainer = Trainer(
        #resume_from_checkpoint=ckp_path,
        gpus=cfg.NUM_GPUS,
        num_nodes=cfg.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        #num_sanity_val_steps=3,
        num_sanity_val_steps=0,
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,
        fast_dev_run=cfg.FAST_DEV_RUN,
        default_root_dir=cfg.OUTPUT_DIR,
        plugins=DDPPlugin(find_unused_parameters=False),
        **args,
    )

    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)
        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    main(cfg)
