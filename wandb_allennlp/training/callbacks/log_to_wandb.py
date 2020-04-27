import logging
import wandb
from typing import Any, Dict, List
from allennlp.training import EpochCallback

logger = logging.getLogger(__name__)

@EpochCallback.register('log_metrics_to_wandb')
class LogMetricsToWandb(EpochCallback):
    def __call__(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int) -> None:
        if epoch== -1:
            logger.info("Writing metrics to wandb {}".format(epoch))
        else:
            logger.info("Writing metrics to wandb {}".format(epoch))
            wandb.log(metrics)
