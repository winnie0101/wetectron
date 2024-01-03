import logging
from .ceymo_eval import do_ceymo_evaluation

def ceymo_evaluation(dataset, predictions, output_folder, box_only, task='det', **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    if task == 'det':
        return do_ceymo_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )