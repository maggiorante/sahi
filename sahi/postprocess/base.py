from sahi.postprocess.interface import IPostProcessor
from sahi.utils.import_utils import check_requirements


class BasePostProcessor(IPostProcessor):
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

        check_requirements(["torch"])
    