from SeqRec.models.discriminative.SASRec.config import SASRecConfig


class BSTCVRConfig(SASRecConfig):
    loss_type: str = "BCE"

