from SeqRec.models.discriminative.SASRec.config import SASRecConfig


class SASRecCVRConfig(SASRecConfig):
    loss_type: str = "BCE"

