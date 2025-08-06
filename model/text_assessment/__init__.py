SEED = 20040508
PRETRAINED_MODEL_NAME = './model/bert-base-uncased'
SAVED_MODEL = './data/text_assessor.pth'
MAX_LEN = 256

from .text_dataset import TextDataset
from .text_assessor import TextAssessor

__all__ = ['TextDataset', 'TextAssessor', "SEED", "PRETRAINED_MODEL_NAME"]