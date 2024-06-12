import numpy as np

from tinygrad.tensor import Tensor
from examples.mlperf.metrics import one_hot, get_dice_score


def dice_ce_loss(y_pred, y_true, use_softmax=True, to_onehot_y=False):
	assert isinstance(y_pred, Tensor)
	if use_softmax: y_pred = y_pred.softmax(axis=1)
	if to_onehot_y: y_pred = one_hot(y_pred)
	dice_loss = (1.0 - get_dice_score(y_pred, y_true)).mean()
	ce_loss = -y_true.mul(y_pred.log()).mean()
	return (dice_loss + ce_loss) / 2
