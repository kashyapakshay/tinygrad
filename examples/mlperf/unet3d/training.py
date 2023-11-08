from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters

from models.unet3d import UNet3D
from extra.datasets.kits19 import iterate, sliding_window_inference
from examples.mlperf.metrics import get_dice_score
from examples.mlperf.unet3d.data.loader import get_data_loaders
from examples.mlperf.unet3d.losses import dice_ce_loss

init_lr, lr = 1e-4, 0.8
max_epochs = 32
Tensor.training = True
train_loader, val_loader = get_data_loaders('extra/datasets/kits19/processed_data')
model = UNet3D()
optimizer = optim.SGD(get_parameters(model), lr=init_lr)
for i, (X, y_true) in enumerate(tqdm(train_loader)):
	optimizer.zero_grad()

	X = Tensor(X).half()
	y_pred = model(X)

	loss = dice_ce_loss(y_pred, y_true)
	loss.backward()
	optimizer.step()