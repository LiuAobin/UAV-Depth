import wandb
import numpy
wandb.init(project="test-project")
# log some values
wandb.define_metric("accuracy", summary="max")
wandb.define_metric("loss", summary="min")
for i in range(10):
    wandb.log({"loss": i * 0.1},step=i)
    image = wandb.Image(numpy.random.rand(256,256,3), caption='test')
    wandb.log({'test':image},step=i)
    wandb.log({"accuracy": i * 0.01},step=i)
