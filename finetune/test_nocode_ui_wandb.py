import wandb

wandb.init(project="annotation-training")
wandb.config.update({"epochs": 50, "lr": 0.001})

# Automatic logging during training
wandb.log({"loss": loss, "mAP": map_score})

# View at: https://wandb.ai/your-project