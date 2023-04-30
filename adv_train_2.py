import textattack
import transformers

model = textattack.models.helpers.LSTMForClassification.from_pretrained('../../lstm_pwws/run-32922848/outputs/2023-04-30-12-02-50-951503/last_model')
#model = textattack.models.helpers.LSTMForClassification.from_pretrained('lstm-sst2')
model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)

# We only use DeepWordBugGao2018 to demonstration purposes.
attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
train_dataset = textattack.datasets.HuggingFaceDataset("sst2", split="train")
eval_dataset = textattack.datasets.HuggingFaceDataset("sst2", split="validation")

# Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
training_args = textattack.TrainingArgs(
    num_epochs=10,
    num_clean_epochs=0,
    num_train_adv_examples=3000,
    learning_rate=5e-5,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification",
    attack,
    train_dataset,
    eval_dataset,
    training_args
)
trainer.train()
