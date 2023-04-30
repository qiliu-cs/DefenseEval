import textattack

model = textattack.models.helpers.LSTMForClassification.from_pretrained('../../lstm_pwws/run-32922848/outputs/2023-04-30-12-02-50-951503/last_model/')
model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)

dataset = textattack.datasets.HuggingFaceDataset("sst2", split="validation")

attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)

attack_args = textattack.AttackArgs(
    num_examples=1000,
)

attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
