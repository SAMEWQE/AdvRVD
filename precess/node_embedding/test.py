import timm

all_pretrained_models_available = timm.list_models(pretrained=True)

for model in all_pretrained_models_available:
    print(model)