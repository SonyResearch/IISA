import pyiqa

dependencies = ["torch", "torchvision", "gdown", "yaml", "cv2", "scipy", "huggingface_hub", "pandas"]

available_models = {
    'arniqa': 'arniqa-iisadb',
    'clipiqa': 'clipiqa+-iisadb',
    'contrique': 'contrique-iisadb',
    'dbcnn': 'dbcnn-iisadb',
    'qualiclip': 'qualiclip+-iisadb',
    'topiq': 'topiq_nr-iisadb',
}

def _make_model(model_key: str, device: str = "cpu"):
    assert model_key in available_models, f"Unsupported model {model_key}"
    return pyiqa.create_metric(available_models[model_key], device=device)


# Entrypoints
def arniqa(device: str = "cpu"):
    return _make_model("arniqa", device)

def clipiqa(device: str = "cpu"):
    return _make_model("clipiqa", device)

def contrique(device: str = "cpu"):
    return _make_model("contrique", device)

def dbcnn(device: str = "cpu"):
    return _make_model("dbcnn", device)

def qualiclip(device: str = "cpu"):
    return _make_model("qualiclip", device)

def topiq(device: str = "cpu"):
    return _make_model("topiq", device)
