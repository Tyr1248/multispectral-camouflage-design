import torch
def save_model(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """加载模型权重"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model