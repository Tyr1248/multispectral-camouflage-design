import torch
def save_model(model, path):
    """Save model weights"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load model weights"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model