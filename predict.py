import torch
from model import BasicNet

def load_best_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = BasicNet(
        input_shape=torch.tensor(checkpoint['input_dim'], dtype = torch.int32),
        nLayers = checkpoint['nLayers'],
        nUnits = checkpoint['nUnits']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

if __name__ == "__main__":
    model = load_best_model('saved_model/best_model.pth')
    print('Model loaded successfully')