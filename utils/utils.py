import numpy as np
import os
def save_model_weights(model, filename='model_weights.npz'):
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'w'):
            weights_dict[f'layer_{i}_w'] = layer.w
        if hasattr(layer, 'b'):
            weights_dict[f'layer_{i}_b'] = layer.b
    
    np.savez(filename, **weights_dict)
    print(f"\nModel weights saved to {filename}")


def load_weights(self, filepath: str):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")
        
    data = np.load(filepath, allow_pickle=True)
        
    print(f"Loading weights from {filepath}")
        
    for i, layer in enumerate(self.layers):
        if f'layer_{i}_w' in data:
            layer.w = data[f'layer_{i}_w']
            print(f"  Loaded weights for layer {i}")
        if f'layer_{i}_b' in data:
            layer.b = data[f'layer_{i}_b']
            print(f"  Loaded biases for layer {i}")
        
    print("Weights loaded successfully!")

def load_image_for_prediction(image_path, target_size=(64, 64)):
    from PIL import Image

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_array = image_array.transpose(2, 0, 1)
    return np.expand_dims(image_array, axis=0)
