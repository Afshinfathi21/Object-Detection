import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_dataset(root_dir, classes, test_size=0.2, target_size=(128, 128)):
    data = []
    targets = []
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        txt_path = os.path.join(root_dir, f"res_{cls_name}.txt")
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found")
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Loading {cls_name}"):

            parts = line.strip().split(',')
            if len(parts) != 5:
                continue

            annot_file, xmin, ymin, xmax, ymax = parts
            xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
            img_file = annot_file.replace("annotation", "image").replace(".mat", ".jpg")
            img_path = os.path.join(root_dir, cls_name, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue

            # Load the image and get original size
            img = Image.open(img_path).convert('RGB')
            original_w, original_h = img.size

            # Resize image to target size
            img_resized = img.resize(target_size)
            new_w, new_h = target_size

            # Convert to numpy
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            n_xmin = xmin / original_w
            n_ymin = ymin / original_h
            n_xmax = xmax / original_w
            n_ymax = ymax / original_h


            # Clip safety
            n_xmin = np.clip(n_xmin, 0, 1)
            n_ymin = np.clip(n_ymin, 0, 1)
            n_xmax = np.clip(n_xmax, 0, 1)
            n_ymax = np.clip(n_ymax, 0, 1)


            # One-hot class vector
            one_hot = np.zeros(len(classes), dtype=np.float32)
            one_hot[class_to_idx[cls_name]] = 1.0

            y_vector = np.concatenate([one_hot, [n_xmin, n_ymin, n_xmax, n_ymax]])

            data.append(img_array)
            targets.append(y_vector)

    if len(data) == 0:
        raise ValueError("No images found. Check your dataset structure and paths!")

    X = np.stack(data, axis=0)
    Y = np.stack(targets, axis=0)

    # Shuffle and split
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(num_samples * (1 - test_size))

    x_train = X[indices[:split_idx]]
    y_train = Y[indices[:split_idx]]
    x_test = X[indices[split_idx:]]
    y_test = Y[indices[split_idx:]]

    print(f"Loaded {len(x_train)} train samples and {len(x_test)} test samples")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    return x_train, x_test, y_train, y_test
