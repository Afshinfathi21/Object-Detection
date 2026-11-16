import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_annotation(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as file:
        content = file.read()

    image_match = re.search(r'Image filename\s*:\s*"([^"]+)"', content)
    if image_match is None:
        raise ValueError(f"Image filename not found in {annotation_path}")

    image_filename = os.path.basename(image_match.group(1))

    boxes = re.findall(
        r'Bounding box for object \d+ .*?\(Xmin, Ymin\)\s*-\s*\(Xmax, Ymax\)\s*:\s*'
        r'\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
        content
    )

    bounding_boxes = [
        tuple(map(int, box))
        for box in boxes
    ]

    return image_filename, bounding_boxes


def load_penn_fudan_dataset(
    root_dir,
    test_size=0.2,
    target_size=(64, 64),
    random_seed=42
):
    image_dir = os.path.join(root_dir, "PNGImages")
    annotation_dir = os.path.join(root_dir, "Annotation")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    image_files = sorted(
        filename for filename in os.listdir(annotation_dir)
        if filename.lower().endswith(".txt")
    )

    if not image_files:
        raise ValueError(f"No annotation files found in {annotation_dir}")

    samples = []

    for annotation_file in tqdm(image_files, desc="Loading Penn-Fudan"):
        annotation_path = os.path.join(annotation_dir, annotation_file)

        image_filename, boxes = parse_annotation(annotation_path)
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not boxes:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            resized_image = image.resize(target_size)
            image_array = np.asarray(resized_image, dtype=np.float32) / 255.0

        xmin, ymin, xmax, ymax = max(
            boxes,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
        )

        box = np.array([
            xmin / original_width,
            ymin / original_height,
            xmax / original_width,
            ymax / original_height
        ], dtype=np.float32)

        box = np.clip(box, 0.0, 1.0)

        if box[0] >= box[2] or box[1] >= box[3]:
            continue

        class_label = np.array([1.0], dtype=np.float32)
        target = np.concatenate((class_label, box))

        samples.append((image_filename, image_array, target))

    if not samples:
        raise ValueError("No valid annotated objects were found.")

    unique_images = sorted({sample[0] for sample in samples})
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_images)

    split_index = int(len(unique_images) * (1 - test_size))
    split_index = min(max(split_index, 1), len(unique_images) - 1)

    train_images = set(unique_images[:split_index])
    test_images = set(unique_images[split_index:])

    train_samples = [sample for sample in samples if sample[0] in train_images]
    test_samples = [sample for sample in samples if sample[0] in test_images]

    x_train = np.stack([sample[1] for sample in train_samples]).astype(np.float32)
    y_train = np.stack([sample[2] for sample in train_samples]).astype(np.float32)
    x_test = np.stack([sample[1] for sample in test_samples]).astype(np.float32)
    y_test = np.stack([sample[2] for sample in test_samples]).astype(np.float32)

    print(f"Training images: {len(train_images)}")
    print(f"Testing images: {len(test_images)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    return x_train, x_test, y_train, y_test
