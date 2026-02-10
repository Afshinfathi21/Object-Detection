import numpy as np
from tqdm import tqdm, trange
from utils.load_dataset import load_dataset
from utils.visualize import show_samples,plot_training_history,visualize_predictions_with_boxes
from utils.utils import save_model_weights
import os

dataset_root = r"DATASET-ROOT"
CLASSES = ['butterfly','dalmatian','dolphin']
TARGET_SIZE = (64, 64)  # All images resized to 64x64 pixels
MAX_OBJECT = 1  # Maximum objects per image
NUM_CLASSES = 3
INPUT_CHANNELS = 3

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

x_train, x_test, y_train, y_test = load_dataset(
    dataset_root,
    classes=CLASSES,
    target_size=(64,64)
)

show_samples(x_train, y_train, CLASSES, n=5)
x_train = x_train.transpose(0, 3, 1, 2)  # (N, C, H, W)
x_test  = x_test.transpose(0, 3, 1, 2)


print("\n" + "=" * 60)
print("Dataset Statistics")
print("=" * 60)
print(f"Train images shape: {x_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")

# Check class distribution
print("\nClass distribution in training:")
unique, counts = np.unique(np.argmax(y_train[:, :NUM_CLASSES], axis=1), return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  {CLASSES[cls]}: {count} samples ({count/len(y_train)*100:.1f}%)")

class Layer:
    def __init__(self):
        self.inp = None
        self.out = None
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def step(self, lr: float, optimizer: str = 'sgd', 
             beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        pass

class Linear(Layer):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        # Xavier/Glorot initialization for better convergence
        scale = np.sqrt(2.0 / (inp_dim + out_dim))
        self.w = scale * np.random.randn(inp_dim, out_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        
        self.m_w = np.zeros_like(self.w)  # First moment for weights
        self.v_w = np.zeros_like(self.w)  # Second moment for weights
        self.m_b = np.zeros_like(self.b)  # First moment for biases
        self.v_b = np.zeros_like(self.b)  # Second moment for biases
        self.t = 0 

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        self.dw = np.dot(self.inp.T, up_grad)
        self.db = np.sum(up_grad, axis=0, keepdims=True)
        down_grad = np.dot(up_grad, self.w.T)
        return down_grad

    def step(self, lr: float, optimizer: str = 'adam', 
             beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        
        self.t += 1
        
        if optimizer.lower() == 'adam':
            self.m_w = beta1 * self.m_w + (1 - beta1) * self.dw
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.db

            self.v_w = beta2 * self.v_w + (1 - beta2) * (self.dw ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db ** 2)

            m_w_hat = self.m_w / (1 - beta1 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)

            v_w_hat = self.v_w / (1 - beta2 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)

            self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            
        elif optimizer.lower() == 'sgd':
            self.w -= lr * self.dw
            self.b -= lr * self.db

class Relu(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = up_grad * (self.inp > 0)
        return down_grad

class LeakyRelu(Layer):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        self.out = np.where(inp > 0, inp, self.alpha * inp)
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = up_grad * np.where(self.inp > 0, 1, self.alpha)
        return down_grad

class Softmax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:

        exp_inp = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_inp / np.sum(exp_inp, axis=1, keepdims=True)
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:

        return up_grad

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization for ReLU
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.w = scale * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros((out_channels, 1))
        
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

        self.m_w = np.zeros_like(self.w)  # First moment for weights
        self.v_w = np.zeros_like(self.w)  # Second moment for weights
        self.m_b = np.zeros_like(self.b)  # First moment for biases
        self.v_b = np.zeros_like(self.b)  # Second moment for biases
        self.t = 0
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        batch_size, in_channels, height, width = inp.shape

        if self.padding > 0:
            self.padded_inp = np.pad(inp, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                mode='constant')
        else:
            self.padded_inp = inp

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.out = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
 
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                
                region = self.padded_inp[:, :, h_start:h_end, w_start:w_end]
                self.out[:, :, i, j] = np.tensordot(
                    region, self.w, axes=([1, 2, 3], [1, 2, 3])
                ) + self.b.T
        
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        batch_size, in_channels, height, width = self.inp.shape
        out_height, out_width = up_grad.shape[2], up_grad.shape[3]
        
        self.dw.fill(0)
        self.db = np.sum(up_grad, axis=(0, 2, 3)).reshape(self.out_channels, 1)

        if self.padding > 0:
            down_grad_padded = np.zeros_like(self.padded_inp)
        else:
            down_grad_padded = np.zeros_like(self.inp)
 
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                
                region = self.padded_inp[:, :, h_start:h_end, w_start:w_end]
                

                for b in range(batch_size):
                    self.dw += np.outer(
                        up_grad[b, :, i, j], 
                        region[b].flatten()
                    ).reshape(self.w.shape)

                for b in range(batch_size):
                    down_grad_padded[b, :, h_start:h_end, w_start:w_end] += np.tensordot(
                        self.w, up_grad[b, :, i, j], axes=(0, 0)
                    )

        if self.padding > 0:
            down_grad = down_grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            down_grad = down_grad_padded
        
        return down_grad
    
    def step(self, lr: float, optimizer: str = 'adam',
             beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        
        self.t += 1
        
        if optimizer.lower() == 'adam':
            self.m_w = beta1 * self.m_w + (1 - beta1) * self.dw
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.db

            self.v_w = beta2 * self.v_w + (1 - beta2) * (self.dw ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db ** 2)

            m_w_hat = self.m_w / (1 - beta1 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)

            v_w_hat = self.v_w / (1 - beta2 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)

            self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            
        elif optimizer.lower() == 'sgd':
            self.w -= lr * self.dw
            self.b -= lr * self.db

class MaxPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mask = None
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        batch_size, channels, height, width = inp.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        self.out = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(inp)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size
                
                region = inp[:, :, h_start:h_end, w_start:w_end]
                self.out[:, :, i, j] = np.max(region, axis=(2, 3))

                for b in range(batch_size):
                    for c in range(channels):
                        max_val = self.out[b, c, i, j]
                        max_pos = np.where(region[b, c] == max_val)
                        if len(max_pos[0]) > 0:
                            self.mask[b, c, h_start + max_pos[0][0], w_start + max_pos[1][0]] = 1
        
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = np.zeros_like(self.inp)
        batch_size, channels, out_height, out_width = up_grad.shape
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size
                
                down_grad[:, :, h_start:h_end, w_start:w_end] += \
                    self.mask[:, :, h_start:h_end, w_start:w_end] * \
                    up_grad[:, :, i, j][:, :, None, None]
        
        return down_grad

class Flatten(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp_shape = inp.shape
        return inp.reshape(inp.shape[0], -1)
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        return up_grad.reshape(self.inp_shape)

class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, inp: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = (np.random.rand(*inp.shape) > self.p) / (1 - self.p)
            self.out = inp * self.mask
        else:
            self.out = inp
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        return up_grad * self.mask

class DetectionSplitLayer(Layer):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp

        self.cls_scores = inp[:, :self.num_classes]
        self.box_coords = inp[:, self.num_classes:]

        exp_scores = np.exp(self.cls_scores - np.max(self.cls_scores, axis=1, keepdims=True))
        self.cls_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        self.out = np.concatenate([self.cls_probs, self.box_coords], axis=1)
        return self.out
    
    def backward(self, up_grad: np.ndarray) -> np.ndarray:

        grad_cls = up_grad[:, :self.num_classes]
        grad_box = up_grad[:, self.num_classes:]
        

        grad_cls_simple = grad_cls
        
        return np.concatenate([grad_cls_simple, grad_box], axis=1)

class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None
    
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)
    
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError
    
    def backward(self) -> np.ndarray:
        raise NotImplementedError

class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target
        

        prediction = np.clip(prediction, 1e-12, 1.0 - 1e-12)

        self.loss = -np.mean(np.sum(target * np.log(prediction), axis=1))
        return self.loss
    
    def backward(self) -> np.ndarray:

        batch_size = self.prediction.shape[0]
        grad = (self.prediction - self.target) / batch_size
        return grad

class MSE(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target
        self.loss = np.mean((prediction - target) ** 2)
        return self.loss
    
    def backward(self) -> np.ndarray:
        batch_size = self.prediction.shape[0]
        grad = 2 * (self.prediction - self.target) / batch_size
        return grad

class DetectionLoss(Loss):
    def __init__(self, num_classes: int, alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = 0
        self.mse_loss = 0
    
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target

        pred_cls = prediction[:, :self.num_classes]
        pred_box = prediction[:, self.num_classes:]
        target_cls = target[:, :self.num_classes]
        target_box = target[:, self.num_classes:]

        pred_cls = np.clip(pred_cls, 1e-12, 1.0 - 1e-12)
        self.ce_loss = -np.mean(np.sum(target_cls * np.log(pred_cls), axis=1))

        self.mse_loss = np.mean((pred_box - target_box) ** 2)

        self.loss = self.ce_loss + self.alpha * self.mse_loss
        
        return self.loss
    
    def backward(self) -> np.ndarray:
        batch_size = self.prediction.shape[0]

        pred_cls = self.prediction[:, :self.num_classes]
        target_cls = self.target[:, :self.num_classes]
        pred_box = self.prediction[:, self.num_classes:]
        target_box = self.target[:, self.num_classes:]

        grad_cls = (pred_cls - target_cls) / batch_size

        grad_box = 2 * self.alpha * (pred_box - target_box) / batch_size

        grad = np.concatenate([grad_cls, grad_box], axis=1)
        
        return grad

class CNN:
    def __init__(self, layers: list, loss_fn, lr: float = 0.001, 
                 optimizer: str = 'adam', beta1: float = 0.9, beta2: float = 0.999):
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def forward(self, inp: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                inp = layer.forward(inp, training=training)
            else:
                inp = layer.forward(inp)
        return inp
    
    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.loss_fn(prediction, target)
    
    def backward(self) -> None:
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'w'):
                layer.step(self.lr, optimizer=self.optimizer, 
                          beta1=self.beta1, beta2=self.beta2)
    
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pred_classes = np.argmax(predictions[:, :NUM_CLASSES], axis=1)
        true_classes = np.argmax(targets[:, :NUM_CLASSES], axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy * 100
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> None:
        
        num_batches = len(x_train) // batch_size

        epoch_pbar = trange(epochs, desc="Training", unit="epoch", 
                           bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')
        
        for epoch in epoch_pbar:

            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0

            batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, unit="batch",
                             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
            
            for batch_idx in batch_pbar:
                start = batch_idx * batch_size
                end = start + batch_size
                
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                predictions = self.forward(x_batch, training=True)

                batch_loss = self.loss(prediction=predictions, target=y_batch)
                epoch_loss += batch_loss

                batch_acc = self.compute_accuracy(predictions, y_batch)
                epoch_acc += batch_acc

                self.backward()
                self.update()

                batch_pbar.set_postfix({
                    'batch_loss': f'{batch_loss:.4f}',
                    'batch_acc': f'{batch_acc:.2f}%'
                })
            
            batch_pbar.close()

            epoch_loss /= num_batches
            epoch_acc /= num_batches
            
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            if x_val is not None and y_val is not None:
                val_pbar = tqdm(total=1, desc="Validating", leave=False,
                               bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
                
                val_pred = self.forward(x_val, training=False)
                val_loss = self.loss(val_pred, y_val)
                val_acc = self.compute_accuracy(val_pred, y_val)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                val_pbar.update(1)
                val_pbar.close()

                epoch_pbar.set_postfix({
                    'train_loss': f'{epoch_loss:.4f}',
                    'train_acc': f'{epoch_acc:.2f}%',
                    'val_acc': f'{val_acc:.2f}%'
                })
            else:
                epoch_pbar.set_postfix({
                    'train_loss': f'{epoch_loss:.4f}',
                    'train_acc': f'{epoch_acc:.2f}%'
                })
        
        epoch_pbar.close()
        
        return self.train_losses, self.train_accs
    
    def predict_with_boxes(self, images: np.ndarray):
        """Make predictions and return class probabilities and bounding boxes"""
        predictions = self.forward(images)
        batch_size = predictions.shape[0]
        
        results = []
        for i in range(batch_size):

            cls_probs = predictions[i, :NUM_CLASSES]
            pred_class = np.argmax(cls_probs)
            confidence = cls_probs[pred_class]

            if predictions.shape[1] > NUM_CLASSES:
                box_coords = predictions[i, NUM_CLASSES:].reshape(-1, 4)
                box_coords = np.clip(box_coords, 0, 64)
            else:
                box_coords = None
            
            results.append({
                'class': pred_class,
                'confidence': confidence,
                'box': box_coords,
                'all_probs': cls_probs
            })
        
        return results
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
    

model_architecture = [
    Conv2D(3, 16, 3, padding=1),
    LeakyRelu(alpha=0.1),
    MaxPool2D(2, 2),
    
    Conv2D(16, 32, 3, padding=1),
    LeakyRelu(alpha=0.1),
    MaxPool2D(2, 2),
    
    Conv2D(32, 64, 3, padding=1),
    LeakyRelu(alpha=0.1),
    MaxPool2D(2, 2),
    
    Flatten(),
    
    Linear(8 * 8 * 64, 256),
    LeakyRelu(alpha=0.1),
    Dropout(0.3),
    
    Linear(256, 128),
    LeakyRelu(alpha=0.1),
    Dropout(0.3),
    
    Linear(128, MAX_OBJECT * (NUM_CLASSES + 4)),
    
    DetectionSplitLayer(NUM_CLASSES)
]


loss_fn = DetectionLoss(num_classes=NUM_CLASSES, alpha=0.1)
model = CNN(layers=model_architecture, loss_fn=loss_fn, 
                   lr=0.001)



# model.load_weights('model_weights_with_adam.npz')
# predictions = model.forward(x_test, training=False)
# test_acc = model.compute_accuracy(predictions, y_test)
# print(f"Loaded model test accuracy: {test_acc:.2f}%")

print("\nSplitting data for training and validation...")
split_idx = int(0.8 * len(x_train))
x_train_final = x_train[:split_idx]
y_train_final = y_train[:split_idx]
x_val_final = x_train[split_idx:]
y_val_final = y_train[split_idx:]

print(f"Training samples: {len(x_train_final)}")
print(f"Validation samples: {len(x_val_final)}")
print(f"Test samples: {len(x_test)}")


print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

EPOCHS = 50
BATCH_SIZE = 32

print(f"\nTraining Configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {model.lr}")
print(f"  Loss Function: DetectionLoss (alpha={loss_fn.alpha})")
print(f"  Model Parameters: {sum(np.prod(layer.w.shape) for layer in model_architecture if hasattr(layer, 'w')):,}")

losses, accs = model.train(
    x_train=x_train_final,
    y_train=y_train_final,
    x_val=x_val_final,
    y_val=y_val_final,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nPlotting training history...")
if len(model.train_losses) > 0 and len(model.val_losses) > 0:
    plot_training_history(model.train_losses, model.val_losses, 
                         model.train_accs, model.val_accs)

print("\nEvaluating on Test Set...")


test_pbar = tqdm(total=1, desc="Testing", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')

test_pred = model.forward(x_test, training=False)
test_acc = model.compute_accuracy(test_pred, y_test)

test_pbar.update(1)
test_pbar.close()

print(f"\nTest Accuracy: {test_acc:.2f}%")

if len(model.train_losses) > 0:
    print(f"\nFinal Training Statistics:")
    print(f"  Final Train Loss: {model.train_losses[-1]:.4f}")
    print(f"  Final Train Accuracy: {model.train_accs[-1]:.2f}%")
    
    if len(model.val_losses) > 0:
        print(f"  Final Validation Loss: {model.val_losses[-1]:.4f}")
        print(f"  Final Validation Accuracy: {model.val_accs[-1]:.2f}%")

    if len(model.train_accs) > 1:
        improvement = model.train_accs[-1] - model.train_accs[0]
        print(f"  Total Accuracy Improvement: {improvement:+.2f}%")



visualize_predictions_with_boxes(model, x_test, y_test, CLASSES, num_samples=12)


print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

print(f"\nClassification Performance:")
print(f"  Test Accuracy: {test_acc:.2f}%")

if len(model.train_accs) > 0 and len(model.val_accs) > 0:
    print(f"  Best Train Accuracy: {max(model.train_accs):.2f}%")
    print(f"  Best Validation Accuracy: {max(model.val_accs):.2f}%")
    

print(f"\nLoss Metrics:")
print(f"  Final Train Loss: {model.train_losses[-1]:.4f}")
if len(model.val_losses) > 0:
    print(f"  Final Validation Loss: {model.val_losses[-1]:.4f}")


save_model_weights(model)