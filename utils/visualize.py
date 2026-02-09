from matplotlib import pyplot as plt

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation loss and accuracy"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Validation Loss
    axes[0, 1].plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Training Accuracy
    axes[1, 0].plot(train_accs, 'g-', linewidth=2, label='Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Validation Accuracy
    axes[1, 1].plot(val_accs, 'orange', linewidth=2, label='Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Combined loss plot
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Combined accuracy plot
    ax2.plot(train_accs, 'g-', linewidth=2, label='Training Accuracy')
    ax2.plot(val_accs, 'orange', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('combined_training_history.png', dpi=100, bbox_inches='tight')
    plt.show()


import matplotlib.patches as patches
import numpy as np

def show_samples(X, Y, classes, n=5):
    """
    X: (N,H,W,3) images in [0,1]
    Y: (N, C+4)  one-hot + normalized box
    """

    for i in range(n):
        img = X[i]
        label = Y[i]

        class_id = np.argmax(label[:len(classes)])
        class_name = classes[class_id]

        xmin, ymin, xmax, ymax = label[len(classes):]

        h, w, _ = img.shape

        # Convert normalized → pixel coords
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.set_title(class_name)

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )

        ax.add_patch(rect)
        plt.show()

def visualize_predictions_with_boxes(model, x_test, y_test, classes,NUM_CLASSES=3, num_samples=12):
    """Visualize model predictions with bounding boxes on test images"""
    
    # Select random samples
    sample_indices = np.random.choice(len(x_test), size=min(num_samples, len(x_test)), replace=False)
    samples = x_test[sample_indices]
    true_labels = y_test[sample_indices]
    
    # Get predictions
    predictions = model.predict_with_boxes(samples)
    
    # Create figure
    rows = int(np.ceil(num_samples / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 5))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (ax, sample_idx, pred_info, true_label) in enumerate(zip(axes, sample_indices, predictions, true_labels)):
        # Get image (C, H, W) -> (H, W, C)
        img = samples[idx].transpose(1, 2, 0)
        
        # Display image
        ax.imshow(img)
        
        # Get ground truth
        true_class_idx = np.argmax(true_label[:NUM_CLASSES])
        true_class = classes[true_class_idx]
        
        # Get prediction
        pred_class_idx = pred_info['class']
        pred_class = classes[pred_class_idx]
        confidence = pred_info['confidence']
        
        # Get ground truth bounding box
        if true_label.shape[0] > NUM_CLASSES:
            true_box = true_label[NUM_CLASSES:].reshape(-1, 4)
        else:
            true_box = None
        
        # Get predicted bounding box
        pred_box = pred_info['box']
        
        # Draw ground truth box (green)
        if true_box is not None and np.any(true_box != 0):
            for box in true_box:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor='g', facecolor='none', 
                                        linestyle='--', label='Ground Truth')
                ax.add_patch(rect)
        
        # Draw predicted box (red)
        if pred_box is not None and np.any(pred_box != 0):
            for box in pred_box:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rect = patches.Rectangle((x1, y1), width, height, 
                                        linewidth=2, edgecolor='r', facecolor='none', 
                                        label='Prediction')
                ax.add_patch(rect)
        
        # Set title with prediction info
        title_color = 'green' if pred_class_idx == true_class_idx else 'red'
        title = f"True: {true_class}\nPred: {pred_class} ({confidence:.1%})"
        ax.set_title(title, color=title_color, fontsize=10)
        ax.axis('off')
        
        # Add legend
        if idx == 0:
            ax.legend(handles=[
                patches.Patch(color='g', alpha=0.5, label='Ground Truth'),
                patches.Patch(color='r', alpha=0.5, label='Prediction')
            ], loc='upper right', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions on Test Images', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Create a summary statistics plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy by class
    all_predictions = model.predict_with_boxes(x_test)
    pred_classes = [p['class'] for p in all_predictions]
    true_classes = np.argmax(y_test[:, :NUM_CLASSES], axis=1)
    
    class_accuracies = []
    for cls_idx, cls_name in enumerate(classes):
        mask = true_classes == cls_idx
        if np.sum(mask) > 0:
            correct = np.sum(np.array(pred_classes)[mask] == cls_idx)
            accuracy = correct / np.sum(mask) * 100
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    # Plot accuracy by class
    bars = axes[0].bar(range(len(classes)), class_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy by Class')
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Confusion matrix (simplified)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for pred, true in zip(pred_classes, true_classes):
        confusion[true, pred] += 1
    
    im = axes[1].imshow(confusion, cmap='Blues', aspect='auto')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    
    # Add text in confusion matrix cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = axes[1].text(j, i, confusion[i, j],
                              ha="center", va="center", 
                              color="white" if confusion[i, j] > confusion.max()/2 else "black",
                              fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig('test_statistics.png', dpi=100, bbox_inches='tight')
    plt.show()