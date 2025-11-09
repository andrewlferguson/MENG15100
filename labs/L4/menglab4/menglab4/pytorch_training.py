def set_pytorch_seed(seed):
    # Set the seed
  torch.manual_seed(seed)

  # If you're using CUDA (GPU)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


# Helper function
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_model_predictions_pytorch(model, X_train, y_train, X_test=None, y_test=None, n_points=400):
    """
    Plot a trained PyTorch model's predictions against the data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (e.g., nn.Sequential).
    X_train, y_train : torch.Tensor
        Training inputs and targets. Can be 1D or 2D, on CPU or GPU.
    X_test, y_test : torch.Tensor, optional
        Optional test inputs and targets to visualize.
    n_points : int
        Number of evenly spaced points for the smooth prediction curve.
    """

    def to_numpy(t):
        """Detach, move to CPU, and convert to NumPy."""
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def ensure_2d_col(t):
        """Ensure shape is (N, 1) for nn.Linear layers."""
        if t.ndim == 1:
            return t.unsqueeze(1)
        return t

    # Move to NumPy for plotting
    X_train_np = to_numpy(X_train).reshape(-1)
    y_train_np = to_numpy(y_train).reshape(-1)
    if X_test is not None and y_test is not None:
        X_test_np = to_numpy(X_test).reshape(-1)
        y_test_np = to_numpy(y_test).reshape(-1)
    else:
        X_test_np = y_test_np = None

    # Smooth x-grid for model prediction
    X_all = X_train_np if X_test_np is None else np.concatenate([X_train_np, X_test_np])
    xs_np = np.linspace(X_all.min(), X_all.max(), n_points)

    # Detect model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    xs_t = torch.from_numpy(xs_np.astype(np.float32)).to(device)
    xs_t = ensure_2d_col(xs_t)

    # Model prediction (no gradient tracking)
    with torch.no_grad():
        ys_t = model(xs_t)

    ys_np = to_numpy(ys_t).reshape(-1)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.scatter(X_train_np, y_train_np, alpha=0.6, label='Train data')
    if X_test_np is not None:
        plt.scatter(X_test_np, y_test_np, alpha=0.6, label='Test data')
    plt.plot(xs_np, ys_np, 'r-', linewidth=2, label='Model prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Helper function (execute to define helper plotting function)
def plot_pytorch_training(model, X_train, X_test, y_train, y_test, losses):  
    """
    Visualize PyTorch model training results.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model
    X_train, X_test, y_train, y_test : torch.Tensor
        Training and test data as PyTorch tensors
    losses : list of float
        Training loss history
    """
    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Convert tensors to numpy for plotting
    X_train_np = X_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: model fit on both splits
    ax1.scatter(X_train_np, y_train_np, alpha=0.6, label='Train data')
    ax1.scatter(X_test_np, y_test_np, alpha=0.6, label='Test data')

    # Smooth prediction curve across the full range
    X_all_np = np.concatenate([X_train_np, X_test_np])
    xs = np.linspace(X_all_np.min(), X_all_np.max(), 400)
    
    # Convert to tensor on the same device as model, predict, convert back to numpy
    model.eval()
    with torch.no_grad():
        xs_tensor = torch.FloatTensor(xs).reshape(-1, 1).to(device)  # Move to same device!
        ys_tensor = model(xs_tensor)
        ys = ys_tensor.cpu().numpy()
    
    ax1.plot(xs, ys, 'r-', linewidth=2, label='Model prediction')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_title('Model Fit (Train/Test)')

    # Right: training loss curve
    ax2.plot(losses)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()

def plot_circle_results(model, X_train, y_train, X_test, y_test,
                        train_losses, test_losses, train_accs, test_accs):
    """
    Visualize classification results with decision boundary and training curves.
    """
    fig = plt.figure(figsize=(12, 12))
    
    # ========================================================================
    # TOP: Decision Boundary
    # ========================================================================
    ax1 = plt.subplot(2, 1, 1)
    
    # Create mesh grid for decision boundary
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    h = 0.1  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict probability for each point in the mesh
    model.eval()
    with torch.no_grad():
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid_points).numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (color-coded by probability)
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5,
               linestyles='--')
    plt.colorbar(contour, ax=ax1, label='P(Class 1)')
    
    # Plot training data
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy().ravel()
    ax1.scatter(X_train_np[y_train_np == 0, 0], X_train_np[y_train_np == 0, 1],
               c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
               label='Train Class 0 (Inner)', marker='o')
    ax1.scatter(X_train_np[y_train_np == 1, 0], X_train_np[y_train_np == 1, 1],
               c='red', s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
               label='Train Class 1 (Outer)', marker='o')
    
    # Plot test data
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy().ravel()
    ax1.scatter(X_test_np[y_test_np == 0, 0], X_test_np[y_test_np == 0, 1],
               c='blue', s=80, alpha=0.9, edgecolors='yellow', linewidth=2.5,
               label='Test Class 0', marker='s')
    ax1.scatter(X_test_np[y_test_np == 1, 0], X_test_np[y_test_np == 1, 1],
               c='red', s=80, alpha=0.9, edgecolors='yellow', linewidth=2.5,
               label='Test Class 1', marker='s')
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('x₁ (Feature 1)', fontsize=12)
    ax1.set_ylabel('x₂ (Feature 2)', fontsize=12)
    ax1.set_title('Decision Boundary\n(Black dashed line = 50% probability)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    # Remove set_aspect('equal') to allow natural sizing
    
    # ========================================================================
    # BOTTOM LEFT: Loss Curves
    # ========================================================================
    ax2 = plt.subplot(2, 2, 3)
    
    epochs = range(len(train_losses))
    ax2.plot(epochs, train_losses, label='Train Loss',
            linewidth=2.5, alpha=0.8, color='steelblue')
    ax2.plot(epochs, test_losses, label='Test Loss',
            linewidth=2.5, alpha=0.8, color='coral')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (BCE)', fontsize=12)
    ax2.set_title('Training History - Loss', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add final loss annotation
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    ax2.text(0.98, 0.97,
            f'Final Train: {final_train_loss:.4f}\nFinal Test: {final_test_loss:.4f}',
            transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # ========================================================================
    # BOTTOM RIGHT: Accuracy Curves
    # ========================================================================
    ax3 = plt.subplot(2, 2, 4)
    
    ax3.plot(epochs, [acc * 100 for acc in train_accs],
            label='Train Accuracy', linewidth=2.5, alpha=0.8, color='steelblue')
    ax3.plot(epochs, [acc * 100 for acc in test_accs],
            label='Test Accuracy', linewidth=2.5, alpha=0.8, color='coral')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Training History - Accuracy', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # Add final accuracy annotation
    final_train_acc = train_accs[-1] * 100
    final_test_acc = test_accs[-1] * 100
    ax3.text(0.98, 0.03,
            f'Final Train: {final_train_acc:.1f}%\nFinal Test: {final_test_acc:.1f}%',
            transform=ax3.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def train_mnist(model, n_epochs=5):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import random

    def set_seed(seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model():
        return nn.Sequential(
            nn.Flatten(),          
            nn.Linear(784, 16),   
            nn.ReLU(),    
            nn.Linear(16, 10)     
        )

    def train_one_epoch(model, loader, loss_fn, optimizer, device):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        return running_loss / total, correct / total

    @torch.no_grad()
    def evaluate(model, loader, loss_fn, device):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        return running_loss / total, correct / total

    def show_samples(model, loader, device, num_samples=25):
        model.eval()
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)

        # Move to CPU for plotting
        x, y, preds = x.cpu(), y.cpu(), preds.cpu()
        x = x[:num_samples]
        y = y[:num_samples]
        preds = preds[:num_samples]

        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = x[i].squeeze(0)
            ax.imshow(img, cmap="gray")
            color = "green" if preds[i] == y[i] else "red"
            ax.set_title(f"Pred: {preds[i].item()}\nTrue: {y[i].item()}", color=color)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    set_seed(42)
    device = get_device()
    model.to(device)
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    data_root = Path("./data")

    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root=data_root,  train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False)

    loss_fn  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- history containers ---
    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []

    EPOCHS = n_epochs
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # --- plots ---
    epochs = range(1, EPOCHS + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, test_losses,  marker='s', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in train_accs], marker='o', label='Train Acc (%)')
    plt.plot(epochs, [a * 100 for a in test_accs],  marker='s', label='Test Acc (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Show 25 test images with predictions
    show_samples(model, test_loader, device, num_samples=25)

    # (Optional) return history if you want to reuse it later
    return {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs,
    }


