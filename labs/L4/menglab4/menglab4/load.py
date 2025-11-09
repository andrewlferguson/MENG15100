
def generate_circle_data(n_samples=500, noise=0.0, random_state=42):
    """
    Generate circle dataset (inner circle = class 0, outer ring = class 1).
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    noise : float
        Standard deviation of Gaussian noise (0.0 to 0.5)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Binary labels (0 or 1)
    """
    import numpy as np

    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Inner circle (class 0)
    r_inner = np.random.uniform(0, 2, n_per_class)
    theta_inner = np.random.uniform(0, 2*np.pi, n_per_class)
    X_inner = np.column_stack([
        r_inner * np.cos(theta_inner),
        r_inner * np.sin(theta_inner)
    ])
    y_inner = np.zeros(n_per_class)
    
    # Outer ring (class 1)
    r_outer = np.random.uniform(3, 5, n_per_class)
    theta_outer = np.random.uniform(0, 2*np.pi, n_per_class)
    X_outer = np.column_stack([
        r_outer * np.cos(theta_outer),
        r_outer * np.sin(theta_outer)
    ])
    y_outer = np.ones(n_per_class)
    
    # Combine
    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer])
    
    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    # Normalize to [-6, 6] range (matching playground)
    X = (X / 5) * 6
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    return X, y

def plot_circle_dataset(X, y, title="Dataset", ax=None):
    """
    Plot a 2D classification dataset.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        Feature matrix
    y : np.ndarray, shape (n_samples,)
        Binary labels
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot class 0 and class 1
    ax.scatter(X[y == 0, 0], X[y == 0, 1], 
              c='steelblue', s=30, alpha=0.6, 
              edgecolors='black', linewidth=0.5, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], 
              c='coral', s=30, alpha=0.6,
              edgecolors='black', linewidth=0.5, label='Class 1')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

def generate_circle_train_test_split(dataset_name='circle', n_samples=500, 
                                noise=0.0, test_split=0.2, random_state=42):
    """
    Generate and prepare classification data for training.
    
    Parameters
    ----------
    dataset_name : str
        One of: 'circle', 'xor', 'gaussian', 'spiral'
    n_samples : int
        Total number of samples
    noise : float
        Noise level (0.0 to 0.5)
    test_split : float
        Fraction of data to use for testing
    random_state : int
        Random seed
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test splits
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # Generate data
    generators = {
        'circle': generate_circle_data,
    }
    
    if dataset_name not in generators:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from: {list(generators.keys())}")
    
    X, y = generators[dataset_name](n_samples=n_samples, 
                                    noise=noise, 
                                    random_state=random_state)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
