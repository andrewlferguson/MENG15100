def interactive_activation_functions():
  import numpy as np
  import matplotlib.pyplot as plt
  from ipywidgets import interact, FloatSlider, Dropdown, VBox, HBox, Output
  import ipywidgets as widgets

  def sigmoid(z):
      """Sigmoid activation function"""
      return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

  def relu(z):
      """ReLU activation function"""
      return np.maximum(0, z)

  def create_activation_explorer():
      """Create interactive widget for exploring activation functions"""
      
      # Create widgets
      activation_dropdown = Dropdown(
          options=['sigmoid', 'relu'],
          value='sigmoid',
          description='Activation:',
          style={'description_width': '100px'}
      )
      
      w_slider = FloatSlider(
          value=1.0,
          min=-3.0,
          max=3.0,
          step=0.1,
          description='w (slope):',
          style={'description_width': '100px'},
          continuous_update=True
      )
      
      b_slider = FloatSlider(
          value=0.0,
          min=-5.0,
          max=5.0,
          step=0.1,
          description='b (shift):',
          style={'description_width': '100px'},
          continuous_update=True
      )
      
      # Output widget for plot
      output = Output()
      
      def update_plot(activation, w, b):
          """Update plot based on widget values"""
          # Create x values
          x = np.linspace(-10, 10, 500).reshape(-1, 1)
          
          # Compute linear transformation
          z = w * x + b
          
          # Apply activation function
          if activation == 'sigmoid':
              y = sigmoid(z)
              activation_func = sigmoid
              y_label = 'σ(w·x + b)'
          else:  # relu
              y = relu(z)
              activation_func = relu
              y_label = 'ReLU(w·x + b)'
          
          # Clear and create new plot
          with output:
              output.clear_output(wait=True)
              
              fig, axes = plt.subplots(1, 3, figsize=(15, 4))
              
              # Plot 1: Linear transformation (before activation)
              ax1 = axes[0]
              ax1.plot(x, z, 'b-', linewidth=2)
              ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax1.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax1.grid(True, alpha=0.3)
              ax1.set_xlabel('x', fontsize=12)
              ax1.set_ylabel('z = w·x + b', fontsize=12)
              ax1.set_title('Step 1: Linear Transformation', fontsize=13, fontweight='bold')
              ax1.set_ylim([-10, 10])
              
              # Add annotation for the kink point (where z=0)
              if w != 0:
                  x_kink = -b / w
                  if -10 <= x_kink <= 10:
                      ax1.plot(x_kink, 0, 'ro', markersize=10, label=f'z=0 at x={x_kink:.2f}')
                      ax1.legend()
              
              # Plot 2: Activation function
              ax2 = axes[1]
              z_range = np.linspace(-10, 10, 500)
              y_range = activation_func(z_range)
              ax2.plot(z_range, y_range, 'g-', linewidth=2)
              ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax2.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax2.grid(True, alpha=0.3)
              ax2.set_xlabel('z', fontsize=12)
              ax2.set_ylabel(f'{activation}(z)', fontsize=12)
              ax2.set_title(f'Step 2: {activation.capitalize()} Activation', fontsize=13, fontweight='bold')
              
              if activation == 'sigmoid':
                  ax2.set_ylim([-0.1, 1.1])
                  # Mark important points
                  ax2.plot(0, 0.5, 'ro', markersize=8, label='Midpoint (0, 0.5)')
                  ax2.legend()
              else:
                  ax2.set_ylim([-1, 10])
                  # Mark the kink
                  ax2.plot(0, 0, 'ro', markersize=8, label='Kink at (0, 0)')
                  ax2.legend()
              
              # Plot 3: Final output
              ax3 = axes[2]
              ax3.plot(x, y, 'r-', linewidth=3)
              ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax3.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
              ax3.grid(True, alpha=0.3)
              ax3.set_xlabel('x', fontsize=12)
              ax3.set_ylabel(y_label, fontsize=12)
              ax3.set_title('Step 3: Final Output', fontsize=13, fontweight='bold')
              
              if activation == 'sigmoid':
                  ax3.set_ylim([-0.1, 1.1])
              else:
                  ax3.set_ylim([-1, 10])
              
              plt.tight_layout()
              plt.show()
      
      # Create layout
      controls = VBox([
          activation_dropdown,
          w_slider,
          b_slider
      ])
      
      # Connect widgets to update function
      interactive_plot = interact(
          update_plot,
          activation=activation_dropdown,
          w=w_slider,
          b=b_slider
      )
      
      # Display output
      display(output)
      
      # Initial plot
      update_plot('sigmoid', 1.0, 0.0)

  # Create the widget
  create_activation_explorer()

def interactive_single_layer_model():
  import numpy as np
  import matplotlib.pyplot as plt
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import TensorDataset, DataLoader
  import ipywidgets as widgets
  from IPython.display import display, clear_output

  # Set random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  # ============================================================================
  # Generate Gaussian Bumps Dataset
  # ============================================================================

  def generate_gaussian_bumps(n_samples=200, noise_level=0.1):
      """Generate multiple Gaussian bumps dataset."""
      X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
      y = (np.exp(-((X + 2)**2) / 1.0) +
          0.8 * np.exp(-((X - 1)**2) / 0.5) +
          0.6 * np.exp(-((X - 3)**2) / 2.0))
      y += noise_level * np.random.randn(n_samples, 1)
      return X, y


  def prepare_data(n_samples=200, noise_level=0.1, test_split=0.2):
      """Generate and split data into train/test sets."""
      X, y = generate_gaussian_bumps(n_samples=n_samples, noise_level=noise_level)

      # Shuffle
      indices = np.random.permutation(n_samples)
      X, y = X[indices], y[indices]

      # Split
      n_train = int(n_samples * (1 - test_split))
      X_train, X_test = X[:n_train], X[n_train:]
      y_train, y_test = y[:n_train], y[n_train:]

      # Convert to PyTorch tensors
      X_train = torch.FloatTensor(X_train)
      X_test = torch.FloatTensor(X_test)
      y_train = torch.FloatTensor(y_train)
      y_test = torch.FloatTensor(y_test)

      return X_train, X_test, y_train, y_test


  # ============================================================================
  # Define Neural Network with Dropout Support
  # ============================================================================

  class SimpleNeuralNetwork(nn.Module):
      """
      Simple feedforward neural network with one hidden layer.

      Now includes dropout regularization support.
      """

      def __init__(self, input_dim=1, hidden_dim=20, output_dim=1,
                  activation='relu', dropout_rate=0.0):
          """
          Initialize the neural network.

          Parameters:
          -----------
          input_dim : int
              Number of input features
          hidden_dim : int
              Number of neurons in the hidden layer
          output_dim : int
              Number of output values
          activation : str
              Activation function ('relu', 'tanh', 'sigmoid')
          dropout_rate : float
              Dropout probability (0.0 = no dropout, 0.5 = 50% dropout)
          """
          super(SimpleNeuralNetwork, self).__init__()

          # Layer 1: Input -> Hidden
          self.hidden = nn.Linear(input_dim, hidden_dim)

          # Activation function
          if activation == 'relu':
              self.activation = nn.ReLU()
          elif activation == 'tanh':
              self.activation = nn.Tanh()
          elif activation == 'sigmoid':
              self.activation = nn.Sigmoid()
          else:
              raise ValueError(f"Unknown activation: {activation}")

          # Dropout layer (only active during training)
          self.dropout = nn.Dropout(p=dropout_rate)

          # Layer 2: Hidden -> Output
          self.output = nn.Linear(hidden_dim, output_dim)

          self.activation_name = activation
          self.dropout_rate = dropout_rate

      def forward(self, x):
          """
          Forward pass through the network.

          Parameters:
          -----------
          x : torch.Tensor
              Input tensor

          Returns:
          --------
          torch.Tensor
              Output predictions
          """
          # Input -> Hidden layer
          hidden_output = self.hidden(x)

          # Apply activation function
          hidden_activated = self.activation(hidden_output)

          # Apply dropout (only active in training mode)
          hidden_dropout = self.dropout(hidden_activated)

          # Hidden -> Output layer
          output = self.output(hidden_dropout)

          return output


  # ============================================================================
  # Training Function with Weight Decay Support
  # ============================================================================

  def train_model(model, X_train, y_train, X_test, y_test,
                  n_epochs=500, learning_rate=0.01, batch_size=32, weight_decay=0.0):
      """
      Train the neural network model.

      Parameters:
      -----------
      model : nn.Module
          The neural network to train
      X_train, y_train : torch.Tensor
          Training data
      X_test, y_test : torch.Tensor
          Test data
      n_epochs : int
          Number of training epochs
      learning_rate : float
          Learning rate for optimizer
      batch_size : int
          Batch size for mini-batch gradient descent
      weight_decay : float
          L2 regularization strength (0.0 = no regularization)

      Returns:
      --------
      dict
          Training history
      """
      # Loss function
      criterion = nn.MSELoss()

      # Optimizer with weight decay (L2 regularization)
      optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

      # Create data loader
      train_dataset = TensorDataset(X_train, y_train)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

      # Track training history
      history = {
          'train_loss': [],
          'test_loss': [],
          'epoch': []
      }

      # Training loop
      for epoch in range(n_epochs):
          # Training mode (enables dropout)
          model.train()

          epoch_loss = 0.0
          n_batches = 0

          for batch_X, batch_y in train_loader:
              # Zero gradients
              optimizer.zero_grad()

              # Forward pass
              predictions = model(batch_X)

              # Compute loss
              loss = criterion(predictions, batch_y)

              # Backward pass
              loss.backward()

              # Update weights
              optimizer.step()

              epoch_loss += loss.item()
              n_batches += 1

          avg_train_loss = epoch_loss / n_batches

          # Evaluation mode (disables dropout)
          model.eval()
          with torch.no_grad():
              test_predictions = model(X_test)
              test_loss = criterion(test_predictions, y_test).item()

          # Record history
          history['epoch'].append(epoch)
          history['train_loss'].append(avg_train_loss)
          history['test_loss'].append(test_loss)

      return history


  # ============================================================================
  # Interactive Plotting Function with Regularization Controls
  # ============================================================================

  def create_interactive_plot(X_train, y_train, X_test, y_test):
      """
      Create an interactive plot with controls for all hyperparameters including regularization.
      """

      # Storage for the current model and history
      state = {
          'model': None,
          'history': None,
          'is_training': False
      }

      # ========================================================================
      # Create Widgets
      # ========================================================================

      # Model Architecture
      hidden_size_slider = widgets.IntSlider(
          value=20,
          min=1,
          max=100,
          step=1,
          description='Hidden Neurons:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px')
      )

      activation_dropdown = widgets.Dropdown(
          options=['relu', 'tanh', 'sigmoid'],
          value='relu',
          description='Activation:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='300px')
      )

      # Training Parameters
      epochs_slider = widgets.IntSlider(
          value=500,
          min=100,
          max=2000,
          step=100,
          description='Epochs:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px')
      )

      learning_rate_slider = widgets.FloatLogSlider(
          value=0.01,
          base=10,
          min=-4,
          max=-1,
          step=0.1,
          description='Learning Rate:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px'),
          readout_format='.4f'
      )

      batch_size_slider = widgets.IntSlider(
          value=32,
          min=8,
          max=128,
          step=8,
          description='Batch Size:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px')
      )

      # Regularization Parameters
      dropout_slider = widgets.FloatSlider(
          value=0.0,
          min=0.0,
          max=0.7,
          step=0.05,
          description='Dropout Rate:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px'),
          readout_format='.2f'
      )

      weight_decay_slider = widgets.FloatLogSlider(
          value=0.0,
          base=10,
          min=-6,
          max=-1,
          step=0.5,
          description='Weight Decay:',
          style={'description_width': '140px'},
          layout=widgets.Layout(width='500px'),
          readout_format='.6f'
      )

      # Add checkbox to enable/disable weight decay
      weight_decay_checkbox = widgets.Checkbox(
          value=False,
          description='Enable Weight Decay',
          style={'description_width': 'initial'},
          layout=widgets.Layout(width='200px')
      )

      # Train button and status
      train_button = widgets.Button(
          description='Train Model',
          button_style='success',
          icon='play',
          layout=widgets.Layout(width='200px', height='40px')
      )

      status_label = widgets.HTML(
          value='<b>Status:</b> Ready to train',
          layout=widgets.Layout(width='500px')
      )

      output = widgets.Output()

      # ========================================================================
      # Widget Interaction
      # ========================================================================

      def on_weight_decay_checkbox_changed(change):
          """Enable/disable weight decay slider."""
          weight_decay_slider.disabled = not change['new']

      weight_decay_checkbox.observe(on_weight_decay_checkbox_changed, names='value')
      weight_decay_slider.disabled = True  # Start disabled

      # ========================================================================
      # Update Plot Function
      # ========================================================================

      def update_plot():
          """Update the visualization."""
          with output:
              clear_output(wait=True)

              if state['model'] is None or state['history'] is None:
                  # Show initial data plot only
                  fig, ax = plt.subplots(1, 1, figsize=(14, 6))

                  ax.scatter(X_train.numpy(), y_train.numpy(), alpha=0.6, s=40,
                            label='Training data', color='steelblue',
                            edgecolors='black', linewidth=0.5)
                  ax.scatter(X_test.numpy(), y_test.numpy(), alpha=0.6, s=40,
                            label='Test data', color='coral',
                            edgecolors='black', linewidth=0.5)
                  ax.set_xlabel('x (input)', fontsize=12)
                  ax.set_ylabel('y (output)', fontsize=12)
                  ax.set_title('Gaussian Bumps Dataset - Click "Train Model" to begin',
                            fontsize=14, fontweight='bold')
                  ax.legend(fontsize=11)
                  ax.grid(True, alpha=0.3)

                  plt.tight_layout()
                  plt.show()
                  return

              # Create side-by-side plots
              fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

              # LEFT PLOT: Model predictions
              X_plot = torch.FloatTensor(np.linspace(-5, 5, 500).reshape(-1, 1))

              state['model'].eval()
              with torch.no_grad():
                  y_plot = state['model'](X_plot).numpy()

              X_plot_np = X_plot.numpy()

              ax1.scatter(X_train.numpy(), y_train.numpy(), alpha=0.6, s=40,
                        label='Training data', color='steelblue',
                        edgecolors='black', linewidth=0.5, zorder=2)
              ax1.scatter(X_test.numpy(), y_test.numpy(), alpha=0.6, s=40,
                        label='Test data', color='coral',
                        edgecolors='black', linewidth=0.5, zorder=2)
              ax1.plot(X_plot_np, y_plot, 'r-', linewidth=3,
                      label='NN Prediction', zorder=3)

              ax1.set_xlabel('x (input)', fontsize=12)
              ax1.set_ylabel('y (output)', fontsize=12)

              # Get final test loss for title
              final_test_loss = state['history']['test_loss'][-1]
              hidden_dim = hidden_size_slider.value

              ax1.set_title(f'Model Predictions (Hidden Neurons: {hidden_dim}, Test Loss: {final_test_loss:.6f})',
                          fontsize=13, fontweight='bold')
              ax1.legend(fontsize=10, loc='upper left')
              ax1.grid(True, alpha=0.3)

              # RIGHT PLOT: Training history
              epochs = state['history']['epoch']
              train_losses = state['history']['train_loss']
              test_losses = state['history']['test_loss']

              ax2.plot(epochs, train_losses, label='Training Loss',
                      linewidth=2, alpha=0.8, color='steelblue')
              ax2.plot(epochs, test_losses, label='Test Loss',
                      linewidth=2, alpha=0.8, color='coral')

              # Calculate gap between train and test loss
              final_train_loss = train_losses[-1]
              final_test_loss = test_losses[-1]
              gap = final_test_loss - final_train_loss

              # Add gap annotation
              gap_color = 'green' if gap < 0.01 else ('orange' if gap < 0.05 else 'red')
              ax2.text(0.98, 0.97, f'Train-Test Gap: {gap:.6f}',
                      transform=ax2.transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3),
                      fontsize=10, fontweight='bold')

              ax2.set_xlabel('Epoch', fontsize=12)
              ax2.set_ylabel('Loss (MSE)', fontsize=12)
              ax2.set_title('Training History', fontsize=13, fontweight='bold')
              ax2.legend(fontsize=10)
              ax2.grid(True, alpha=0.3)
              ax2.set_yscale('log')

              plt.tight_layout()
              plt.show()

              # Show detailed metrics
              print(f"\n{'='*70}")
              print(f"Model Configuration:")
              print(f"  Hidden Layer Size: {hidden_dim} neurons")
              print(f"  Activation: {state['model'].activation_name}")
              print(f"  Dropout Rate: {state['model'].dropout_rate:.2f}")
              print(f"  Total Parameters: {sum(p.numel() for p in state['model'].parameters())}")
              print(f"\nRegularization:")
              if state['model'].dropout_rate > 0:
                  print(f"  ✓ Dropout: {state['model'].dropout_rate:.2%} of neurons dropped during training")
              else:
                  print(f"  ✗ No dropout")

              wd = weight_decay_slider.value if weight_decay_checkbox.value else 0.0
              if wd > 0:
                  print(f"  ✓ Weight Decay (L2): {wd:.6f}")
              else:
                  print(f"  ✗ No weight decay")

              print(f"\nFinal Performance:")
              print(f"  Training Loss: {final_train_loss:.6f}")
              print(f"  Test Loss: {final_test_loss:.6f}")
              print(f"  Train-Test Gap: {gap:.6f}", end='')

              if gap < 0.01:
                  print(" ✓ (Good generalization)")
              elif gap < 0.05:
                  print(" ⚠ (Mild overfitting)")
              else:
                  print(" ✗ (Significant overfitting - try more regularization!)")

              # Calculate R² score
              from sklearn.metrics import r2_score
              with torch.no_grad():
                  y_pred = state['model'](X_test).numpy()
              r2 = r2_score(y_test.numpy(), y_pred)
              print(f"  Test R² Score: {r2:.4f}")
              print(f"{'='*70}")

      # ========================================================================
      # Train Button Handler
      # ========================================================================

      def on_train_button_clicked(b):
          """Handle train button click."""
          if state['is_training']:
              return

          state['is_training'] = True
          train_button.disabled = True
          status_label.value = '<b>Status:</b> <span style="color: orange;">Training...</span>'

          # Get hyperparameters
          hidden_dim = hidden_size_slider.value
          n_epochs = epochs_slider.value
          lr = learning_rate_slider.value
          batch_size = batch_size_slider.value
          activation = activation_dropdown.value
          dropout_rate = dropout_slider.value

          # Get weight decay (only if enabled)
          wd = weight_decay_slider.value if weight_decay_checkbox.value else 0.0

          # Create model with dropout
          model = SimpleNeuralNetwork(
              input_dim=1,
              hidden_dim=hidden_dim,
              output_dim=1,
              activation=activation,
              dropout_rate=dropout_rate
          )

          # Train with weight decay
          history = train_model(
              model, X_train, y_train, X_test, y_test,
              n_epochs=n_epochs,
              learning_rate=lr,
              batch_size=batch_size,
              weight_decay=wd
          )

          # Update state
          state['model'] = model
          state['history'] = history

          # Update display
          update_plot()

          status_label.value = '<b>Status:</b> <span style="color: green;">Training complete!</span>'
          train_button.disabled = False
          state['is_training'] = False

      # Connect button to handler
      train_button.on_click(on_train_button_clicked)

      # ========================================================================
      # Create Layout
      # ========================================================================

      architecture_section = widgets.VBox([
          widgets.HTML('<h4 style="margin-bottom: 10px;">🏗️ Model Architecture</h4>'),
          hidden_size_slider,
          activation_dropdown,
      ])

      training_section = widgets.VBox([
          widgets.HTML('<h4 style="margin-bottom: 10px; margin-top: 20px;">⚙️ Training Parameters</h4>'),
          epochs_slider,
          learning_rate_slider,
          batch_size_slider,
      ])

      regularization_section = widgets.VBox([
          widgets.HTML('<h4 style="margin-bottom: 10px; margin-top: 20px;">🛡️ Regularization (Prevent Overfitting)</h4>'),
          dropout_slider,
          widgets.HBox([weight_decay_checkbox, weight_decay_slider]),
          widgets.HTML('''
              <p style="font-size: 11px; color: #666; margin-top: 10px; line-height: 1.5;">
              <b>Dropout:</b> Randomly drops neurons during training to prevent co-adaptation<br>
              <b>Weight Decay:</b> Penalizes large weights (L2 regularization) to encourage simpler models
              </p>
          ''')
      ])

      controls = widgets.VBox([
          widgets.HTML('<h3>Neural Network Configuration</h3>'),
          architecture_section,
          training_section,
          regularization_section,
          widgets.HTML('<br>'),
          widgets.HBox([train_button, status_label]),
          widgets.HTML('''
              <p style="font-style: italic; margin-top: 15px; color: #555;">
              💡 <b>Tip:</b> Try training with 80 hidden neurons and no regularization to see overfitting,
              then add dropout (0.3) or weight decay (0.001) to see how it improves generalization!
              </p>
          ''')
      ])

      # Display initial plot
      update_plot()

      # Return the interactive interface
      return widgets.VBox([controls, output])


  # ============================================================================
  # Main Execution
  # ============================================================================

  # Prepare data
  print("Preparing data...")
  X_train, X_test, y_train, y_test = prepare_data(
      n_samples=40,
      noise_level=0.2,
      test_split=0.2
  )

  print(f"Training samples: {len(X_train)}")
  print(f"Test samples: {len(X_test)}")
  print("\n" + "="*70)
  print("Interactive Neural Network Trainer with Regularization")
  print("="*70)
  print("\nFeatures:")
  print("  • Adjustable hidden layer size (1-100 neurons)")
  print("  • Dropout regularization (0-70%)")
  print("  • Weight decay / L2 regularization")
  print("  • Real-time visualization of predictions and training curves")
  print("  • Overfitting detection (train-test gap)")
  print("\nReady! Adjust parameters and click 'Train Model'.\n")

  # Create and display interactive plot
  interactive_widget = create_interactive_plot(X_train, y_train, X_test, y_test)
  display(interactive_widget)
