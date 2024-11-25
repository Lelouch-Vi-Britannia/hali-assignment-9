import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from sklearn.linear_model import LogisticRegression

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights and biases with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.activations = None
        self.gradients = None

    def forward(self, X, store_activations=True):
        # Forward pass
        z1 = X.dot(self.W1) + self.b1
        if self.activation_fn == 'tanh':
            a1 = np.tanh(z1)
        elif self.activation_fn == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation_fn == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))
        else:
            raise ValueError("Unsupported activation function")

        z2 = a1.dot(self.W2) + self.b2
        a2 = 1 / (1 + np.exp(-z2))

        if store_activations:
            # Store activations for visualization
            self.z1 = z1
            self.a1 = a1
            self.z2 = z2
            self.a2 = a2
            self.activations = {
                'a1': self.a1,
                'a2': self.a2
            }

        out = a2
        if store_activations:
            return out
        else:
            return out, a1  # Return a1 when not storing activations

    def backward(self, X, y):
        m = y.shape[0]
        # Compute the derivative of the loss with respect to the output
        delta2 = self.a2 - y  # Shape: (n_samples, output_dim)

        # Compute gradients with respect to W2 and b2
        dW2 = self.a1.T.dot(delta2) / m  # Shape: (hidden_dim, output_dim)
        db2 = np.sum(delta2, axis=0, keepdims=True) / m  # Shape: (1, output_dim)

        # Compute the derivative of the activation function at z1
        if self.activation_fn == 'tanh':
            activation_derivative = 1 - self.a1 ** 2
        elif self.activation_fn == 'relu':
            activation_derivative = (self.z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            activation_derivative = self.a1 * (1 - self.a1)
        else:
            raise ValueError("Unsupported activation function")

        # Compute delta for the first layer
        delta1 = delta2.dot(self.W2.T) * activation_derivative  # Shape: (n_samples, hidden_dim)

        # Compute gradients with respect to W1 and b1
        dW1 = X.T.dot(delta1) / m  # Shape: (input_dim, hidden_dim)
        db1 = np.sum(delta1, axis=0, keepdims=True) / m  # Shape: (1, hidden_dim)

        # Update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }

    def get_gradients(self):
        # Return a copy of gradients
        return {
            'W1': self.gradients['W1'].copy(),
            'b1': self.gradients['b1'].copy(),
            'W2': self.gradients['W2'].copy(),
            'b2': self.gradients['b2'].copy()
        }

def generate_data(n_samples=200):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Define grid size for plotting
grid_size = 20

# Function to plot hidden space (modified to match the first code)
def plot_hidden_space(ax, hidden_features, y, step, xlim, ylim, zlim, grid_hidden=None):
    ax.clear()
    ax.set_title(f"Hidden Space at Step {step}")
    ax.set_xlabel('h1')
    ax.set_ylabel('h2')
    ax.set_zlabel('h3')

    # Set fixed axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Fix the view angle
    ax.view_init(elev=30, azim=45)

    # Plot the transformed grid as a surface
    if grid_hidden is not None:
        H1 = grid_hidden[:, 0].reshape((grid_size, grid_size))
        H2 = grid_hidden[:, 1].reshape((grid_size, grid_size))
        H3 = grid_hidden[:, 2].reshape((grid_size, grid_size))
        ax.plot_surface(H1, H2, H3, color='lightblue', alpha=0.3, linewidth=0)

    # Scatter plot
    ax.scatter(hidden_features[y.ravel() == 0, 0],
               hidden_features[y.ravel() == 0, 1],
               hidden_features[y.ravel() == 0, 2],
               color='blue', label='Class 0', alpha=0.6)
    ax.scatter(hidden_features[y.ravel() == 1, 0],
               hidden_features[y.ravel() == 1, 1],
               hidden_features[y.ravel() == 1, 2],
               color='red', label='Class 1', alpha=0.6)

    # Fit a plane (hyperplane) to separate the classes in hidden space
    clf = LogisticRegression()
    clf.fit(hidden_features, y.ravel())

    # Create grid to plot the hyperplane
    h1_range = np.linspace(xlim[0], xlim[1], grid_size)
    h2_range = np.linspace(ylim[0], ylim[1], grid_size)
    H1_plane, H2_plane = np.meshgrid(h1_range, h2_range)

    # Compute H3 based on the plane equation
    w = clf.coef_[0]
    b = clf.intercept_[0]
    # Avoid division by zero
    if np.abs(w[2]) < 1e-4:
        return  # Skip plotting the plane if w[2] is too small
    H3_plane = (-w[0]*H1_plane - w[1]*H2_plane - b) / w[2]

    # Plot the plane
    ax.plot_surface(H1_plane, H2_plane, H3_plane, color='orange', alpha=0.5)

    ax.legend()

# Function to plot gradient graph (modified to match the first code)
def plot_gradient_graph(ax, gradients):
    ax.clear()
    ax.set_title("Gradients at Step")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.4)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.41, 0.2))
    ax.grid(True)

    # Define node positions
    node_positions = {
        'x1': (0.2, 1.2),
        'x2': (0.2, 0.4),
        'h1': (0.5, 1.2),
        'h2': (0.5, 0.8),
        'h3': (0.5, 0.4),
        'y': (0.8, 0.8)
    }
    nodes = ['x1', 'x2', 'h1', 'h2', 'h3', 'y']

    # Plot nodes
    for node in nodes:
        x, y_pos = node_positions[node]
        ax.scatter(x, y_pos, s=300, color='blue', edgecolors='k', zorder=3)
        ax.text(x, y_pos + 0.05, node, horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder=4)

    # Define edges with corresponding gradients
    edges = [
        ('x1', 'h1'),
        ('x1', 'h2'),
        ('x1', 'h3'),
        ('x2', 'h1'),
        ('x2', 'h2'),
        ('x2', 'h3'),
        ('h1', 'y'),
        ('h2', 'y'),
        ('h3', 'y')
    ]

    # Collect all gradient magnitudes
    grad_magnitudes = []
    edge_grad_map = {}
    for edge in edges:
        src, dst = edge
        if src.startswith('x') and dst.startswith('h'):
            # Weights from input to hidden
            idx = int(src[1:]) - 1  # 'x1'->0, 'x2'->1
            hid = {'h1': 0, 'h2': 1, 'h3': 2}[dst]
            grad = np.abs(gradients['W1'][idx, hid])
        elif src.startswith('h') and dst == 'y':
            # Weights from hidden to output
            hid = {'h1': 0, 'h2': 1, 'h3': 2}[src]
            grad = np.abs(gradients['W2'][hid, 0])
        else:
            grad = 0  # For any other connections that don't exist
        grad_magnitudes.append(grad)
        edge_grad_map[edge] = grad

    # Normalize gradient magnitudes for thickness
    max_grad = max(grad_magnitudes) if grad_magnitudes else 1
    min_grad = min(grad_magnitudes) if grad_magnitudes else 0
    for edge in edges:
        grad = edge_grad_map[edge]
        # Normalize between 1 and 5
        thickness = 1 + 4 * (grad - min_grad) / (max_grad - min_grad + 1e-8)
        src, dst = edge
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]
        ax.plot([x1, x2], [y1, y2], linewidth=thickness, color='purple')

    ax.set_aspect('equal')

# Function to plot input space decision boundary (modified)
def plot_input_space(ax, mlp, X, y, step):
    ax.clear()
    ax.set_title(f"Input Space at Step {step}")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _ = mlp.forward(grid, store_activations=False)
    probs = probs.reshape(xx.shape)

    # Plot the decision boundary
    contour = ax.contourf(xx, yy, probs, levels=50, cmap='bwr', alpha=0.6)
    # plt.colorbar(contour, ax=ax)

    # Contour line for probability=0.5
    ax.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1)

    # Scatter plot of the data points
    ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1],
               color='blue', label='Class 0', edgecolors='k', alpha=0.6)
    ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1],
               color='red', label='Class 1', edgecolors='k', alpha=0.6)

    ax.legend()

# Visualization update function (corrected)
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, loss_history, step_num, xlim, ylim, zlim, step_per_frame=10):
    current_step = frame * step_per_frame
    for _ in range(step_per_frame):
        train_outputs = mlp.forward(X)
        mlp.backward(X, y)
        current_step += 1
        if current_step >= step_num:
            break

    # Compute loss
    loss = -np.mean(y * np.log(train_outputs + 1e-8) + (1 - y) * np.log(1 - train_outputs + 1e-8))
    loss_history.append(loss)

    # Get hidden features before any forward pass for plotting
    hidden_features = mlp.activations['a1'].copy()

    # Get gradients
    gradients = mlp.get_gradients()

    # Plot input space decision boundary (without storing activations)
    plot_input_space(ax_input, mlp, X, y, current_step)

    # Generate grid in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_grid, yy_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                                   np.linspace(y_min, y_max, grid_size))
    grid_input = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # Map grid points to hidden space without storing activations
    _, grid_hidden = mlp.forward(grid_input, store_activations=False)

    # Plot hidden space with transformed grid and decision hyperplane
    plot_hidden_space(ax_hidden, hidden_features, y, current_step, xlim, ylim, zlim, grid_hidden=grid_hidden)

    # Plot gradient graph
    plot_gradient_graph(ax_gradient, gradients)

    print(f"Completed step {current_step}")

    # Stop the animation if current_step exceeds step_num
    if current_step >= step_num:
        plt.close()

    fig = ax_input.get_figure()
    fig.suptitle(f'Step {current_step}, Loss: {loss:.4f}', fontsize=16)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Manually set axis limits for hidden space
    xlim = (-1.2, 1.2)
    ylim = (-1.2, 1.2)
    zlim = (-1.2, 1.2)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    loss_history = []

    # Total frames
    step_per_frame = 10
    total_frames = (step_num + step_per_frame - 1) // step_per_frame  # Ensure covering all steps

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y, loss_history=loss_history, step_num=step_num,
                                     xlim=xlim, ylim=ylim, zlim=zlim, step_per_frame=step_per_frame),
                        frames=total_frames, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=5)
    plt.close()



if __name__ == "__main__":
    activation = "relu"
    lr = 0.1  # Adjusted learning rate
    step_num = 1000
    visualize(activation, lr, step_num)