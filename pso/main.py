import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

# Optimization functions

def schwefel_function(position):
    x, y = position
    return 418.9829 * 2 - (x * math.sin(math.sqrt(abs(x))) + y * math.sin(math.sqrt(abs(y))))

def banana_function(position):
    x, y = position
    return (1 - x)**2 + 100 * (y - x**2)**2

# PSO algorithm parts
max_iter = 500  # the total number of iterations

class Particle:
    def __init__(self, bounds, v_max=None):
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([random.uniform(-1, 1) for _ in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        # Define the maximum velocity if not given, it could be a fraction of the bound range for each dimension
        self.v_max = v_max if v_max is not None else np.array([(b[1] - b[0]) / 2 for b in bounds])

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = random.random(), random.random()
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        # Apply velocity clamping
        for i, v in enumerate(self.velocity):
            if abs(v) > self.v_max[i]:
                self.velocity[i] = math.copysign(self.v_max[i], v)

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.maximum(self.position, [b[0] for b in bounds])
        self.position = np.minimum(self.position, [b[1] for b in bounds])

class Swarm:
    def __init__(self, num_particles, bounds, function, v_max=None):
        self.particles = [Particle(bounds, v_max) for _ in range(num_particles)]
        self.global_best_position = np.zeros(len(bounds))
        self.global_best_value = float('inf')
        self.function = function

    def update(self, i, max_iter, w_start, w_end, c1, c2):
        w = w_start - ((w_start - w_end) * (i / max_iter))  # Decreasing inertia weight
        for particle in self.particles:
            value = self.function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(particle.position)
            particle.update_velocity(self.global_best_position, w, c1, c2)
            particle.update_position(bounds)

# Contour plot for the optimization function

def plot_function(function, bounds):
    x = np.linspace(bounds[0][0], bounds[0][1], 400)
    y = np.linspace(bounds[1][0], bounds[1][1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[function([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])
    return X, Y, Z

# Animation function to update both subplots
def animate(i):
    w_start = 0.9  # starting inertia weight
    w_end = 0.4    # ending inertia weight
    swarm_schwefel.update(i, max_iter, w_start, w_end, c1, c2)
    swarm_banana.update(i, max_iter, w_start, w_end, c1, c2)
    
    ax1.clear()
    ax1.contourf(X_schwefel, Y_schwefel, Z_schwefel, levels=50, cmap='viridis')
    ax1.scatter([p.position[0] for p in swarm_schwefel.particles], 
                [p.position[1] for p in swarm_schwefel.particles], color='r')
    ax1.scatter(swarm_schwefel.global_best_position[0], 
                swarm_schwefel.global_best_position[1], color='g', marker='*', s=100)
    ax1.set_title('Schwefel Function')
    ax1.set_xlim(bounds[0])
    ax1.set_ylim(bounds[1])
    
    ax2.clear()
    ax2.contourf(X_banana, Y_banana, Z_banana, levels=50, cmap='viridis')
    ax2.scatter([p.position[0] for p in swarm_banana.particles], 
                [p.position[1] for p in swarm_banana.particles], color='r')
    ax2.scatter(swarm_banana.global_best_position[0], 
                swarm_banana.global_best_position[1], color='g', marker='*', s=100)
    ax2.set_title('Banana (Rosenbrock) Function')
    ax2.set_xlim(bounds[0])
    ax2.set_ylim(bounds[1])

# PSO parameters

num_particles = 500
bounds = [(-500, 500), (-500, 500)]
w = 0.5  # inertia
c1 = 0.8  # cognitive parameter
c2 = 0.9  # social parameter

# Create two swarms for the two functions
schwefel_v_max = [200, 200]  #Values for velocity clamping
banana_v_max = [100, 100]  #Values for velocity clamping
swarm_schwefel = Swarm(num_particles, bounds, schwefel_function, schwefel_v_max)
swarm_banana = Swarm(num_particles, bounds, banana_function, banana_v_max)

# Plot both functions
X_schwefel, Y_schwefel, Z_schwefel = plot_function(schwefel_function, bounds)
X_banana, Y_banana, Z_banana = plot_function(banana_function, bounds)

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Create the animation

animation = FuncAnimation(fig, animate, frames=max_iter, interval=100)

# Save the animation as a video file

animation.save('pso_simulation.mp4', writer='ffmpeg', fps=10)

plt.show()
