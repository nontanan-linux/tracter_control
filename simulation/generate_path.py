import numpy as np
import math
import csv
import os

def generate_figure_eight(scale=15.0, steps=1000, velocity=2.22):
    traj = []
    # Figure 8 parameters
    a = scale
    
    for i in range(steps):
        t = (i / steps) * 2 * math.pi
        
        # Parametric Equations
        x = a * math.sin(t)
        y = a * math.sin(t) * math.cos(t)
        
        # First Derivatives (Velocity direction)
        dx = a * math.cos(t)
        dy = a * (math.cos(t)**2 - math.sin(t)**2)
        theta = math.atan2(dy, dx)
        
        # Second Derivatives (for Curvature)
        ddx = -a * math.sin(t)
        ddy = -4 * a * math.sin(t) * math.cos(t) # Simplified
        
        # Curvature: k = (dx*ddy - dy*ddx) / (dx^2 + dy^2)^(1.5)
        num = dx * ddy - dy * ddx
        den = (dx**2 + dy**2)**1.5
        kappa = num / den
        
        v = velocity
        
        traj.append([x, y, theta, v, kappa])
    return traj

def save_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'theta', 'v', 'kappa'])
        writer.writerows(data)
    print(f"Path saved to {filename}")

if __name__ == "__main__":
    path_data = generate_figure_eight()
    output_path = os.path.join(os.path.dirname(__file__), 'paths/figure8.csv')
    save_csv(output_path, path_data)
