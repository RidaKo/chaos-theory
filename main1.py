import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def curve(x):
    return x**2 - 3*x + 1

def distance_to_curve(x, x0, y0):
    return np.sqrt((x - x0)**2 + (curve(x) - y0)**2)

def closest_point_on_curve(x0, y0):
    result = minimize(distance_to_curve, x0, args=(x0, y0))
    closest_x = result.x[0]
    closest_y = curve(closest_x)
    return closest_x, closest_y


def find_symmetric_point(x0, y0, x_closest, y_closest):
    dx = x0 - x_closest
    dy = y0 - y_closest
    
    x_symmetric = x_closest - dx
    y_symmetric = y_closest - dy
    
    return x_symmetric, y_symmetric

points = []


def plot_all_points():
    x_vals = np.linspace(-2, 5, 400)
    y_vals = curve(x_vals)

    plt.clf()
    plt.plot(x_vals, y_vals, label="y = x^2 - 3x + 1", color="blue")  # Curve
    
    for (x0, y0), (x_closest, y_closest), (x_symmetric, y_symmetric) in points:
        plt.scatter([x0], [y0], color="red", label="Point A" if points.index(((x0, y0), (x_closest, y_closest) ,(x_symmetric, y_symmetric))) == 0 else "")
        plt.scatter([x_symmetric], [y_symmetric], color="orange", label="Symmetric Point A'" if points.index(((x0, y0), (x_closest, y_closest), (x_symmetric, y_symmetric))) == 0 else "")
        
        plt.scatter([x_closest], [y_closest], color="green", label="Closest Point on Curve" if points.index(((x0, y0), (x_closest, y_closest), (x_symmetric, y_symmetric))) == 0 else "")
        plt.plot([x0, x_closest], [y0, y_closest], color="gray", linestyle="--")
        plt.plot([x_closest, x_symmetric], [y_closest, y_symmetric], color="gray", linestyle="--")
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Symmetric Points Across the Curve")
    plt.grid(True)
    plt.draw()

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x0, y0 = event.xdata, event.ydata
        print(f"Selected point: ({x0}, {y0})")

        x_closest, y_closest = closest_point_on_curve(x0, y0)
        x_symmetric, y_symmetric = find_symmetric_point(x0, y0, x_closest, y_closest)

        points.append(((x0, y0), (x_closest, y_closest), (x_symmetric, y_symmetric)))

        plot_all_points()



# Main
def interactive_plot():
    fig, ax = plt.subplots()
    x_vals = np.linspace(-2, 5, 400)
    y_vals = curve(x_vals)
    ax.plot(x_vals, y_vals, label="y = x^2 - 3x + 1", color="blue")  
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Click on the plot to choose a point")
    ax.grid(True)
    plt.legend()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

interactive_plot()
