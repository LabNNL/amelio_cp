import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class LollipopPlot:
    def __init__(self):
        pass

    def _absolute_formatter(self, x, pos):
        """Formatter to display absolute values on the X-axis."""
        return f"{abs(x):.2f}"

    def plot_lollipop(self, x1, x2, y, labels, title: str, output_path: str = None, show=True):
        """
        Plots a lollipop chart comparing two sets of values.

        Parameters:
        - x1: First set of values (e.g., proba in class 0).
        - x2: Second set of values (e.g., proba in class 1).
        - y: Y-axis positions for the labels.
        - labels: Labels for each data point.
        - title: Title of the plot.
        - output_path: Path to save the plot (if None, the plot is not saved).
        - show: Whether to display the plot.
        """
        fig, ax = plt.subplots()

        yy = np.arange(len(y))

        ax.hlines(yy, x1, x2, color="gray", alpha=0.5, zorder=1)
        ax.scatter(x1, yy, color="skyblue", alpha=1, label=labels[0])
        ax.scatter(x2, yy, color="lightgreen", alpha=1, label=labels[1])
        ax.vlines(x=0, ymin=-1, ymax=max(yy), color="lightgrey", alpha=0.5, zorder=1)

        ax.set_xlim(-1, 1)

        # Apply the absolute value formatter to the X-axis
        formatter = FuncFormatter(self._absolute_formatter)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_yticks(yy, labels=y)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Samples")
        ax.set_title(title)
        ax.legend()

        ax.text(-0.85, max(yy), labels[0], horizontalalignment="center", verticalalignment="center")
        ax.text(0.85, max(yy), labels[1], horizontalalignment="center", verticalalignment="center")

        if output_path:
            plt.savefig(f"{output_path}lollipop_plot.svg", dpi=300, bbox_inches="tight")
            print(f"Lollipop plot saved to: {output_path}")

        if show:
            plt.show()

    @staticmethod
    def order_values(x1, x2, y):
        """
        Orders the values based on the y positions.

        Parameters:
        - x1: First set of values.
        - x2: Second set of values.
        - y: Y-axis positions.

        Returns:
        - Ordered x1, x2, and y based on ascending order of y.
        """
        idx = np.argsort(y, kind="stable")  # indices that sort y ascending
        x1_sorted = np.array(x1)[idx]
        x2_sorted = np.array(x2)[idx]
        y_sorted = np.array(y)[idx]
        print("Values ordered for lollipop plot.")
        return x1_sorted, x2_sorted, y_sorted
