import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os
import matplotlib.pyplot as plt


class Correlations:
    def __init__(self):
        pass

    @staticmethod
    def plot_correlation_matrix(data, title, output_folder=None, show=True, correlation_method="pearson"):
        l, L = data.shape

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corr_matrix_{title}_{correlation_method}_{timestamp}.svg"

        corr_matrix = data.corr(method=correlation_method)

        plt.figure(figsize=(20, 18))
        sns.set(font_scale=2)
        sns.heatmap(
            corr_matrix, annot=True, annot_kws={"size": L // 1.5}, center=0, fmt=".1f", linewidths=0.5, cmap="coolwarm"
        )

        plt.title("Correlation Matrix of " + title, fontsize=L // 16)
        plt.tight_layout()

        if show:
            plt.show()

        if output_folder is not None:
            output_path = os.path.join(output_folder, filename)
            # TODO: savefig doesn't work, need to be fixed
            plt.savefig(output_path, bbox_inches="tight")

        return corr_matrix

    @staticmethod
    def plot_correlation(X1, X2, show=True):
        X1_clean, X2_clean = Correlations._dropna_together(X1, X2)
        r = np.corrcoef(X1_clean, X2_clean)[0, 1]
        sns.regplot(x=X1_clean, y=X2_clean)
        plt.text(0.05, 0.95, f"r = {r:.3f}", transform=plt.gca().transAxes, fontsize=14, va="top")
        if X1.name and X2.name:
            x1_name = X1.name
            x2_name = X2.name
            plt.title(f"Correlation between {x1_name} and {x2_name}")
        else:
            plt.title("Correlation Plot")

        if show:
            plt.show()

        return r

    @staticmethod
    def _dropna_together(s1, s2):
        """
        Remove rows where either s1 or s2 contains NaN.
        Returns cleaned copies of both series, aligned.
        """
        # Combine into a temporary DataFrame
        df = pd.concat([s1, s2], axis=1)

        # Drop rows with ANY NaN
        df = df.dropna()

        # Return the cleaned series separately
        s1_clean = df.iloc[:, 0]
        s2_clean = df.iloc[:, 1]

        return s1_clean, s2_clean
