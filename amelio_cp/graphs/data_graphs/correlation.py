import pandas as pd
import seaborn as sns
from datetime import datetime
import os
import matplotlib.pyplot as plt

class Correlations:
    def __init__(self):
        pass

    @staticmethod
    def plot_correlation_matrix(data, title, output_folder, show=True, correlation_method='pearson'):
        l, L = data.shape

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corr_matrix_{title}_{correlation_method}_{timestamp}.svg"
        output_path = os.path.join(output_folder, filename)

        corr_matrix = data.corr(method=correlation_method)

        plt.figure(figsize=(20, 18))
        sns.set(font_scale=2) 
        sns.heatmap(
            corr_matrix,
            annot=True,
            annot_kws={"size": L//1.5},
            center=0,
            fmt='.1f',
            linewidths=0.5,
            cmap='coolwarm'
        )

        plt.title('Correlation Matrix of ' + title, fontsize=L//16)
        plt.tight_layout()

        if show:
            plt.show()
        #TODO: savefig doesn't work, need to be fixed
        plt.savefig(output_path, bbox_inches='tight')

    @staticmethod
    def plot_correlation(X1, X2):
        
        sns.regplot(x=X1, y=X2, )
        plt.title("Correlation between x and y")
        plt.show()

