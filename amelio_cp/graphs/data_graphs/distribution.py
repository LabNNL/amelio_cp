import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Distributions:
    pass

    @staticmethod


    def plot_violin(X1, X2=None, show=True, highlight_idx=None, output_path=None):
        """
        Plot a violin plot of X1, optionally grouped by X2.
        Optionally highlight one specific point using highlight_idx.

        Parameters:
        - X1 : Series
        - X2 : Series or None
            If provided, makes grouped violins.
            If None, plots a single violin for X1.
        """
        
        if not isinstance(X1, pd.Series):
            raise TypeError("X1 must be a pandas Series.")
        
        if X2 is not None and not isinstance(X2, pd.Series):
            raise TypeError("X2 must be a pandas Series or None.")
        
        # --- Create figure ---
        sns.set_context('notebook', font_scale=1.2)
        fig, ax = plt.subplots()

        # --- Case 1: Only one series ---
        if X2 is None:
            sns.violinplot(
                y=X1,
                density_norm='count',
                common_norm=True,
                cut=1,
                inner="stick",
                palette="Set3",
                bw_method=0.5,
                ax=ax
            )

            sns.swarmplot(
                y=X1,
                color="gray",
                edgecolor="white",
                s=6,
                ax=ax
            )

            if highlight_idx is not None:
                if highlight_idx in X1.index:
                    ax.scatter(
                        x=0,                         # single violin is at x=0
                        y=X1.loc[highlight_idx],
                        color="red",
                        s=40,
                        edgecolor="white",
                        zorder=10,
                        marker="D"
                    )
                ax.set_title(f"Violin plot of {X1.name} for {highlight_idx} for {highlight_idx}")
            else: 
                ax.set_title(f"Violin plot of {X1.name}")
        
        # --- Case 2: Series vs grouping variable ---
        else:
            df = pd.DataFrame({X1.name: X1, X2.name: X2})

            sns.violinplot(
                x=X2.name,
                y=X1.name,
                data=df,
                density_norm='count',
                common_norm=True,
                cut=1,
                inner="stick",
                palette="Set3",
                bw_method=0.5,
                ax=ax
            )

            sns.swarmplot(
                x=X2.name,
                y=X1.name,
                data=df,
                color="gray",
                edgecolor="white",
                s=6,
                ax=ax
            )

            if highlight_idx is not None:
                if highlight_idx in df.index:
                    x_val = df.loc[highlight_idx, X2.name]
                    y_val = df.loc[highlight_idx, X1.name]

                    ax.scatter(
                        x=x_val,
                        y=y_val,
                        color="red",
                        s=40,
                        edgecolor="white",
                        zorder=10,
                        marker="D" 
                    )
                ax.set_title(f"Violin plot of {X1.name} by {X2.name} for {highlight_idx}")
            else:
                ax.set_title(f"Violin plot of {X1.name} by {X2.name}")

        # --- Visual touches ---
        ax.grid(axis='y')
        ax.set_axisbelow(True)

        if show:
            plt.show()

        if output_path:
            if X2 is None:
                plt.savefig(f"{output_path}violinplot_{X1.name}_idx{highlight_idx}.svg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{output_path}violinplot_{X1.name}_{X2.name}_idx{highlight_idx}.svg", dpi=300, bbox_inches="tight")
            print(f"SHAP plot saved to: {output_path}")

        return ax

    @staticmethod
    def scatter_plot_by_individual(x, y, cat=None, show=True):

        if not isinstance(x, pd.Series):
            raise TypeError("x must be a pandas Series.")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series or None.")
        
        fig, ax = plt.subplots(figsize=(15,6))
        sns.scatterplot(x=x, y=y, hue=cat, palette='viridis')
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)
        ax.set_xlabel('Subject Id')
        plt.grid(axis='x', alpha=0.5)

        if show:
            plt.show()