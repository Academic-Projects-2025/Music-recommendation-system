from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.music_recommender.utils.logger import get_logger

logger = get_logger()


class ModelVisualizer:
    """Analyze and visualize model comparison results"""

    def __init__(self, results_path: Path, save_dir: Path):
        """
        Initialize ModelAnalyzer
        
        Args:
            results_path: Path to the CSV file containing model results
            save_dir: Directory to save outputs (figures and summaries)
        """
        self.results_df = pd.read_csv(results_path)
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_models_df = None
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (16, 12)

    def get_best_models(self):
        """
        Find the best model for each task group based on primary metrics.
        For regression: use r2 (higher is better)
        For classification: use f1_weighted (higher is better)
        
        Returns:
            pd.DataFrame: DataFrame containing best models for each task
        """
        best_models = []

        # Group by task_type and group
        for (task_type, group), group_df in self.results_df.groupby(["task_type", "group"]):
            if task_type == "regression":
                # For regression, maximize R²
                best_idx = group_df["r2"].idxmax()
                metric_col = "r2"
                metric_value = group_df.loc[best_idx, "r2"]
            else:  # classification
                # For classification, maximize F1-weighted
                best_idx = group_df["f1_weighted"].idxmax()
                metric_col = "f1_weighted"
                metric_value = group_df.loc[best_idx, "f1_weighted"]

            best_model = group_df.loc[best_idx]
            best_models.append(
                {
                    "task_type": task_type,
                    "group": group,
                    "best_model": best_model["model"],
                    "targets": best_model["targets"],
                    "primary_metric": metric_col,
                    "primary_score": metric_value,
                    "best_params": best_model["best_params"],
                }
            )

        self.best_models_df = pd.DataFrame(best_models)
        return self.best_models_df

    def print_best_models(self):
        """Print best models summary to console"""
        if self.best_models_df is None:
            self.get_best_models()
        
        print("=" * 80)
        print("BEST MODELS FOR EACH TASK:")
        print("=" * 80)
        print(self.best_models_df.to_string(index=False))
        print("\n")

    def plot_comprehensive_comparison(self):
        """Create comprehensive 6-subplot comparison"""
        fig = plt.figure(figsize=(18, 14))

        # --- SUBPLOT 1: Regression Tasks Comparison ---
        ax1 = plt.subplot(3, 2, 1)
        reg_df = self.results_df[self.results_df["task_type"] == "regression"].copy()
        reg_pivot = reg_df.pivot_table(index="model", columns="group", values="r2")
        reg_pivot.plot(kind="bar", ax=ax1, colormap="viridis", width=0.8)
        ax1.set_title(
            "Regression Models: R² Score by Task Group", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("R² Score", fontsize=12)
        ax1.set_xlabel("Model", fontsize=12)
        ax1.legend(title="Task Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(axis="y", alpha=0.3)
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- SUBPLOT 2: Regression MAE Comparison ---
        ax2 = plt.subplot(3, 2, 2)
        mae_pivot = reg_df.pivot_table(index="model", columns="group", values="neg_mae")
        mae_pivot = -mae_pivot  # Convert to positive MAE
        mae_pivot.plot(kind="bar", ax=ax2, colormap="plasma", width=0.8)
        ax2.set_title("Regression Models: MAE by Task Group", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Mean Absolute Error", fontsize=12)
        ax2.set_xlabel("Model", fontsize=12)
        ax2.legend(title="Task Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(axis="y", alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- SUBPLOT 3: Classification Accuracy ---
        ax3 = plt.subplot(3, 2, 3)
        cls_df = self.results_df[self.results_df["task_type"] == "classification"].copy()
        acc_pivot = cls_df.pivot_table(index="model", columns="group", values="accuracy")
        acc_pivot.plot(kind="bar", ax=ax3, colormap="coolwarm", width=0.8)
        ax3.set_title("Classification Models: Accuracy by Task", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Accuracy", fontsize=12)
        ax3.set_xlabel("Model", fontsize=12)
        ax3.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(axis="y", alpha=0.3)
        ax3.set_ylim([0, 1])
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- SUBPLOT 4: Classification F1-Weighted ---
        ax4 = plt.subplot(3, 2, 4)
        f1_pivot = cls_df.pivot_table(index="model", columns="group", values="f1_weighted")
        f1_pivot.plot(kind="bar", ax=ax4, colormap="RdYlGn", width=0.8)
        ax4.set_title(
            "Classification Models: F1-Weighted by Task", fontsize=14, fontweight="bold"
        )
        ax4.set_ylabel("F1-Weighted Score", fontsize=12)
        ax4.set_xlabel("Model", fontsize=12)
        ax4.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(axis="y", alpha=0.3)
        ax4.set_ylim([0, 1])
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- SUBPLOT 5: Best Model Summary (Regression) ---
        ax5 = plt.subplot(3, 2, 5)
        if self.best_models_df is None:
            self.get_best_models()
        
        best_reg = self.best_models_df[self.best_models_df["task_type"] == "regression"]
        x_pos = np.arange(len(best_reg))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_reg)))
        bars = ax5.barh(x_pos, best_reg["primary_score"], color=colors)
        ax5.set_yticks(x_pos)
        ax5.set_yticklabels(best_reg["group"])
        ax5.set_xlabel("R² Score", fontsize=12)
        ax5.set_title("Best Regression Models per Task Group", fontsize=14, fontweight="bold")
        ax5.grid(axis="x", alpha=0.3)

        # Add value labels and model names
        for i, (bar, model) in enumerate(zip(bars, best_reg["best_model"])):
            width = bar.get_width()
            ax5.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{model}: {width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # --- SUBPLOT 6: Best Model Summary (Classification) ---
        ax6 = plt.subplot(3, 2, 6)
        best_cls = self.best_models_df[self.best_models_df["task_type"] == "classification"]
        x_pos = np.arange(len(best_cls))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(best_cls)))
        bars = ax6.barh(x_pos, best_cls["primary_score"], color=colors)
        ax6.set_yticks(x_pos)
        ax6.set_yticklabels(best_cls["group"])
        ax6.set_xlabel("F1-Weighted Score", fontsize=12)
        ax6.set_title("Best Classification Models per Task", fontsize=14, fontweight="bold")
        ax6.grid(axis="x", alpha=0.3)
        ax6.set_xlim([0, 1])

        # Add value labels and model names
        for i, (bar, model) in enumerate(zip(bars, best_cls["best_model"])):
            width = bar.get_width()
            ax6.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{model}: {width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        save_path = self.save_dir / "MFCC_model_comparison_comprehensive.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_heatmaps(self):
        """Create heatmaps for regression and classification"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Regression heatmap
        ax_reg = axes[0]
        reg_df = self.results_df[self.results_df["task_type"] == "regression"].copy()
        reg_heatmap_data = reg_df.pivot_table(index="group", columns="model", values="r2")
        sns.heatmap(
            reg_heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax_reg,
            cbar_kws={"label": "R² Score"},
            vmin=0,
            vmax=0.5,
        )
        ax_reg.set_title("Regression: R² Scores Heatmap", fontsize=14, fontweight="bold")
        ax_reg.set_xlabel("Model", fontsize=12)
        ax_reg.set_ylabel("Task Group", fontsize=12)

        # Classification heatmap
        ax_cls = axes[1]
        cls_df = self.results_df[self.results_df["task_type"] == "classification"].copy()
        cls_heatmap_data = cls_df.pivot_table(
            index="group", columns="model", values="f1_weighted"
        )
        sns.heatmap(
            cls_heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax_cls,
            cbar_kws={"label": "F1-Weighted"},
            vmin=0,
            vmax=1,
        )
        ax_cls.set_title("Classification: F1-Weighted Heatmap", fontsize=14, fontweight="bold")
        ax_cls.set_xlabel("Model", fontsize=12)
        ax_cls.set_ylabel("Task", fontsize=12)

        plt.tight_layout()
        save_path = self.save_dir / "MFCC_model_comparison_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def export_best_models(self):
        """Export best models summary to CSV"""
        if self.best_models_df is None:
            self.get_best_models()
        
        save_path = self.save_dir / "MFCC_best_models_summary.csv"
        self.best_models_df.to_csv(save_path, index=False)
        print(f"Best models summary saved: {save_path}")

    def generate_all(self):
        """Generate all analyses and visualizations"""
        
        # Get and print best models
        self.get_best_models()
        self.print_best_models()
        
        # Generate visualizations
        self.plot_comprehensive_comparison()
        self.plot_heatmaps()
        
        # Export results
        self.export_best_models()
    
        print("Visualizations saved as:")
        print("  - MFCC_model_comparison_comprehensive.png")
        print("  - MFCC_model_comparison_heatmaps.png")
        print("Best models summary saved as: MFCC_best_models_summary.csv")

        logger.success("All visualizations generated successfully!")