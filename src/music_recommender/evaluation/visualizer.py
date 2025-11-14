from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.music_recommender.utils.logger import get_logger

logger = get_logger()


class ModelVisualizer:
    """Visualize model comparison results"""

    def __init__(self, results_df: pd.DataFrame, save_dir: Path):
        self.results_df = results_df
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (16, 12)

    def plot_comprehensive_comparison(self):
        """Create comprehensive 6-subplot comparison"""
        logger.info("Creating comprehensive comparison plots...")

        fig = plt.figure(figsize=(18, 14))

        reg_df = self.results_df[self.results_df["task_type"] == "regression"].copy()
        cls_df = self.results_df[
            self.results_df["task_type"] == "classification"
        ].copy()

        # Subplot 1: Regression R² Scores
        ax1 = plt.subplot(3, 2, 1)
        self._plot_regression_r2(ax1, reg_df)

        # Subplot 2: Regression MAE
        ax2 = plt.subplot(3, 2, 2)
        self._plot_regression_mae(ax2, reg_df)

        # Subplot 3: Classification Accuracy
        ax3 = plt.subplot(3, 2, 3)
        self._plot_classification_accuracy(ax3, cls_df)

        # Subplot 4: Classification F1-Weighted
        ax4 = plt.subplot(3, 2, 4)
        self._plot_classification_f1(ax4, cls_df)

        # Subplot 5: Best Regression Models
        ax5 = plt.subplot(3, 2, 5)
        self._plot_best_regression(ax5, reg_df)

        # Subplot 6: Best Classification Models
        ax6 = plt.subplot(3, 2, 6)
        self._plot_best_classification(ax6, cls_df)

        plt.tight_layout()
        save_path = self.save_dir / "model_comparison_comprehensive.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.success(f"Saved: {save_path}")

    def plot_heatmaps(self):
        """Create heatmaps for regression and classification"""
        logger.info("Creating heatmap visualizations...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        reg_df = self.results_df[self.results_df["task_type"] == "regression"].copy()
        cls_df = self.results_df[
            self.results_df["task_type"] == "classification"
        ].copy()

        # Regression heatmap
        reg_heatmap_data = reg_df.pivot_table(
            index="group", columns="model", values="r2"
        )
        sns.heatmap(
            reg_heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=axes[0],
            cbar_kws={"label": "R² Score"},
            vmin=0,
            vmax=0.5,
        )
        axes[0].set_title(
            "Regression: R² Scores Heatmap", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel("Model", fontsize=12)
        axes[0].set_ylabel("Task Group", fontsize=12)

        # Classification heatmap
        cls_heatmap_data = cls_df.pivot_table(
            index="group", columns="model", values="f1_weighted"
        )
        sns.heatmap(
            cls_heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=axes[1],
            cbar_kws={"label": "F1-Weighted"},
            vmin=0,
            vmax=1,
        )
        axes[1].set_title(
            "Classification: F1-Weighted Heatmap", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel("Model", fontsize=12)
        axes[1].set_ylabel("Task", fontsize=12)

        plt.tight_layout()
        save_path = self.save_dir / "model_comparison_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.success(f"Saved: {save_path}")

    def _plot_regression_r2(self, ax, reg_df):
        """Plot regression R² scores"""
        reg_pivot = reg_df.pivot_table(index="model", columns="group", values="r2")
        reg_pivot.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
        ax.set_title(
            "Regression Models: R² Score by Task Group", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.legend(title="Task Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_regression_mae(self, ax, reg_df):
        """Plot regression MAE"""
        mae_pivot = reg_df.pivot_table(index="model", columns="group", values="neg_mae")
        mae_pivot = -mae_pivot  # Convert to positive MAE
        mae_pivot.plot(kind="bar", ax=ax, colormap="plasma", width=0.8)
        ax.set_title(
            "Regression Models: MAE by Task Group", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Mean Absolute Error", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.legend(title="Task Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_classification_accuracy(self, ax, cls_df):
        """Plot classification accuracy"""
        acc_pivot = cls_df.pivot_table(
            index="model", columns="group", values="accuracy"
        )
        acc_pivot.plot(kind="bar", ax=ax, colormap="coolwarm", width=0.8)
        ax.set_title(
            "Classification Models: Accuracy by Task", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_classification_f1(self, ax, cls_df):
        """Plot classification F1-weighted"""
        f1_pivot = cls_df.pivot_table(
            index="model", columns="group", values="f1_weighted"
        )
        f1_pivot.plot(kind="bar", ax=ax, colormap="RdYlGn", width=0.8)
        ax.set_title(
            "Classification Models: F1-Weighted by Task", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("F1-Weighted Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_best_regression(self, ax, reg_df):
        """Plot best regression models summary"""
        from src.music_recommender.evaluation.evaluator import get_best_models

        best_reg = get_best_models(self.results_df)
        best_reg = best_reg[best_reg["task_type"] == "regression"]

        x_pos = np.arange(len(best_reg))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_reg)))
        bars = ax.barh(x_pos, best_reg["primary_score"], color=colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(best_reg["group"])
        ax.set_xlabel("R² Score", fontsize=12)
        ax.set_title(
            "Best Regression Models per Task Group", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

        for i, (bar, model) in enumerate(zip(bars, best_reg["best_model"])):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{model}: {width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    def _plot_best_classification(self, ax, cls_df):
        """Plot best classification models summary"""
        from src.music_recommender.evaluation.evaluator import get_best_models

        best_cls = get_best_models(self.results_df)
        best_cls = best_cls[best_cls["task_type"] == "classification"]

        x_pos = np.arange(len(best_cls))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(best_cls)))
        bars = ax.barh(x_pos, best_cls["primary_score"], color=colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(best_cls["group"])
        ax.set_xlabel("F1-Weighted Score", fontsize=12)
        ax.set_title(
            "Best Classification Models per Task", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim([0, 1])

        for i, (bar, model) in enumerate(zip(bars, best_cls["best_model"])):
            width = bar.get_width()
            ax.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{model}: {width:.3f}",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    def generate_all(self):
        """Generate all visualizations"""
        self.plot_comprehensive_comparison()
        self.plot_heatmaps()
        logger.success("All visualizations generated successfully!")
