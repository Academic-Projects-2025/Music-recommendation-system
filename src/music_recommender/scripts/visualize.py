import pandas as pd

from src.music_recommender.config import Config
from src.music_recommender.evaluation.visualizer import ModelVisualizer
from src.music_recommender.utils.logger import get_logger

logger = get_logger(context="visualization")
cfg = Config()

logger.info("Loading results...")
results_df = pd.read_csv(cfg.paths.models / "MFCC_model_comparison_results.csv")

logger.info("Generating visualizations...")
visualizer = ModelVisualizer(results_df, cfg.paths.reports / "figures")
visualizer.generate_all()

logger.success("Done!")
import pandas as pd

from src.music_recommender.config import Config
from src.music_recommender.evaluation.visualizer import ModelVisualizer
from src.music_recommender.utils.logger import get_logger

logger = get_logger(context="visualization")
cfg = Config()

logger.info("Loading results...")
results_df = pd.read_csv(cfg.paths.models / "model_comparison_results.csv")

logger.info("Generating visualizations...")
visualizer = ModelVisualizer(results_df, cfg.paths.reports / "figures")
visualizer.generate_all()

logger.success("Done!")
