"""Project configuration and paths."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "dellavigna_pope_2018")
OUTPUT_DIR = os.path.join(ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
IMAGES_DIR = os.path.join(ROOT, "presentation", "unifi_latex_overleaf", "images")

DATA_SHORT = os.path.join(DATA_DIR, "mturk_clean_data_short.dta")
DATA_FULL = os.path.join(DATA_DIR, "original_data", "MTurkCleanedData.dta")
DATA_EXPERTS = os.path.join(DATA_DIR, "original_data", "ExpertForecastCleanWide.dta")

# Treatment labels used throughout
TREATMENT_ORDER = [
    "1.1", "1.2", "1.3", "1.4", "2",
    "3.1", "3.2",
    "4.1", "4.2",
    "5.1", "5.2", "5.3",
    "6.1", "6.2",
    "7", "8", "9", "10",
]

TREATMENT_NAMES = {
    "1.1": "1c Piece Rate",
    "1.2": "10c Piece Rate",
    "1.3": "No Payment",
    "1.4": "4c Piece Rate",
    "2": "Very Low Pay",
    "3.1": "1c Red Cross",
    "3.2": "10c Red Cross",
    "4.1": "1c 2 Weeks",
    "4.2": "1c 4 Weeks",
    "5.1": "Gain 40c",
    "5.2": "Loss 40c",
    "5.3": "Gain 80c",
    "6.1": "Prob .01 $1",
    "6.2": "Prob .5 2c",
    "7": "Social Comp.",
    "8": "Ranking",
    "9": "Task Signif.",
    "10": "Gift Exch.",
}
