"""Shared, restrained publication palette and typography."""

COLORS = {
    "OCT4": "#1B4965",
    "MED1": "#C05A24",
    "DNA": "#4A5568",
    "mixed": "#007C83",
    "neutral": "#505050",
    "light": "#D0D0D0",
}
OCT4_REGIONS = ["#1B4965", "#367FA5"]
MED1_REGIONS = ["#8C360F", "#B34D18", "#CD6C2C", "#E29450"]
# DNA controls are unfilled; the treatment uses forest green.
CONDITIONS = {"without_DNA": "none", "with_DNA": "#2D6A4F"}
LAYERS = ["#2D6A4F", "#52B788", "#B7E4C7"]
DNA_SNAPSHOT_COLOR = CONDITIONS["with_DNA"]
# Figure 2D uses black/white markers and line styles only.
DNA_METRIC_COLOR = "#000000"
FONT = "Nimbus Sans"
AXIS_LABEL_SIZE = 10.0
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.4
PANEL_LABEL_SIZE = 11.0
D90_CONTOUR = "#222222"
