import numpy as np

GRID_OUT    = 450
CELL_SIZE   = GRID_OUT // 9
CELL_MARGIN = 4
CELL_INNER  = CELL_SIZE - 2 * CELL_MARGIN

DST_PTS = np.array([
    [0,            0           ],
    [GRID_OUT - 1, 0           ],
    [GRID_OUT - 1, GRID_OUT - 1],
    [0,            GRID_OUT - 1],
], dtype=np.float32)

# Reintento de resolución con candidatos alternativos de la CNN
RETRY_CONFIDENCE_THRESHOLD = 0.5
RETRY_MAX_SUSPECTS         = 5
RETRY_MAX_DEPTH            = 2
