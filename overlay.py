import cv2
import numpy as np

from config import GRID_OUT, CELL_SIZE


def draw_on_rectified(rectified: np.ndarray,
                      original:  list[list[int]],
                      solved:    list[list[int]]) -> np.ndarray:
    out  = rectified.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(9):
        for col in range(9):
            if original[row][col] == 0 and solved[row][col] != 0:
                digit = str(solved[row][col])
                (tw, th), _ = cv2.getTextSize(digit, font, 1.0, 2)
                px = col * CELL_SIZE + (CELL_SIZE - tw) // 2
                py = row * CELL_SIZE + (CELL_SIZE + th) // 2
                cv2.putText(out, digit, (px, py), font, 1.0, (220, 80, 0), 2)
    return out

#Por si esta doblado, vuelve a doblar la perspectiva de la solucion
def project_back(frame: np.ndarray,
                 solution_rect: np.ndarray,
                 H: np.ndarray) -> np.ndarray:
    h, w  = frame.shape[:2]
    H_inv = np.linalg.inv(H)

    warped_back = cv2.warpPerspective(solution_rect, H_inv, (w, h))

    mask_src = np.full((GRID_OUT, GRID_OUT), 255, dtype=np.uint8)
    mask     = cv2.warpPerspective(mask_src, H_inv, (w, h))
    mask_3ch = cv2.merge([mask, mask, mask])

    return np.where(mask_3ch > 0, warped_back, frame)


def put_text(img, text, pos=(10, 30), color=(255, 255, 255), scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def print_grid(grid: list[list[int]], label: str = ""):
    if label:
        print(f"\n── {label} ──")
    for i, row in enumerate(grid):
        line = " ".join(str(v) if v else "." for v in row)
        if i in (3, 6):
            print("──────┼───────┼──────")
        print(line[:5] + " │ " + line[6:11] + " │ " + line[12:])
    print()
