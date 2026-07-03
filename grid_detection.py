import cv2
import numpy as np

from config import DST_PTS, GRID_OUT

#Ordena los puntos del cuadrado que se muestre
def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_grid(frame: np.ndarray) -> np.ndarray | None:
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blur, 30, 120)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w     = frame.shape[:2]
    min_area = h * w * 0.05

    for cnt in contours[:10]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2).astype(np.float32)
    return None

#Es lo que corrige la perspectiva
def rectify(frame: np.ndarray, corners: np.ndarray):
    ordered = order_points(corners)
    H, _    = cv2.findHomography(ordered, DST_PTS)
    warped  = cv2.warpPerspective(frame, H, (GRID_OUT, GRID_OUT))
    return warped, H


# Las 4 orientaciones posibles de la grilla rectificada (de costado / al revés)
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

_INVERSE_ROTATION = {
    None:                          None,
    cv2.ROTATE_90_CLOCKWISE:       cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_180:                cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE,
}

_ROTATION_NAMES = {
    None:                          "0°",
    cv2.ROTATE_90_CLOCKWISE:       "90° horario",
    cv2.ROTATE_180:                "180°",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "90° antihorario",
}


def rotate_image(img: np.ndarray, rotation) -> np.ndarray:
    """Aplica una rotación de ROTATIONS (None = sin cambios)."""
    return img if rotation is None else cv2.rotate(img, rotation)


def inverse_rotation(rotation):
    """Rotación que revierte `rotation`."""
    return _INVERSE_ROTATION[rotation]


def rotation_name(rotation) -> str:
    return _ROTATION_NAMES[rotation]
