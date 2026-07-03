import cv2
import gzip
import io
import numpy as np
import os
import struct
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import CELL_SIZE, CELL_MARGIN, CELL_INNER
from grid_detection import ROTATIONS, rotate_image


_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

_MODEL_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cnn_model.pt"
)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess_cell(gray_rect: np.ndarray, row: int, col: int) -> np.ndarray:
    y0 = row * CELL_SIZE + CELL_MARGIN
    x0 = col * CELL_SIZE + CELL_MARGIN
    y1 = y0 + CELL_INNER
    x1 = x0 + CELL_INNER

    cell   = gray_rect[y0:y1, x0:x1]
    cell   = cv2.GaussianBlur(cell, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        cell, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 5
    )
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def has_digit(thresh: np.ndarray) -> bool:
    h, w = thresh.shape
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    border = 5
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        cx, cy = centroids[i]
        if border < cx < w - border and border < cy < h - border:
            return True
    return False


def _normalize_digit(thresh: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 4
        x1 = max(0, x - pad);       y1 = max(0, y - pad)
        x2 = min(thresh.shape[1], x + w + pad)
        y2 = min(thresh.shape[0], y + h + pad)
        crop = thresh[y1:y2, x1:x2]
        size   = max(crop.shape[0], crop.shape[1])
        square = np.zeros((size, size), dtype=np.uint8)
        off_y  = (size - crop.shape[0]) // 2
        off_x  = (size - crop.shape[1]) // 2
        square[off_y:off_y + crop.shape[0], off_x:off_x + crop.shape[1]] = crop
        resized = cv2.resize(square, (28, 28))
    else:
        resized = np.zeros((28, 28), dtype=np.uint8)
    return resized


def _mnist_download(filename: str) -> bytes:
    for base in _MNIST_MIRRORS:
        try:
            with urllib.request.urlopen(base + filename, timeout=20) as r:
                return r.read()
        except Exception:
            continue
    raise RuntimeError(f"No se pudo descargar {filename}.")


def _mnist_parse_images(raw: bytes) -> np.ndarray:
    with gzip.open(io.BytesIO(raw)) as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return (data > 128).astype(np.uint8) * 255


def _mnist_parse_labels(raw: bytes) -> np.ndarray:
    with gzip.open(io.BytesIO(raw)) as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def _load_mnist_images(n_train: int = 4000, n_val: int = 400, n_test: int = 400):
    print("  Cargando MNIST...")
    imgs   = _mnist_parse_images(_mnist_download("train-images-idx3-ubyte.gz"))
    labels = _mnist_parse_labels(_mnist_download("train-labels-idx1-ubyte.gz"))

    morph_kernel = np.ones((2, 2), np.uint8)

    def _build(start, end):
        sel_imgs, sel_lbls = [], []
        for digit in range(1, 10):
            idx = np.where(labels == digit)[0][start:end]
            for i in idx:
                img28 = imgs[i].reshape(28, 28)
                img28 = cv2.morphologyEx(img28, cv2.MORPH_OPEN, morph_kernel)
                sel_imgs.append(img28)
            sel_lbls.extend([digit] * len(idx))
        return np.array(sel_imgs, dtype=np.float32), np.array(sel_lbls, dtype=np.int64)

    train_imgs, train_lbls = _build(0, n_train)
    val_imgs,   val_lbls   = _build(n_train, n_train + n_val)
    test_imgs,  test_lbls  = _build(n_train + n_val, n_train + n_val + n_test)

    print(f"  ✓ MNIST: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test.")
    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)


def _synthetic_training_images():
    images, labels = [], []
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    for digit in range(1, 10):
        for font in fonts:
            for scale in np.arange(0.5, 2.8, 0.15):
                for thickness in [1, 2]:
                    img  = np.zeros((CELL_INNER, CELL_INNER), dtype=np.uint8)
                    text = str(digit)
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    px = max(0, (CELL_INNER - tw) // 2)
                    py = min(CELL_INNER - 1, (CELL_INNER + th) // 2)
                    cv2.putText(img, text, (px, py), font, scale, 255, thickness)
                    if np.count_nonzero(img) > 20:
                        img28 = _normalize_digit(img)
                        images.append(img28)
                        labels.append(digit)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def _augment(images: np.ndarray, labels: np.ndarray, max_total: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    aug_imgs, aug_lbls = list(images), list(labels)
    n_to_add = max_total - len(images)
    if n_to_add <= 0:
        return images, labels

    indices = np.random.choice(len(images), size=n_to_add, replace=True)
    for i in indices:
        img, lbl = images[i], labels[i]
        h, w = img.shape
        choice = np.random.randint(3)

        if choice == 0:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug_imgs.append(cv2.warpAffine(img, M, (w, h)))
        elif choice == 1:
            dx, dy = np.random.randint(-3, 4), np.random.randint(-3, 4)
            M_t = np.float32([[1, 0, dx], [0, 1, dy]])
            aug_imgs.append(cv2.warpAffine(img, M_t, (w, h)))
        else:
            k = np.ones((2, 2), np.uint8)
            if np.random.random() > 0.5:
                aug_imgs.append(cv2.dilate(img, k, iterations=1))
            else:
                aug_imgs.append(cv2.erode(img, k, iterations=1))
        aug_lbls.append(lbl)

    return np.array(aug_imgs, dtype=np.float32), np.array(aug_lbls, dtype=np.int64)


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images / 255.0).float().unsqueeze(1)


def train_classifier():
    if os.path.exists(_MODEL_CACHE):
        print("Cargando modelo CNN desde cache...")
        model = DigitCNN()
        model.load_state_dict(torch.load(_MODEL_CACHE, weights_only=True))
        model.eval()
        print("  ✓ Modelo CNN cargado.\n")
        return model

    print("Entrenando clasificador CNN...")

    try:
        (tr_imgs, tr_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls) = _load_mnist_images()
    except Exception as e:
        print(f"  ⚠ MNIST no disponible ({e}). Usando solo datos sintéticos.")
        tr_imgs = np.empty((0, 28, 28), np.float32)
        tr_lbls = np.empty((0,), np.int64)
        val_imgs, val_lbls = tr_imgs, tr_lbls
        test_imgs, test_lbls = tr_imgs, tr_lbls

    syn_imgs, syn_lbls = _synthetic_training_images()
    print(f"  ✓ Sintéticos: {len(syn_imgs)} muestras.")

    all_imgs = np.concatenate([tr_imgs, syn_imgs]) if len(tr_imgs) else syn_imgs
    all_lbls = np.concatenate([tr_lbls, syn_lbls]) if len(tr_lbls) else syn_lbls

    all_imgs, all_lbls = _augment(all_imgs, all_lbls)
    print(f"  ✓ Con augmentation: {len(all_imgs)} muestras totales.")

    # Labels 1-9 → indices 0-8 para CrossEntropyLoss
    train_x = _to_tensor(all_imgs)
    train_y = torch.from_numpy(all_lbls - 1).long()

    dataset = TensorDataset(train_x, train_y)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(15):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/15 — loss: {total_loss / len(loader):.4f}")

    model.eval()

    if len(val_imgs) > 0:
        val_acc  = _accuracy(model, val_imgs, val_lbls)
        test_acc = _accuracy(model, test_imgs, test_lbls)
        print(f"  ✓ Accuracy validación ({len(val_imgs)} muestras): {val_acc:.2%}")
        print(f"  ✓ Accuracy test       ({len(test_imgs)} muestras): {test_acc:.2%}")

    torch.save(model.state_dict(), _MODEL_CACHE)
    print(f"  ✓ Modelo guardado en {_MODEL_CACHE}\n")
    return model


def _accuracy(model: DigitCNN, images: np.ndarray, labels: np.ndarray) -> float:
    x = _to_tensor(images)
    with torch.no_grad():
        preds = model(x).argmax(dim=1).numpy() + 1
    return float(np.mean(preds == labels))


def read_grid(rectified: np.ndarray, model: DigitCNN):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    grid = []
    candidates = [[None]*9 for _ in range(9)]
    total_confidence = 0.0
    n_digits = 0
    for row in range(9):
        row_vals = []
        for col in range(9):
            thresh = preprocess_cell(gray, row, col)
            if has_digit(thresh):
                img28 = _normalize_digit(thresh)
                x = _to_tensor(img28[np.newaxis])[0:1]
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    sorted_idx = probs[0].argsort(descending=True)
                    candidates[row][col] = [(idx.item() + 1, probs[0][idx].item()) for idx in sorted_idx[:3]]
                    confidence = probs.max().item()
                    digit = probs.argmax().item() + 1
                total_confidence += confidence
                n_digits += 1
                row_vals.append(digit)
            else:
                row_vals.append(0)
        grid.append(row_vals)
    avg_confidence = total_confidence / n_digits if n_digits > 0 else 0.0
    grid = _fix_violations(grid, candidates)
    return grid, avg_confidence, candidates


def _cell_has_violation(grid, row, col) -> bool:
    val = grid[row][col]
    if val == 0:
        return False
    if grid[row].count(val) > 1:
        return True
    if sum(1 for r in range(9) if grid[r][col] == val) > 1:
        return True
    br, bc = 3 * (row // 3), 3 * (col // 3)
    box = [grid[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)]
    if box.count(val) > 1:
        return True
    return False


def _fix_violations(grid, candidates):
    for row in range(9):
        for col in range(9):
            if candidates[row][col] is None:
                continue
            if not _cell_has_violation(grid, row, col):
                continue
            for alt_digit, _ in candidates[row][col][1:]:
                grid[row][col] = alt_digit
                if not _cell_has_violation(grid, row, col):
                    break
            else:
                grid[row][col] = candidates[row][col][0][0]
    return grid


def _grid_violations(grid: list[list[int]]) -> int:
    violations = 0
    for row in grid:
        vals = [v for v in row if v != 0]
        violations += len(vals) - len(set(vals))
    for col in range(9):
        vals = [grid[r][col] for r in range(9) if grid[r][col] != 0]
        violations += len(vals) - len(set(vals))
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = [grid[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3) if grid[r][c] != 0]
            violations += len(vals) - len(set(vals))
    return violations


def read_grid_auto_orient(rectified: np.ndarray, model: DigitCNN):
    best = None
    for rot in ROTATIONS:
        img  = rotate_image(rectified, rot)
        grid, confidence, candidates = read_grid(img, model)
        # Menos violaciones primero, luego mayor confianza (negamos para que menor = mejor)
        score = (_grid_violations(grid), -confidence)
        if best is None or score < best[0]:
            best = (score, grid, rot, img, candidates)
    _, grid, rot, img, candidates = best
    return grid, rot, img, candidates
