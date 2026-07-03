import copy
import cv2
import numpy as np

from digit_recognition import train_classifier, read_grid_auto_orient
from grid_detection    import detect_grid, rectify, rotate_image, inverse_rotation, rotation_name
from overlay           import draw_on_rectified, project_back, put_text, print_grid
from solver            import solve_with_candidates


def main():
    model = train_classifier()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    H             = None
    original_grid = None
    solved_grid   = None
    solution_img  = None
    solved        = False
    show_debug    = False

    intentos_totales = 0
    resueltos        = 0
    fallidos         = 0

    print("=== SUDOKU SOLVER ===")
    print("Apuntá la cámara al sudoku y presioná ESPACIO cuando la grilla esté detectada.")
    print("ESPACIO → resolver | R → reiniciar | D → debug | M → métricas | Q → salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        corners = detect_grid(frame)

        if solved and H is not None and solution_img is not None:
            display = project_back(display, solution_img, H)

        if corners is not None:
            cv2.polylines(display,
                          [corners.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 255, 0), 2)
            for pt in corners.astype(int):
                cv2.circle(display, tuple(pt), 6, (0, 255, 0), -1)
            put_text(display, "Grilla detectada — ESPACIO para resolver",
                     (10, 30), color=(0, 255, 0))
        else:
            put_text(display, "Buscando grilla del sudoku...",
                     (10, 30), color=(0, 150, 255))

        if solved:
            put_text(display, "RESUELTO", (10, 65), color=(0, 220, 0))

        put_text(display,
                 "ESPACIO=resolver  R=reset  D=debug  M=metricas  Q=salir",
                 (10, display.shape[0] - 12),
                 color=(255, 220, 0), scale=0.5, thickness=1)

        cv2.imshow("Sudoku Solver", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                for win in ("Rectificada", "Solución rectificada"):
                    try:
                        cv2.destroyWindow(win)
                    except Exception:
                        pass

        elif key == ord('r'):
            H, original_grid, solved_grid, solution_img = None, None, None, None
            solved = False
            for win in ("Rectificada", "Solución rectificada"):
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    pass
            print("→ Reiniciado.\n")

        elif key == ord('m'):
            pct = (resueltos / intentos_totales * 100) if intentos_totales > 0 else 0
            print(f"\n=== MÉTRICAS ===")
            print(f"  Intentos totales:        {intentos_totales}")
            print(f"  Resueltos exitosamente:  {resueltos}")
            print(f"  Fallidos (no resolvible): {fallidos}")
            print(f"  Tasa de éxito:           {pct:.1f}%\n")

        elif key == ord(' ') and not solved:
            if corners is None:
                print("⚠ No se detectó ninguna grilla. Reintentá con mejor iluminación o ángulo.")
                continue

            print("Procesando frame...")

            rectified, H = rectify(frame, corners)
            if show_debug:
                cv2.imshow("Rectificada", rectified)

            print("Leyendo dígitos...")
            original_grid, rot, rectified_oriented, candidates = read_grid_auto_orient(rectified, model)
            if rot is not None:
                print(f"  → Sudoku detectado rotado {rotation_name(rot)}, corrigiendo orientación.")
            print_grid(original_grid, "Grilla leída (. = vacío)")

            intentos_totales += 1
            solved_grid = copy.deepcopy(original_grid)
            print("Resolviendo sudoku...")
            if solve_with_candidates(solved_grid, candidates):
                resueltos += 1
                print_grid(solved_grid, "Solución")
                solution_oriented = draw_on_rectified(rectified_oriented, original_grid, solved_grid)
                solution_img = rotate_image(solution_oriented, inverse_rotation(rot))
                if show_debug:
                    cv2.imshow("Solución rectificada", solution_img)
                solved = True
                print("¡Sudoku resuelto! Presioná R para reiniciar con otro.\n")
            else:
                fallidos += 1
                print("✗ No se pudo resolver el sudoku.")
                print("  Posibles causas: dígitos mal reconocidos, puzzle inválido.")
                print("  Intentá presionar ESPACIO de nuevo o R para reiniciar.\n")

    cap.release()
    cv2.destroyAllWindows()

    if intentos_totales > 0:
        pct = resueltos / intentos_totales * 100
        print(f"\n=== MÉTRICAS FINALES ===")
        print(f"  Intentos totales:        {intentos_totales}")
        print(f"  Resueltos exitosamente:  {resueltos}")
        print(f"  Fallidos (no resolvible): {fallidos}")
        print(f"  Tasa de éxito:           {pct:.1f}%\n")


if __name__ == '__main__':
    main()
