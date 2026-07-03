import itertools

from config import RETRY_CONFIDENCE_THRESHOLD, RETRY_MAX_SUSPECTS, RETRY_MAX_DEPTH


def _is_valid(grid: list[list[int]], row: int, col: int, num: int) -> bool:
    if num in grid[row]:
        return False
    if any(grid[r][col] == num for r in range(9)):
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] == num:
                return False
    return True


def _count_clues(grid: list[list[int]]) -> int:
    return sum(1 for row in grid for val in row if val != 0)


def solve(grid: list[list[int]], _max_steps: int = 500_000) -> bool:
    counter = [0]

    def _solve() -> bool:
        counter[0] += 1
        if counter[0] > _max_steps:
            return False
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    for num in range(1, 10):
                        if _is_valid(grid, row, col, num):
                            grid[row][col] = num
                            if _solve():
                                return True
                            grid[row][col] = 0
                    return False
        return True

    return _solve()


def solve_with_candidates(
    grid: list[list[int]],
    candidates: list[list[list[tuple[int, float]] | None]],
    confidence_threshold: float = RETRY_CONFIDENCE_THRESHOLD,
    max_suspects: int = RETRY_MAX_SUSPECTS,
    max_depth: int = RETRY_MAX_DEPTH,
) -> bool:
    if solve(grid):
        return True

    suspects = [
        (row, col)
        for row in range(9)
        for col in range(9)
        if candidates[row][col] is not None
        and len(candidates[row][col]) > 1
        and candidates[row][col][0][1] < confidence_threshold
    ]
    suspects.sort(key=lambda rc: candidates[rc[0]][rc[1]][0][1])
    suspects = suspects[:max_suspects]

    original_values = {(row, col): grid[row][col] for row, col in suspects}

    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(suspects, depth):
            alt_choices = [candidates[row][col][1:] for row, col in combo]
            for alts in itertools.product(*alt_choices):
                for (row, col), (digit, _) in zip(combo, alts):
                    grid[row][col] = digit
                if solve(grid):
                    return True
                for row, col in combo:
                    grid[row][col] = original_values[(row, col)]

    return False
