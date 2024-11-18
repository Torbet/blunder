import chess
import numpy as np
import chess.pgn
import os
import shutil
from helpers.evaluate import evaluate_stockfish, evaluate_lazy

# parameters
limit = 100
years = [2023, 2022, 2021]
modes = ["HvH", "HvC", "CvC"]
num_moves = 40
elo_range = (1800, 2200)
label_mappings = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
num_best_moves = 5


def parse_files(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    moves = np.zeros((4 * limit, num_moves, 6, 8, 8))
    evals = np.zeros((4 * limit, num_moves))
    times = np.zeros((4 * limit, num_moves))
    best_moves = np.zeros((4 * limit, num_moves))
    labels = np.zeros((4 * limit))

    count = 0
    counts = {label: 0 for label in range(4)}

    for path in paths:
        print(path)
        with open(path) as file:
            while (game := chess.pgn.read_game(file)) is not None:
                # metadata
                headers = game.headers
                white_comp = int(headers.get("WhiteIsComp", "No") == "Yes")
                black_comp = int(headers.get("BlackIsComp", "No") == "Yes")
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
                ply_count = int(headers.get("PlyCount", 0))

                label = label_mappings[(white_comp, black_comp)]

                # early stopping
                if "HvH" in path:
                    if counts[0] >= limit:
                        break
                    if label != 0:
                        continue
                if "HvC" in path:
                    if counts[1] >= limit and counts[2] >= limit:
                        break
                    if label not in [1, 2] or counts[label] >= limit:
                        continue
                if "CvC" in path:
                    if counts[3] >= limit:
                        break
                    if label != 3:
                        continue

                # filter
                if white_comp == 0:
                    if white_elo < elo_range[0] or white_elo > elo_range[1]:
                        continue
                if black_comp == 0:
                    if black_elo < elo_range[0] or black_elo > elo_range[1]:
                        continue
                if ply_count < num_moves + 20:
                    continue

                # parse
                moves[count], evals[count], times[count], best_moves[count] = (
                    parse_game(game)
                )
                labels[count] = label

                # iterate
                count += 1
                counts[label] += 1
                print(count, counts)

    return moves, evals, times, best_moves, labels


def parse_game(game: chess.pgn.Game) -> np.ndarray:
    moves = np.zeros((num_moves, 6, 8, 8))
    evals = np.zeros(num_moves)
    times = np.zeros(num_moves)
    best_moves = np.zeros(num_moves)

    bms = []
    board = game.board()
    for i, node in enumerate(game.mainline()):
        # stop after num_moves moves
        if i == num_moves + 10:
            break
        board.push(node.move)
        # skip first 10 moves
        if i >= 10:
            moves[i - 10] = parse_board(board)
            best_moves[i - 10] = 1 if node.move.uci() in bms else 0
            evals[i - 10], bms = evaluate_stockfish(board, num_best_moves)
            times[i - 10] = node.emt()

    return moves, evals, times, best_moves


def parse_board(board: chess.Board) -> np.ndarray:
    # 6 channels for each piece type
    output = np.zeros((6, 8, 8))
    for i in range(64):
        if (piece := board.piece_at(i)) is not None:
            piece_type = piece.piece_type
            piece_color = piece.color
            output[piece_type - 1][i // 8][i % 8] = 1 if piece_color else -1
    return output


if __name__ == "__main__":
    paths = [f"data/raw/{year}_{mode}.pgn" for year in years for mode in modes]
    output_dir = f"data/processed/{limit}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    moves, evals, times, best_moves, labels = parse_files(paths)
    for name, data in zip(
        ["moves", "evals", "times", "best_moves", "labels"],
        [moves, evals, times, best_moves, labels],
    ):
        np.save(f"{output_dir}/{name}.npy", data)

    print(f"Saved to {output_dir}")
