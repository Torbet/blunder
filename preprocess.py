import chess
import numpy as np
import chess.pgn
import os
import shutil
from helpers.evaluate import evaluate_stockfish, evaluate_lazy

# parameters
limit = 1000
years = [2023, 2022, 2021]
modes = ["HvH", "HvC", "CvC"]
num_moves = 40
elo_range = (1800, 2200)


def parse_files(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    moves = np.zeros((4 * limit, num_moves, 6, 8, 8))
    evals = np.zeros((4 * limit, num_moves))
    times = np.zeros((4 * limit, num_moves))
    labels = np.zeros((4 * limit, 2))

    count = 0
    counts = {(0, 0): 0, (0, 1): 0, (1, 1): 0, (1, 0): 0}

    for path in paths:
        print(path)
        with open(path) as file:
            while True:
                game = chess.pgn.read_game(file)
                if game is None:
                    break

                # metadata
                headers = game.headers
                white_comp = int(headers.get("WhiteIsComp", "No") == "Yes")
                black_comp = int(headers.get("BlackIsComp", "No") == "Yes")
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
                ply_count = int(headers.get("PlyCount", 0))

                """
                HvH: (0, 0)
                HvC: (0, 1), (1, 0)
                CvC: (1, 1)
                """
                label = (white_comp, black_comp)

                if "HvH" in path:
                    if label != (0, 0):
                        continue
                    if counts[(0, 0)] >= limit:
                        break
                if "HvC" in path:
                    if label not in [(0, 1), (1, 0)]:
                        continue
                    if counts[(0, 1)] >= limit and counts[(1, 0)] >= limit:
                        break
                if "CvC" in path:
                    if label != (1, 1):
                        continue
                    if counts[label] >= limit:
                        break

                if counts[label] >= limit:
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
                moves[count], evals[count], times[count] = parse_game(game)
                labels[count] = label

                # iterate
                count += 1
                counts[label] += 1
                print(count, counts)
                game = chess.pgn.read_game(file)

    return moves, evals, times, labels


def parse_game(game: chess.pgn.Game) -> np.ndarray:
    moves = np.zeros((num_moves, 6, 8, 8))
    evals = np.zeros(num_moves)
    times = np.zeros(num_moves)

    board = game.board()
    for i, node in enumerate(game.mainline()):
        # stop after num_moves moves
        if i == num_moves + 10:
            break
        board.push(node.move)
        # skip first 10 moves
        if i < 10:
            continue
        moves[i - 10] = parse_board(board)
        evals[i - 10] = evaluate_stockfish(board)
        times[i - 10] = node.emt()

    return moves, evals, times


def parse_board(board: chess.Board) -> np.ndarray:
    # 6 channels for each piece type
    output = np.zeros((6, 8, 8))
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            colour = int(piece.color)
            piece_type = piece.piece_type
            # output[piece_type - 1 + 6 * colour][i // 8][i % 8] = 1
            output[piece_type - 1][i // 8][i % 8] = 1 if colour == 0 else -1
    return output


if __name__ == "__main__":
    paths = [f"data/raw/{year}_{mode}.pgn" for year in years for mode in modes]
    output = f"data/processed/{limit}"
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    moves, evals, times, labels = parse_files(paths)
    print("evals", evals, "\n\n")
    print("times", times, "\n\n")
    np.save(f"{output}/moves.npy", moves)
    np.save(f"{output}/evals.npy", evals)
    np.save(f"{output}/times.npy", times)
    np.save(f"{output}/labels.npy", labels)

    print(f"Saved to {output}")
