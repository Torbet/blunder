import numpy as np
import chess
from stockfish import Stockfish
from lczero.backends import Weights, Backend, GameState
from preprocess import parse_board
from helpers.evaluate import evaluate_stockfish
import csv
import os
import shutil

# Parameters
limit = 10000
num_moves = 40
engine_prob = 0.3

# Stockfish setup
stockfish = Stockfish(path="/home/gtorbet/downloads/stockfish/stockfish-ubuntu-x86-64-avx2")


def stockfish_move(board: chess.Board) -> chess.Move:
    stockfish.set_fen_position(board.fen())
    return chess.Move.from_uci(stockfish.get_best_move())


# Maia setup
weights = Weights("data/maia-1900.pb.gz")
backend = Backend(weights)


def maia_move(board: chess.Board) -> chess.Move:
    state = GameState(fen=board.fen())
    output = backend.evaluate(state.as_input(backend))[0]
    moves = list(zip(state.moves(), output.p_softmax(*state.policy_indices())))
    return chess.Move.from_uci(max(moves, key=lambda x: x[1])[0]) if moves else None


# Load openings
with open("data/openings.tsv", "r") as f:
    openings = [l[4] for l in csv.reader(f, delimiter="\t") if 5 <= len(l[3].split(" ")) <= 10]

moves = np.zeros((4 * limit, num_moves, 6, 8, 8))
labels = np.zeros((4 * limit, num_moves))
evals = np.zeros((4 * limit, num_moves))

for c in range(4):
    for g in range(limit):
        board = chess.Board(openings[np.random.randint(len(openings))])

        for i in range(num_moves):
            if board.is_game_over():
                print(f"Game over in class {c}, game {g}, move {i}. Skipping...")
                break

            r = np.random.rand()

            if c == 0:  # HvH
                move, label = maia_move(board), 0
            elif c == 1:  # HvC
                if board.turn == chess.WHITE:
                    move, label = maia_move(board), 0
                else:
                    move = stockfish_move(board) if r < engine_prob else maia_move(board)
                    label = int(r < engine_prob)
            elif c == 2:  # CvH
                if board.turn == chess.WHITE:
                    move = stockfish_move(board) if r < engine_prob else maia_move(board)
                    label = int(r < engine_prob)
                else:
                    move, label = maia_move(board), 0
            else:  # CvC
                move = stockfish_move(board) if r < engine_prob else maia_move(board)
                label = int(r < engine_prob)

            # Record data
            board.push(move)
            moves[c * limit + g, i] = parse_board(board)
            evals[c * limit + g, i], _ = evaluate_stockfish(board)
            labels[c * limit + g, i] = label

        print(f"Class {c}, Game {g}/{limit}, Completed.")

# Save dataset
output_dir = f"data/synthesized/{limit}"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

np.save(f"{output_dir}/moves.npy", moves)
np.save(f"{output_dir}/labels.npy", labels)
np.save(f"{output_dir}/evals.npy", evals)
print("Dataset saved.")
