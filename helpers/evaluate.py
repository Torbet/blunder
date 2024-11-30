import chess
from stockfish import Stockfish

stockfish = Stockfish(depth=8, path="/home/gtorbet/downloads/stockfish/stockfish-ubuntu-x86-64-avx2")


def evaluate_stockfish(board: chess.Board, num_best_moves: int = 5) -> tuple[int, list[str]]:
    # Evaluate the board position using Stockfish.
    stockfish.set_fen_position(board.fen())
    evaluation = stockfish.get_evaluation()
    # top 5 best moves
    best_moves = [x["Move"] for x in stockfish.get_top_moves(num_best_moves)]
    if evaluation["type"] == "mate":
        return 10000 if evaluation["value"] > 0 else -10000, best_moves
    else:
        return evaluation["value"], best_moves


def evaluate_lazy(board: chess.Board) -> int:
    # Evaluate the board position with a material count
    PV = {"pawn": 100, "knight": 320, "bishop": 330, "rook": 500, "queen": 950}

    if board.is_insufficient_material():
        return 0

    piece_counts = {
        color: {
            piece: len(board.pieces(piece_type, color))
            for piece, piece_type in zip(
                PV.keys(),
                [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN],
            )
        }
        for color in (chess.WHITE, chess.BLACK)
    }

    value = sum(PV[piece] * (piece_counts[chess.WHITE][piece] - piece_counts[chess.BLACK][piece]) for piece in PV)

    return value if board.turn == chess.WHITE else -value
