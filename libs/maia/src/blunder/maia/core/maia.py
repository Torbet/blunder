from collections.abc import Iterable
from itertools import batched
from types import SimpleNamespace

import chess
import torch
from maia3.dataset import get_historical_tokens, tokenize_board
from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
from maia3.uci import load_model
from maia3.utils import get_all_possible_moves, mirror_move

from blunder.maia.models.position import Position
from blunder.maia.models.prediction import Prediction, Wdl


class Maia:
    def __init__(self, model: str = "maia3-23m-ponder", device: str | None = None, batch: int = 256) -> None:
        spec = resolve_model_spec(model)
        self.device = device or str(torch.accelerator.current_accelerator() or "cpu")
        self.config = SimpleNamespace(
            **spec.config,
            device=self.device,
            checkpoint_path=resolve_checkpoint_path(spec),
        )
        self.model = load_model(self.config)
        self.indices = {uci: index for index, uci in enumerate(get_all_possible_moves())}
        self.batch = batch

    @torch.inference_mode()
    def predict(self, positions: Iterable[Position]) -> list[Prediction]:
        return [prediction for batch in batched(positions, self.batch) for prediction in self._predict(batch)]

    def _predict(self, positions: tuple[Position, ...]) -> list[Prediction]:
        tokens = torch.stack([self._tokens(position) for position in positions])
        self_elos, oppo_elos = torch.tensor(
            [(p.elos.white, p.elos.black) if p.board.turn else (p.elos.black, p.elos.white) for p in positions]
        ).T

        moves, values, ponders = self.model(
            tokens.to(self.device), self_elos.to(self.device), oppo_elos.to(self.device)
        )
        moves, values, ponders = moves.cpu(), values.softmax(-1).tolist(), ponders.tolist()

        predictions = []

        for position, logits, (loss, draw, win), ponder in zip(positions, moves, values, ponders):
            if not position.board.turn:
                win, loss = loss, win

            legal = [move.uci() for move in position.board.legal_moves]
            probabilities = logits[[self._index(position.board, uci) for uci in legal]].softmax(-1)
            predictions.append(
                Prediction(
                    policy=dict(zip(legal, probabilities.tolist())),
                    wdl=Wdl(win=win, draw=draw, loss=loss),
                    ponder=ponder if self.config.include_time_info else None,
                )
            )

        return predictions

    def _index(self, board: chess.Board, uci: str) -> int:
        return self.indices[uci if board.turn else mirror_move(uci)]

    def _tokens(self, position: Position) -> torch.Tensor:
        board = position.board.copy()
        history = [tokenize_board(board)]

        while len(history) < self.config.history and board.move_stack:
            board.pop()
            history.append(tokenize_board(board))

        return get_historical_tokens(
            history[::-1],
            self.config,
            base=position.control.base,
            inc=position.control.increment,
            clk_left_before=position.clock,
            clk_ponder=0.0,
        )
