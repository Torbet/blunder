import chess.engine
from rich.progress import Progress

from blunder.analysis.core.pipeline import PipelineStep
from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.core.records import GameRecord


class EvaluationStep(PipelineStep):
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    async def run(self, games: list[GameRecord]) -> None:
        _, engine = await chess.engine.popen_uci("stockfish")
        limit = chess.engine.Limit(depth=self.config.depth)

        with Progress() as progress:
            task = progress.add_task("Evaluating", total=sum(len(game.moves) for game in games))

            for game in games:
                for move in game.moves:
                    board = game.board(move.ply + 1)
                    evaluation = await engine.analyse(board, limit)
                    move.evaluation = evaluation["score"].white().score(mate_score=10000)
                    progress.advance(task)

        await engine.quit()
