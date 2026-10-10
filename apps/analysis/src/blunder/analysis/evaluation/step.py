import chess
import chess.engine
from rich.progress import Progress

from blunder.analysis.core.pipeline import PipelineContext, PipelineStep
from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.models.game import FeatureBase


class EvaluationStep(PipelineStep):
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    async def run(self, ctx: PipelineContext) -> None:
        _, engine = await chess.engine.popen_uci("stockfish")
        limit = chess.engine.Limit(depth=self.config.depth)

        with Progress() as progress:
            task = progress.add_task("Evaluating", total=sum(len(game.moves) for game in ctx.games))

            for game in ctx.games:
                board = chess.Board()
                info = await engine.analyse(board, limit)

                for move in game.moves:
                    best = info["pv"][0].uci()
                    board.push_uci(move.uci)
                    info = await engine.analyse(board, limit)
                    game.features.append(
                        FeatureBase(
                            analysis_id=ctx.analysis.id,
                            ply=move.ply,
                            evaluation=info["score"].white().score(mate_score=10000),
                            best=best,
                        )
                    )
                    progress.advance(task)

        await engine.quit()
