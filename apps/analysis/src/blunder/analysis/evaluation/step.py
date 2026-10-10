from itertools import pairwise

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
                infos = [await engine.analyse(board, limit, game=game.id) for board in game.positions()]

                for move, (before, after) in zip(game.moves, pairwise(infos), strict=True):
                    game.features.append(
                        FeatureBase(
                            analysis_id=ctx.analysis.id,
                            ply=move.ply,
                            evaluation=after["score"].white().score(mate_score=10000),
                            best=before["pv"][0].uci(),
                        )
                    )

                progress.advance(task, len(game.moves))

        await engine.quit()
