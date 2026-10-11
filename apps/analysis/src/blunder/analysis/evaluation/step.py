from itertools import pairwise

import chess.engine
from rich import progress

from blunder.analysis.core.pipeline import PipelineContext, PipelineStep
from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.models.game import FeatureBase


class EvaluationStep(PipelineStep):
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    async def run(self, ctx: PipelineContext) -> None:
        _, engine = await chess.engine.popen_uci("stockfish")
        limit = chess.engine.Limit(depth=self.config.depth)

        for game in progress.track(ctx.games, description="Evaluating"):
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

        await engine.quit()
