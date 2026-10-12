from itertools import chain

from blunder.analysis.core.pipeline import PipelineContext, PipelineStep
from blunder.maia.core.maia import Maia
from blunder.maia.models.position import Position
from blunder.shared.analysis.maia import MaiaConfig


class MaiaStep(PipelineStep):
    def __init__(self, config: MaiaConfig) -> None:
        self.maia = Maia(config.model, batch=config.batch)

    async def run(self, ctx: PipelineContext) -> None:
        games = ctx.games
        moves = [(game, move) for game in games for move in game.moves]
        predictions = iter(self.maia.predict(chain.from_iterable(map(Position.replay, games))))

        for (game, move), prediction in zip(moves, predictions, strict=True):
            feature = ctx.feature(game, move.ply)
            feature.probability = prediction.policy[move.uci]
            feature.entropy = prediction.entropy
            feature.ponder = prediction.ponder
            feature.win = prediction.wdl.win
            feature.draw = prediction.wdl.draw
            feature.loss = prediction.wdl.loss
