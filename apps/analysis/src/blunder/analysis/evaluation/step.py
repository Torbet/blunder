import asyncio
import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from itertools import pairwise

import chess.engine

from blunder.analysis.core.pipeline import PipelineContext, PipelineStep
from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.core.records import GameRecord

MATE = 10000


class EvaluationStep(PipelineStep):
    def __init__(self, config: EvaluationConfig) -> None:
        self.limit = chess.engine.Limit(depth=config.depth)
        self.engines: list[chess.engine.UciProtocol] = []

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[None]:
        engines = await asyncio.gather(
            *(chess.engine.popen_uci("stockfish") for _ in range(os.process_cpu_count() or 1))
        )
        self.engines = [engine for _, engine in engines]

        try:
            yield
        finally:
            await asyncio.gather(*(engine.quit() for engine in self.engines))

    async def run(self, ctx: PipelineContext) -> None:
        games = iter(ctx.games)
        await asyncio.gather(*(self.work(engine, ctx, games) for engine in self.engines))

    async def work(self, engine: chess.engine.UciProtocol, ctx: PipelineContext, games: Iterator[GameRecord]) -> None:
        for game in games:
            infos = [await engine.analyse(board, self.limit, game=game.id) for board in game.positions()]

            for move, (before, after) in zip(game.moves, pairwise(infos), strict=True):
                feature = ctx.feature(game, move.ply)
                feature.evaluation = after["score"].white().score(mate_score=MATE)
                feature.best = before["pv"][0].uci()
