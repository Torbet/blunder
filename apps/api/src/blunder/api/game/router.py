from uuid import UUID

from fastapi import APIRouter, HTTPException

from blunder.api.core.dependencies import Session
from blunder.api.game.models import Game, Move
from blunder.api.game.schemas import GameCreate, GameRead

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/{game_id}")
async def get_game(game_id: UUID, session: Session) -> GameRead:
    game = await session.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameRead.model_validate(game)


@router.post("/")
async def create_games(body: list[GameCreate], session: Session) -> list[GameRead]:
    games = [
        Game(
            **game.model_dump(exclude={"moves"}),
            moves=[Move(**move.model_dump()) for move in game.moves],
        )
        for game in body
    ]
    session.add_all(games)
    await session.commit()
    return [GameRead.model_validate(game) for game in games]
