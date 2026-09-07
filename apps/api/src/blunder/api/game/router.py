from uuid import UUID

from fastapi import APIRouter, HTTPException

from blunder.api.core.dependencies import Session
from blunder.api.game.models import Game, Move
from blunder.api.game.schemas import GameCreate, GameRead, MoveCreate, MoveRead

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/{game_id}")
async def read_game(game_id: UUID, session: Session) -> GameRead:
    game = await session.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameRead.model_validate(game)


@router.post("/")
async def create_game(body: GameCreate, session: Session) -> GameRead:
    game = Game(**body.model_dump())
    session.add(game)
    await session.commit()
    await session.refresh(game)
    return GameRead.model_validate(game)


@router.post("/{game_id}/moves")
async def create_moves(game_id: UUID, body: list[MoveCreate], session: Session) -> list[MoveRead]:
    game = await session.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = [Move(game_id=game_id, **move.model_dump()) for move in body]
    session.add_all(moves)
    await session.commit()
    return [MoveRead.model_validate(move) for move in moves]
