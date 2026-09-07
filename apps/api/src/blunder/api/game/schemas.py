from blunder.shared.models.game import Game, GameBase, Move, MoveBase


class MoveRead(Move): ...


class MoveCreate(MoveBase): ...


class GameRead(Game):
    moves: list[MoveRead]


class GameCreate(GameBase):
    moves: list[MoveCreate]
