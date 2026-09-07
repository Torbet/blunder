from blunder.shared.models.game import Game, GameBase, Move, MoveBase


class GameRead(Game):
    moves: list[Move]


class GameCreate(GameBase): ...


class MoveRead(Move): ...


class MoveCreate(MoveBase): ...
