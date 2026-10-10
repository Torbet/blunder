from blunder.shared.models.game import Feature, FeatureBase, Game, GameBase, Move, MoveBase


class GameRead(Game):
    moves: list[MoveRead]


class GameCreate(GameBase):
    moves: list[MoveCreate]


class MoveRead(Move): ...


class MoveCreate(MoveBase): ...


class FeatureRead(Feature): ...


class FeatureCreate(FeatureBase): ...
