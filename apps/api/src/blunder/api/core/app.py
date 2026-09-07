from fastapi import FastAPI

from blunder.api.game.router import router as game_router

app = FastAPI()

app.include_router(game_router)
