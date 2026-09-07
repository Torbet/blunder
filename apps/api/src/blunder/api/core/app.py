from fastapi import FastAPI

from blunder.api.game.router import router as game_router

app = FastAPI(generate_unique_id_function=lambda route: route.name)

app.include_router(game_router)
