from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Timestamped(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    created_at: datetime
    updated_at: datetime


class Identifiable(Timestamped):
    id: UUID
