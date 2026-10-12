from uuid import UUID, uuid4

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from blunder.api.core.database import Base
from blunder.shared.analysis import PipelineConfig


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    config: Mapped[PipelineConfig] = mapped_column(JSONB)
