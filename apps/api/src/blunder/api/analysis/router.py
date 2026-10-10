from uuid import UUID

from fastapi import APIRouter, HTTPException

from blunder.api.analysis.models import Analysis
from blunder.api.analysis.schemas import AnalysisCreate, AnalysisRead
from blunder.api.core.dependencies import Session

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.get("/{analysis_id}")
async def get_analysis(analysis_id: UUID, session: Session) -> AnalysisRead:
    analysis = await session.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return AnalysisRead.model_validate(analysis)


@router.post("/")
async def create_analysis(body: AnalysisCreate, session: Session) -> AnalysisRead:
    analysis = Analysis(**body.model_dump())
    session.add(analysis)
    await session.commit()
    return AnalysisRead.model_validate(analysis)
