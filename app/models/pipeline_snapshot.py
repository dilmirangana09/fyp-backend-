from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from app.db.base import Base


class PipelineSnapshot(Base):
    __tablename__ = "pipeline_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    pipeline_type = Column(String(50), nullable=False, index=True)   # retail / wholesale
    stage = Column(String(50), nullable=False, index=True)           # merged / filtered / interpolated / long_format
    week_label = Column(String(255), nullable=True)
    filename = Column(String(255), nullable=True)
    data_json = Column(Text, nullable=False)
    row_count = Column(Integer, nullable=False, default=0)
    is_latest = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)