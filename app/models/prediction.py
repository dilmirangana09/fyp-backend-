from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone

from app.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    fish_name = Column(String(100), nullable=False)
    predicted_price = Column(Float, nullable=False)
    week = Column(String(20), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )