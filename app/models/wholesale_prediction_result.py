from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String
from app.db.base_class import Base

class WholesalePredictionResult(Base):
    __tablename__ = "wholesale_predicted_prices"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(100), index=True, nullable=False)
    model_name = Column(String(255), nullable=True)

    sinhala_name = Column(String(255), nullable=True)
    common_name = Column(String(255), nullable=True)

    year = Column(Integer, nullable=True)
    month = Column(String(50), nullable=True)
    week = Column(Integer, nullable=True)
    week_label = Column(String(100), nullable=True)

    predicted_price = Column(Numeric(10, 2), nullable=True)

    source_long_file = Column(String(255), nullable=True)
    source_prediction_file = Column(String(255), nullable=True)

    is_published = Column(Boolean, default=False)
    published_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))