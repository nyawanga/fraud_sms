from sqlalchemy import Column, Integer, String, Numeric

from database import Base

class FraudSMS(Base):
    __tablename__="fraud_sms"

    id = Column(Integer, primary_key=True, index=True)
    sms_text = Column(String)
    #prediction = Column(Integer)
    probability = Column(Numeric(10,2))


