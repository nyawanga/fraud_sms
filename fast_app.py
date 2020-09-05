from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from database import SessionLocal, engine
from sqlalchemy.orm import Session
import models
from models import FraudSMS
from pydantic import BaseModel

from lib import custom_processor
import joblib
import sys


app = FastAPI()

models.Base.metadata.create_all(bind=engine)   # create the tables specified in the models table

templates = Jinja2Templates(directory="templates")

class FraudSMSRequest(BaseModel):
    text: str

def get_db():
    try:
        db = SessionLocal()
        yield db

    finally:
        db.close()

@app.get("/")
def home(request: Request):
    """
    This is the hoempage

    """
    return templates.TemplateResponse( "sms_predictions.html", {
        "request" : request
        } )

def make_prediction(id: int):
    """
    This is what make the predictions
    """
    db = SessionLocal()
    sms_table = db.query(FraudSMS).filter(FraudSMS.id == id).first()
    model_path = "./lib/xgb_model.pkl"
    xgb_model = joblib.load(model_path)

    try:
        raw = sms_table.sms_text
        processor = custom_processor.InputTransformer()
        data = processor.transform(raw)
        print(data)
    except ValueError:
        print(
            "Request has no data or is not a valid json object"
            )

    predictions = xgb_model.predict(data)
    probability = xgb_model.predict_proba(data)
    results = dict()

    sms_table.probability = list(probability)[0][1]
   # sms_table.prediction = list(predictions)[0]

    db.add(sms_table)
    db.commit()

@app.post("/predict")
async def predict(fraud_sms: FraudSMSRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    this is the function that makes the prediction prediction
    """

    sms_table = FraudSMS()                                 #intiialize the table object
    sms_table.sms_text = fraud_sms.text                    # assign property text

    db.add(sms_table)                                      # add the table data
    db.commit()                                            # commit changes

    # this runs the predictions in the background from the function make_prediction
    background_tasks.add_task(make_prediction, sms_table.id)

    return {
        "code" : "success",
        "message" : "this post was successfull"
    }
