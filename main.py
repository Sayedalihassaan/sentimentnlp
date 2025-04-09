from fastapi import FastAPI , HTTPException , Depends
from src.helpers.config import APP_NAME , VERSION , API_SECRET_KEY
from src.controllers.NLPTrainer import NLPTrainer
from src.models.request import TrainingData , TestingData , QueryText
from src.models.response import StatusObject , PredictionObject , PredictionsObjects

from fastapi.security import API


trainer = NLPTrainer()

app = FastAPI(title=APP_NAME , version = VERSION)

app.add_middleware(

)