import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

from src.misc import (
    GenderEnum,
    Lunch,
    Parental_Level_Of_Eductaion,
    RaceEthnicity,
    TestPreparationCourse,
)
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    """Render the index.html template.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        HTMLResponse: The rendered HTML response.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predictdata")
async def predict_datapoint(
    gender: GenderEnum = Form(title="Gender", description="Select your gender"),
    race_ethnicity: RaceEthnicity = Form(
        title="Race Or Ethnicity", description="Select your race or ethinicity"
    ),
    parental_level_of_education: Parental_Level_Of_Eductaion = Form(
        title="Parental level of education",
        description="Select level of your parent's education",
    ),
    lunch: Lunch = Form(title="Lunch", description="Enter Lunch type"),
    test_preparation_course: TestPreparationCourse = Form(
        title="Test preparation course", description="Enter test preparation course"
    ),
    reading_score: int = Form(title="Reading Score", description="Enter reading score"),
    writing_score: int = Form(title="Writing Score", description="Enter writing score"),
):
    """Predict the math score based on input data.

    Args:
        gender (GenderEnum): Gender of the student.
        race_ethnicity (RaceEthnicity): Race or ethnicity of the student.
        parental_level_of_education (Parental_Level_Of_Eductaion): Parental level of education.
        lunch (Lunch): Type of lunch.
        test_preparation_course (TestPreparationCourse): Test preparation course.
        reading_score (int): Reading score of the student.
        writing_score (int): Writing score of the student.

    Returns:
        float: Predicted math score.
    """
    data = {
        "gender": [gender.value],
        "race_ethnicity": [race_ethnicity.value],
        "parental_level_of_education": [parental_level_of_education.value],
        "lunch": [lunch.value],
        "test_preparation_course": [test_preparation_course.value],
        "reading_score": [reading_score],
        "writing_score": [writing_score],
    }

    pred_df = pd.DataFrame(data)
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)
    output = f"The predicted output is {np.round(result[0])}"
    return output
