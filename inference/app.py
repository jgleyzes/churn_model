from enum import Enum
from pydantic import BaseModel, Field
from config import MODEL_PATH, THRESHOLD
from inference_model import LGBMInferenceModel

inference_model = LGBMInferenceModel(model_path=MODEL_PATH, threshold=THRESHOLD)


class GeographyEnum(Enum):
    Germany = "Germany"
    France = "France"
    Spain = "Spain"


class GenderEnum(Enum):
    Female = "Female"
    Male = "Male"


class FeaturesInput(BaseModel):
    Age: int = Field(..., ge=0, le=150, description="Age of customer")
    Tenure: int = Field(..., ge=0, le=15, description="Number of years as customer")
    NumOfProducts: int = Field(
        ..., ge=0, le=4, description="Number of banking product for customer"
    )
    CreditScore: int = Field(..., ge=0, le=850, description="Credit Score of customer")
    Balance: float = Field(..., ge=0, description="Amount in the account")
    EstimatedSalary: float = Field(
        ..., ge=0, description="Estimated Salary of customer"
    )
    HasCrCard: bool = Field(..., description="Whether customer has a credit card")
    IsActiveMember: bool = Field(..., description="Whether customer is active")

    Geography: GeographyEnum
    Gender: GenderEnum

    class Config:
        use_enum_values = True


class Output(BaseModel):
    will_churn: bool


def predict(input: FeaturesInput) -> Output:
    """Returns the `message` of the input data."""

    will_churn = inference_model.predict(input.dict())
    return Output(will_churn=will_churn)
