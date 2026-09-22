from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import warnings

# Inicializar la aplicación de FastAPI
app = FastAPI(
    title="Titanic Survival API",
    description="API RESTful para la predicción de supervivencia en el Titanic utilizando Random Forest",
    version="1.0"
)

warnings.filterwarnings("ignore", category=UserWarning)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde cualquier origen (tu archivo HTML)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],
)





# Cargar el modelo entrenado previamente
model = joblib.load('random_forest_model.pkl')

# Definir el esquema de datos de entrada esperado con Pydantic
class PassengerInput(BaseModel):
    Pclass: int
    Sex: int  # 0: Femenino, 1: Masc
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Family_Size: int
    Fare_Per_Person: float

@app.get("/")
def read_root():
    return {"message": "¡API del Titanic activa! Ve a http://127.0.0.1 para probarla."}

@app.post("/predict")
def predict_survival(data: PassengerInput):
    # Organizar los datos en un DataFrame con las características exactas
    input_df = pd.DataFrame([{
        'Pclass': data.Pclass,
        'Sex': data.Sex,
        'Age': data.Age,
        'SibSp': data.SibSp,
        'Parch': data.Parch,
        'Fare': data.Fare,
        'Family_Size': data.Family_Size,
        'Fare_Per_Person': data.Fare_Per_Person
    }])
    
    # Realizar la inferencia con el modelo de Random Forest
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    return {
        "prediction": prediction,
        "probability": probability,
        "status": "Success",
        "message": "Sobrevivió" if prediction == 1 else "No sobrevivió"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)