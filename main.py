from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import sqlite3
import datetime


app = FastAPI()

# Enable CORS (IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    department_model = joblib.load("models/department_model.pkl")
    risk_model = joblib.load("models/risk_model.pkl")
except:
    department_model = None
    risk_model = None

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptom TEXT,
            department TEXT,
            risk TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class SymptomRequest(BaseModel):
    symptom: str

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "AI Health Backend Running Successfully 🚀"}


@app.post("/predict")
def predict(data: SymptomRequest):
    symptom_text = data.symptom

    # If ML models exist
    if department_model and risk_model:
        department = department_model.predict([symptom_text])[0]
        risk = risk_model.predict([symptom_text])[0]
    else:
        # Fallback if model not loaded
        department = "General Medicine"
        risk = "Low"

    # Save to database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patient_logs (symptom, department, risk, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        symptom_text,
        department,
        risk,
        str(datetime.datetime.now())
    ))
    conn.commit()
    conn.close()

    return {
        "department": department,
        "risk": risk
    }

@app.post("/chat")
def chatbot(data: ChatRequest):
    message = data.message.lower()

    if "fever" in message:
        response = "Please monitor your temperature and stay hydrated."
    elif "chest pain" in message:
        response = "Chest pain can be serious. Please seek immediate medical help."
    elif "headache" in message:
        response = "Take rest and drink water. If severe, consult a doctor."
    else:
        response = "Please describe your symptoms clearly."

    return {"reply": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
