# Tea Demand Forecasting Backend

## Explainable AI-Based Tea Demand Forecasting System Using TFT

This project contains the backend implementation of the Tea Demand Forecasting component of the AI-Powered Tea Factory Management System.

The backend uses FastAPI and a trained Temporal Fusion Transformer (TFT) model to generate four-week tea demand forecasts for five Sri Lankan tea types.

---

## Research Project

**Project ID:** RP26-IT-171

**Research Component:**  
Design and Implementation of an Explainable AI-Based Tea Demand Forecasting System Using Temporal Fusion Transformer (TFT)

---

## Main Objective

The objective of this component is to provide an explainable weekly tea demand forecasting system that can support tea factories with production and inventory planning.

The system supports:

- BOP
- BOPF
- Dust
- OP
- Pekoe

---

## Main Backend Features

The backend provides functionality for:

- Weekly factory and market data input
- Four-week tea demand forecasting
- Lower, median, and upper forecast values
- Demand trend identification
- Demand alerts
- Production and inventory recommendations
- TFT-based forecast explainability
- Forecast history
- Forecast comparison
- MongoDB persistence
- PDF report generation
- Dataset preprocessing and validation
- Model training and evaluation
- FastAPI Swagger documentation

---

## Forecasting Workflow

The operational workflow uses information from the latest fully completed Monday-to-Sunday week.

The process is:

1. Determine the latest fully completed week.
2. Receive weekly information for all five tea types.
3. Save operational weekly input in MongoDB.
4. Prepare the input for the trained TFT model.
5. Generate demand forecasts for the ongoing week and following three weeks.
6. Generate alerts and recommendations.
7. Provide TFT-based forecast explanation.
8. Save generated forecasts.
9. Maintain forecast history.
10. Generate downloadable PDF reports.

Normal operational forecasting uses the existing trained model and does not retrain the model for every weekly forecast.

---

## Forecast Horizon

The system generates exactly four forecast weeks.

Example:

If data for the completed week **24 August 2026 - 30 August 2026** is entered on **31 August 2026**, the system forecasts:

- Week 1: 31 August 2026
- Week 2: 7 September 2026
- Week 3: 14 September 2026
- Week 4: 21 September 2026

---

## Weekly Input Variables

Weekly operational input includes the following information for each tea type:

- Demand (kg)
- Production (kg)
- Inventory (kg)
- Colombo auction price (LKR)
- Rainfall index
- USD/LKR exchange rate
- Festival flag
- Global commodity shock

The system requires one record for each of the five supported tea types.

---

## Machine Learning Model

The forecasting model is a Temporal Fusion Transformer (TFT).

TFT was selected for multi-horizon time-series forecasting and for its ability to provide model interpretation information.

The trained model produces quantile-based outputs that are used to present:

- Lower forecast
- Expected demand
- Upper forecast

This allows the application to provide a forecast range instead of only a single predicted value.

---

## Model Evaluation

The trained model was evaluated using held-out test data.

| Metric | Result |
|---|---:|
| MAE | 527.02 |
| RMSE | 767.23 |
| R² | 0.88715 |

These metrics describe the performance obtained during model evaluation and are separate from future operational forecasts.

---

## Research Dataset

The research dataset contains weekly time-series records for the five tea types.

Important variables include:

- Week start date
- Tea type
- Demand
- Production
- Inventory
- Colombo auction price
- Auction price lag features
- Rainfall index
- USD/LKR exchange rate
- Festival flag
- Global commodity shock
- Seasonal features
- Demand rolling averages

Operational weekly input is stored separately from the original research dataset.

The original research dataset is therefore not directly modified every time a user generates a new operational forecast.

---

## Explainability

The system uses TFT interpretation information to help explain the generated forecasts.

The explainability functionality summarizes the relative importance of model input variables and provides information about important historical and upcoming variables.

Variable importance represents the model's reliance on available inputs. It should not be interpreted as proof that an input variable directly causes demand to increase or decrease.

---

## Demand Alerts

Forecast demand is compared with historical demand levels.

The application categorizes demand into:

- Low
- Normal
- High

The current alert logic uses percentage differences from the historical baseline to identify meaningful demand changes.

---

## Recommendations

Recommendations are generated using the demand alert results.

They are intended to support decisions related to:

- Production planning
- Inventory preparation
- Stock management
- Response to higher or lower expected demand

The recommendations are decision-support information and do not completely replace human planning decisions.

---

## Forecast Persistence

MongoDB Atlas is used to persist operational information.

Stored information includes:

- Weekly operational input
- Generated forecasts
- Forecast history

This allows the frontend to recover the latest saved information even when temporary browser session data is unavailable.

---

## PDF Reports

The backend uses ReportLab to generate downloadable PDF forecast reports.

Reports contain the generated four-week forecast information for the supported tea types.

---

## Technology Stack

- Python 3.11
- FastAPI
- Uvicorn
- Pandas
- PyTorch
- PyTorch Forecasting
- Lightning
- Scikit-learn
- MongoDB
- PyMongo
- ReportLab

---

## Main Project Structure

```text
tea-demand-backend/
|
|-- app/
|   |-- api/
|   |-- core/
|   |-- db/
|   |-- schemas/
|   |-- services/
|   |-- main.py
|
|-- data/
|-- lightning_logs/
|-- models/
|-- reports/
|-- .env
|-- .env.example
|-- .gitignore
|-- migrate_forecast_history.py
|-- requirements.txt
|-- README.md
```

### `app/api`

Contains FastAPI route definitions.

### `app/core`

Contains application configuration.

### `app/db`

Contains MongoDB connection and repository functionality.

### `app/schemas`

Contains request validation schemas.

### `app/services`

Contains forecasting, preprocessing, training, evaluation, explainability, alerts, recommendations, reporting, and other business logic.

### `data`

Contains research data used by the forecasting pipeline.

### `models`

Contains the trained TFT model checkpoint.

### `lightning_logs`

Contains training-related Lightning logs.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the backend root.

The application configuration includes values such as:

```text
APP_NAME=Tea Demand Forecasting API
MONGODB_URI=<your-mongodb-connection-string>
MONGODB_DATABASE=tea_demand_forecasting
```

Do not commit the real `.env` file or MongoDB credentials to a public repository.

Use `.env.example` as a template for required environment variables.

---

## Running the Backend

Start the FastAPI application using:

```bash
uvicorn app.main:app --reload
```

The API is normally available at:

```text
http://127.0.0.1:8000
```

---

## Swagger API Documentation

After starting the backend, Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used during development and research demonstrations to inspect and test available API endpoints.

---

## Health Check

The backend provides a health endpoint:

```text
GET /health
```

Database connectivity can also be checked through the available database health endpoint.

---

## Important Notes

- The forecasting horizon is four weeks.
- Five tea types are supported.
- Weekly operational data must represent a completed week.
- Operational forecasting uses the previously trained TFT model.
- Normal weekly forecasting does not retrain the model.
- Operational input is stored separately from the original research dataset.
- MongoDB credentials must remain private.
- Forecasts should be treated as decision-support information.

---

## Current Status

The backend currently supports the complete Tea Demand Forecasting workflow, including:

- Weekly data handling
- Four-week TFT forecasting
- Forecast ranges
- Alerts
- Recommendations
- Explainability
- MongoDB persistence
- Forecast history
- Forecast comparison
- PDF report generation
- Model training and evaluation functionality

The backend has been tested together with the React frontend for the complete weekly forecasting workflow.