import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


load_dotenv()


MONGODB_URI = os.getenv(
    "MONGODB_URI"
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "tea_demand_forecasting",
)


if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI is not configured "
        "in the .env file."
    )


client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
)


database: Database = client[
    MONGODB_DATABASE
]


def get_database() -> Database:
    """
    Return the MongoDB database used
    by the Tea Demand Forecasting
    component.
    """
    return database


def check_database_connection() -> bool:
    """
    Check whether the backend can
    communicate with MongoDB Atlas.
    """
    client.admin.command(
        "ping"
    )

    return True