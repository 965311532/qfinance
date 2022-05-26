import hashlib
import pandas.util

from pandas import DataFrame
from pathlib import Path

OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


def set_database_path(path):
    """Sets the path to the database"""
    global DATABASE_PATH
    DATABASE_PATH = Path(path)
    check_database_path()


def check_database_path():
    """Checks if the database path is set"""
    if not "DATABASE_PATH" in globals():
        raise ValueError("Database path not set, set it with set_database_path(path)")
    # Check if the path exists
    if not DATABASE_PATH.exists():
        raise ValueError("Database path does not exist: " + str(DATABASE_PATH))
    if not DATABASE_PATH.is_dir():
        raise ValueError("Database path is not a directory: " + str(DATABASE_PATH))


def check_that_ticker_exists(ticker: str):
    """Checks if the ticker exists"""
    # Check that path is set
    check_database_path()
    if not Path(DATABASE_PATH, ticker + ".parquet").exists():
        raise ValueError("Ticker not in database: " + ticker)


def hash_df(df: DataFrame) -> str:
    """Returns a hash of the dataframe"""
    return hashlib.sha256(
        pandas.util.hash_pandas_object(df, index=True).values
    ).hexdigest()


def hash_params(params: dict) -> str:
    """Returns a hash of the parameters"""
    # We need to check what type we are hashing,
    # we currentyly support DataFrames, str, int, floats
    matrix = ""

    for k, v in params.items():
        if isinstance(v, DataFrame):
            matrix += f"{k}=" + hash_df(v) + "&"
        elif isinstance(v, float):
            matrix += f"{k}={round(v, 9)}" + "&"
        elif not isinstance(v, (str, int)):
            raise ValueError("Unsupported type: " + str(type(v)))
        else:
            matrix += f"{k}=" + str(v) + "&"

    return hashlib.sha256(matrix.encode("utf-8")).hexdigest()
