import pandas as pd
import os
import re

DATABASE_PATH: str = ""
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


def set_database_path(path):
    """Sets the path to the database"""
    global DATABASE_PATH
    DATABASE_PATH = path


# Alias for set_database_path
set_db_path = set_database_path


def check_database_path():
    """Checks if the database path is set"""
    if DATABASE_PATH == "":
        raise ValueError("Database path not set, set it with set_database_path(path)")
    # Check if the path exists
    if not os.path.exists(DATABASE_PATH):
        raise ValueError("Database path does not exist: " + DATABASE_PATH)


def check_that_ticker_exists(ticker: str):
    """Checks if the ticker exists"""
    # Check that path is set
    check_database_path()
    if not os.path.exists(DATABASE_PATH + ticker + ".parquet"):
        raise ValueError("Ticker not in database: " + ticker)


def get_ticker(ticker: str, interval: str = "5Y") -> pd.DataFrame:
    """Returns the ticker dataframe"""
    # Performs the check for the database path and throws an error if it is not set
    check_database_path()
    # Check if the ticker exists
    check_that_ticker_exists(ticker)

    df: pd.DataFrame = pd.read_parquet(DATABASE_PATH + ticker + ".parquet")
    df.index = pd.to_datetime(df.index, unit="s")

    if interval != "max":
        unit = re.search(r"([a-zA-Z]+)", interval)
        if unit is None:
            raise ValueError("Invalid interval: " + interval)
        unit = unit.group(1)
        
        if unit.lower() == "y":
            # Year timedelta isn't supported so we convert to days
            interval = str(int(int(interval[:-1]) * 365)) + "D"
        elif unit.lower() == "mo":
            # Month timedelta isn't supported so we convert to days
            interval = str(int(int(interval[:-1]) * 30)) + "D"
        elif unit.lower() == "m":
            raise ValueError(
                "'M' is ambiguos, use 't' or 'min' for minutes and 'mo' for months"
            )
        
        delta = pd.Timedelta(interval)
        df = pd.DataFrame(df[df.index > df.index[-1] - delta])

    return df

def resample(df: pd.DataFrame, period: str, na="drop"):
    """Resamples the dataframe"""
    resamp = df.resample(period).agg(OHLC)
    if na == "drop":
        resamp.dropna(inplace=True)
    elif na == "fill":
        resamp.fillna(method="ffill", inplace=True)
    return resamp