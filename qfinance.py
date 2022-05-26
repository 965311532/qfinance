import pandas as pd
import re

from pathlib import Path

import utils

# Allow access from main module
set_database_path = set_db_path = utils.set_database_path


@utils.cached
def get_ticker(ticker: str, interval: str = "5Y") -> pd.DataFrame:
    """Returns the ticker dataframe"""
    # Performs the check for the database path and throws an error if it is not set
    utils.check_database_path()
    # Check if the ticker exists
    utils.check_that_ticker_exists(ticker)

    filepath = Path(utils.DATABASE_PATH, ticker + ".parquet")

    df: pd.DataFrame = pd.read_parquet(str(filepath.absolute()))
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


def bos(df: pd.DataFrame):
    """A break of structure is defined as a break in the short-term movement of price. In particular a bos happens when a new
    high is higher (or a new low is lower) than the previous high (or previous low)."""
    raise NotImplementedError
    df_ = df.copy()
    ref_high = df_.high[0]
    ref_low = df_.low[0]
    bos_arr = []
    for row in df_.itertuples():
        side = 1 if row.close > row.open else -1
        if row.close - ref_high > 0 and side == 1:
            pass


@utils.cached
def resample(
    df: pd.DataFrame, period: str, na="drop", offset: str | None = None
) -> pd.DataFrame:
    """Resamples the dataframe"""
    resamp = df.resample(period, offset=offset).agg(utils.OHLC)
    if na == "drop":
        resamp.dropna(inplace=True)
    elif na == "fill":
        resamp.fillna(method="ffill", inplace=True)
    return resamp


def drawdown(df: pd.DataFrame, over: str = "r") -> pd.Series:
    """Calculates the drawdown of the dataframe"""
    return df[over].cumsum() - df[over].cumsum().cummax()


def describe(
    df: pd.DataFrame,
    over="r",
    granularity="30D",
    mode="sum",
    percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> pd.DataFrame:
    """Describes the dataframe"""
    return (
        df[over]
        .groupby(pd.Grouper(freq=granularity))
        .__getattr__(mode)()
        .describe(percentiles=percentiles)
    )


def main():
    """Tests the module"""
    # Set the database path
    # set_database_path("/home/jose/qfinance/database") # should raise ValueError
    set_database_path("C:\\FXDB")

    for ticker in utils.get_tickers():
        for _ in range(1000):
            get_ticker(ticker)


if __name__ == "__main__":
    main()
