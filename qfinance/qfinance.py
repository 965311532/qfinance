from pathlib import Path
from typing import Optional, Callable, Iterable
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from . import utils

# Allow access from main module
set_database_path = set_db_path = utils.set_database_path


@utils.cached
def load(
    ticker: str,
    interval: str = "5Y",
    freq: Optional[str] = None,
    offset: Optional[str] = None,
    dropna: bool = False,
    init_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """Loads the ticker from the database and performs all initial processing"""
    data = load_ticker_data(ticker=ticker, interval=interval)
    if freq is not None and freq not in ("T", "1T"):  # As that is the default
        data = data.resample(freq, offset=offset).agg({**utils.OHLC, "ts": "first"})
    if dropna:
        data = data.dropna()
    if init_func is not None:
        data = init_func(data)
    return data


def load_ticker_data(ticker: str, interval: str = "5Y") -> pd.DataFrame:
    """Returns the ticker dataframe from the database"""
    # Performs the check for the database path and throws an error if it is not set
    utils.check_database_path()
    # Check if the ticker exists
    utils.check_that_ticker_exists(ticker)

    filepath = Path(utils.DATABASE_PATH, ticker + ".parquet")

    df: pd.DataFrame = pd.read_parquet(str(filepath.absolute()))

    # Set index to datetime but keep ts column
    df["datetime"] = pd.to_datetime(df.index, unit="s", utc=True)
    df = df.reset_index().set_index("datetime")

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
            e = "'M' is ambiguos, use 't' or 'min' for minutes and 'mo' for months"
            raise ValueError(e)

        delta = pd.Timedelta(interval).total_seconds()
        df = df[df.ts > df.ts[-1] - delta]  # type: ignore

    return df


# def bos(df: pd.DataFrame):
#     """A break of structure is defined as a break in the short-term movement of price. In particular a bos happens when a new
#     high is higher (or a new low is lower) than the previous high (or previous low)."""
#     raise NotImplementedError
#     df_ = df.copy()
#     ref_high = df_.high[0]
#     ref_low = df_.low[0]
#     bos_arr = []
#     for row in df_.itertuples():
#         side = 1 if row.close > row.open else -1
#         if row.close - ref_high > 0 and side == 1:
#             pass


def drawdown(df: pd.DataFrame, over: str = "r") -> pd.Series:
    """Calculates the drawdown of the dataframe"""
    return df[over].cumsum() - df[over].cumsum().cummax()


def score(
    df: pd.DataFrame,
    over: str = "r",
    tr_dir_col: str = "tr_dir",
    params: Optional[dict] = None,
) -> dict:
    """Determines the results of the backtest"""

    # Monthly R & max DD
    avg_r_mo = df[over].groupby(pd.Grouper(freq="M")).sum().mean()
    max_dd = -1 * drawdown(df=df, over=over).min()

    # Winrate
    try:
        winrate = df[df[over] > 0].shape[0] / df[df[over] != 0].shape[0]
    except ZeroDivisionError:
        winrate = 0

    # Weekly rolling winning %
    week_resample = df[over].resample("1D").sum().rolling(7).sum().dropna()
    weekly_winning_pct = (week_resample >= 0).sum() / len(week_resample)

    # Long-short ratio
    long_pct = len(df[df[tr_dir_col] == 1]) / len(df)

    # Long r to short r ratio
    long_r_ratio = (
        df[df[tr_dir_col] == 1][over].mean() / df[df[tr_dir_col] == -1][over].mean()
    )

    # Results
    res = {
        "avg_r_mo": avg_r_mo,
        "max_dd": max_dd,
        "r_by_dd": avg_r_mo / max_dd,
        "wr": winrate,
        "weekly_winning_pct": weekly_winning_pct,
        "long_pct": long_pct,
        "long_r_ratio": long_r_ratio,
        "wow": avg_r_mo / max_dd * weekly_winning_pct,
    }

    if params is not None:
        params.update(res)
        res = params

    return res


def approximate_commissions(sl_pips: float | int) -> float:
    """Approximate the commissions for the given stop loss pips.
    Returns the commission amount as the % of the risk amount."""
    # Crazy poly gotten through manual testing.
    return (
        -0.05435693
        + (6669274 - 0.05435693) / (1 + (sl_pips / 0.0000119139) ** 0.9993275)
    ) / 100


def approximate_spread(timestamps: pd.Index, std_dev: float = 0.05) -> pd.Series:
    """Approximate the spread given the timestamps. Returns a series of the spread."""
    map_ = pd.DataFrame(index=range(1440), columns=["spread"], dtype=np.float32)
    map_.iloc[(0, 1439), -1] = 25  # Midnight is the highest spread
    map_.iloc[(5, 1435), -1] = 10  # 5 minutes before and after midnight
    map_.iloc[(30, 1410), -1] = 2.5  # 30 minutes before and after midnight
    map_.iloc[(35, 1405), -1] = 1.5  # 35 minutes before and after midnight
    map_["spread"] = map_.spread.interpolate(method="linear") * np.random.normal(
        1, std_dev, 1440
    )  # make some noise
    minutes = timestamps // 60 % 1440
    return minutes.map(map_["spread"])


def plot_results(
    df: pd.DataFrame, over: str = "r", figsize=(20, 10), adjust_max_dd=10
) -> None:
    """Plots the results of the backtest"""
    # Get score
    score_ = score(df=df, over=over)

    # Plot
    sns.set_theme(style="whitegrid", palette="muted")

    axes: "array of AxesSubplot" = None  # type: ignore
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=300)
    fig.suptitle(
        f"Analysis of results (adjusted for max_dd={adjust_max_dd}r)\n"
        + f"r_by_dd={score_['r_by_dd']:.2f} | "
        + f"wr={score_['wr']:.2%} | "
        + f"weekly_winning={score_['weekly_winning_pct']:.2%} | "
        + f"long_pct={score_['long_pct']:.2%} | "
        + f"long_r_ratio={score_['long_r_ratio']:.2f}"
    )

    # Adjust max dd value
    df[over] = df[over] * adjust_max_dd / (-1 * drawdown(df=df, over=over).min())

    df[over].cumsum().plot(ax=axes[0, 0], title="Cumulative Return", color="green")
    axes[0, 0].set_ylabel("r")
    axes[0, 0].axhline(0)

    drawdown(df).plot(ax=axes[1, 0], title="Drawdown", color="red")
    axes[1, 0].set_ylabel("r")
    axes[1, 0].axhline(-1 * adjust_max_dd)

    monthly = df.groupby(pd.Grouper(freq="M")).sum()
    monthly["month"] = monthly.index.strftime(r"%b %y")
    sns.barplot(data=monthly, x="month", y="r", ax=axes[0, 1], color="blue")
    axes[0, 1].set_title("Monthly Return")
    axes[0, 1].set_xticks(axes[0, 1].get_xticks()[::2])
    axes[0, 1].tick_params(axis="x", rotation=90)
    axes[0, 1].axhline(0)

    sns.histplot(
        ax=axes[1, 1], data=df[over].rolling("30D").sum(), kde=True, color="purple"
    )
    axes[1, 1].set_title("Rolling 30D Return Distribution")
    axes[1, 1].axvline(0)

    fig.tight_layout()
    return plt.show()


def chunks(lst, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parametric_worker(strategy_func: Callable, combination: dict):
    return score(strategy_func(**combination), params=combination)


def parametrize(strategy_func: Callable, params_from: Optional[dict[str, Iterable]] = None, params: Optional[Iterable[dict]] = None, save_to: str = ""):
    """Parametrize the strategy function with the given parameters.
    You can save the data by providing a filepath (must be csv)."""
    if params == None:
        if params_from == None:
            raise ValueError("You must provide at least one of either params_from or parameters")
        # Make the combinations from the provided dictionary
        params = [{k:v for k, v in zip(params_from.keys(), x)} for x in product(*params_from.values())]
    
    results = []

    with tqdm(
        total=len(params),
        desc="Running parametrization",
        unit="sim",
        smoothing=0.05,
    ) as pbar:

        # Start the workers
        with Pool(cpu_count() - 1) as pool:
            for chunk in chunks(params, 64):  # Chunks of 64 combinations
                workers = []
                for combination in chunk:
                    worker = pool.apply_async(
                        parametric_worker,
                        args=(strategy_func, combination),
                        callback=lambda _: pbar.update(1),
                    )
                    workers.append(worker)
                for worker in workers:
                    results.append(worker.get())
                if save_to:
                    # If size is bigger than threshold, save to file to save memory
                    if len(results) >= 500:
                        data = pd.DataFrame(results)
                        data.to_csv(save_to, mode="a", index=False, header=not Path(save_to).exists())
                        results = []
