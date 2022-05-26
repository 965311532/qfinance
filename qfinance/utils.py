import hashlib
import pandas.util
import itertools
import sys

from pandas import DataFrame
from pathlib import Path

OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


class SimpleCache:
    """Simple cache class with limited capacity"""

    __slots__ = ["capacity", "cache"]

    def __init__(self, capacity: int):
        self.capacity = capacity  # in bytes
        self.cache = {}

    def __contains__(self, key: str) -> bool:
        """Returns True if the key is in the cache"""
        return key in self.cache

    def set_capacity(self, capacity: int) -> None:
        """Sets the cache capacity"""
        self.capacity = capacity
        self.manage_storage()

    def manage_storage(self):
        """Check if the cache is full and if so, delete the oldest item"""
        if total_size(self.cache) > self.capacity and len(self.cache) > 0:
            self.cache.pop(next(iter(self.cache)))

    def get(self, key: str) -> object:
        """Returns the cached object"""
        return self.cache[key]

    def set(self, key: str, value: object) -> None:
        """Sets the cached object"""
        self.cache[key] = value
        self.manage_storage()


# Istantiate the cache
CACHE = SimpleCache(capacity=100_000_000)  # 100 MB


def set_cache_capacity(capacity: int):
    """Sets the cache capacity"""
    CACHE.set_capacity(capacity)


def cached(func):
    """Decorator to cache the results of a function"""

    def wrapper(*args, **kwargs):
        # Get the hash of the parameters
        params_hash = hash_params(
            {**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs}
        )
        # Check if the hash is in the cache
        if params_hash in CACHE:
            return CACHE.get(params_hash)
        # If not, call the function and cache the result
        result = func(*args, **kwargs)
        CACHE.set(params_hash, result)
        return result

    return wrapper


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
        elif not isinstance(v, (str, int, type(None))):
            raise ValueError("Unsupported type: " + str(type(v)) + " for " + str(v))
        else:
            matrix += f"{k}=" + str(v) + "&"

    return hashlib.sha256(matrix.encode("utf-8")).hexdigest()


def get_tickers() -> list[str]:
    """Returns a list of all tickers in the database"""
    check_database_path()
    return [str(p.stem) for p in Path(DATABASE_PATH).glob("*.parquet")]


def total_size(o, handlers={}):
    """Returns the approximate memory footprint an object and all of its contents."""
    dict_handler = lambda d: itertools.chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }

    all_handlers.update(handlers)
    seen = set()
    default_size = sys.getsizeof(0)

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
