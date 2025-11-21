"""Simple limit order book implementation for the synthetic market."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sortedcontainers import SortedDict


Trade = Tuple[float, float]  # price, size


@dataclass
class LimitOrderBook:
    """
    Minimal price-level order book.
    Bids are stored with descending sort order, asks ascending.
    """

    tick_size: float = 0.01
    depth_levels: int = 5

    def __post_init__(self) -> None:
        self.bids: SortedDict[float, float] = SortedDict(lambda x: -x)
        self.asks: SortedDict[float, float] = SortedDict()

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def _round_price(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def get_best_bid(self) -> Optional[float]:
        return self.bids.peekitem(0)[0] if len(self.bids) > 0 else None

    def get_best_ask(self) -> Optional[float]:
        return self.asks.peekitem(0)[0] if len(self.asks) > 0 else None

    def get_midprice(self) -> Optional[float]:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def get_spread(self) -> Optional[float]:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def add_limit_order(self, side: str, price: float, size: float) -> List[Trade]:
        """
        Add a limit order and match if it crosses the spread.
        Returns list of trades executed.
        """
        price = self._round_price(price)
        fills, remaining = self._match(side, size, limit_price=price)
        if remaining > 0:
            book = self.bids if side == "buy" else self.asks
            book[price] = book.get(price, 0.0) + remaining
        return fills

    def add_market_order(self, side: str, size: float) -> List[Trade]:
        """Execute a market order and return fills as (price, size)."""
        fills, _ = self._match(side, size, limit_price=None)
        return fills

    def cancel(self, side: str, price: float, size: Optional[float] = None) -> None:
        """Cancel quantity at a price level (or entire level if size is None)."""
        price = self._round_price(price)
        book = self.bids if side == "buy" else self.asks
        if price not in book:
            return
        if size is None or size >= book[price]:
            del book[price]
        else:
            book[price] -= size

    def get_top_levels(self, side: str, levels: Optional[int] = None) -> List[Trade]:
        """Return best N price levels for side ("buy" or "sell")."""
        levels = levels or self.depth_levels
        book = self.bids if side == "buy" else self.asks
        return list(book.items())[:levels]

    def depth_arrays(self, levels: Optional[int] = None) -> Tuple[List[float], List[float]]:
        """Return top-of-book depth arrays for bids and asks."""
        levels = levels or self.depth_levels
        bid_prices, bid_sizes = zip(*self.get_top_levels("buy", levels)) if len(self.bids) else ([], [])
        ask_prices, ask_sizes = zip(*self.get_top_levels("sell", levels)) if len(self.asks) else ([], [])
        # Keep lengths consistent
        bid_prices = list(bid_prices) + [0.0] * max(0, levels - len(bid_prices))
        ask_prices = list(ask_prices) + [0.0] * max(0, levels - len(ask_prices))
        bid_sizes = list(bid_sizes) + [0.0] * max(0, levels - len(bid_sizes))
        ask_sizes = list(ask_sizes) + [0.0] * max(0, levels - len(ask_sizes))
        return (bid_prices + ask_prices, bid_sizes + ask_sizes)

    def _match(
        self,
        side: str,
        size: float,
        limit_price: Optional[float],
    ) -> Tuple[List[Trade], float]:
        """
        Match an incoming order against the book.
        Returns fills and any residual unfilled size.
        """
        fills: List[Trade] = []
        book = self.asks if side == "buy" else self.bids

        while size > 0 and len(book) > 0:
            best_price, available = book.peekitem(0)
            if limit_price is not None:
                if side == "buy" and best_price > limit_price:
                    break
                if side == "sell" and best_price < limit_price:
                    break
            traded = min(size, available)
            fills.append((best_price, traded))

            if traded >= available:
                book.popitem(0)
            else:
                book[best_price] = available - traded

            size -= traded
        return fills, size

