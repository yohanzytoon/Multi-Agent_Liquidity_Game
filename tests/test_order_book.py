from src.envs.order_book import LimitOrderBook


def test_limit_order_book_basic():
    book = LimitOrderBook(tick_size=0.01, depth_levels=5)
    book.add_limit_order("buy", 100.0, 10.0)
    book.add_limit_order("sell", 101.0, 10.0)

    assert book.get_best_bid() == 100.0
    assert book.get_best_ask() == 101.0
    assert book.get_spread() == 1.0

    fills = book.add_market_order("buy", 5.0)
    assert fills[0][0] == 101.0
    assert abs(book.get_best_ask() - 101.0) < 1e-12  # partial fill keeps level

