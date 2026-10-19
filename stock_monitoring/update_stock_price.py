import sqlite3
import yfinance as yf
from datetime import datetime, timezone

DB_PATH = "investments.db"


def update_stock_prices():
    conn = sqlite3.connect(DB_PATH)

    try:
        # 1. Fetch all active investments
        investments = conn.execute("""
            SELECT id, ticker, entry_price
            FROM monitor_investments
            WHERE status = 'active'
        """).fetchall()

        for investment_id, ticker, entry_price in investments:
            try:
                # 2. Fetch latest available market closing price
                history = yf.Ticker(ticker).history(period="5d")

                if history.empty:
                    print(f"No price data for {ticker}")
                    continue

                closes = history["Close"].dropna()
                if closes.empty:
                    print(f"No valid closing price for {ticker}")
                    continue

                current_price = float(closes.iloc[-1])

                # 3. Fetch the previous snapshot for this investment
                previous = conn.execute("""
                    SELECT price
                    FROM price_history
                    WHERE investment_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (investment_id,)).fetchone()

                previous_price = previous[0] if previous else None

                # 4. Calculate percentage changes
                change_since_previous = None
                if previous_price is not None and previous_price > 0:
                    change_since_previous = (
                        (current_price - previous_price)
                        / previous_price
                    ) * 100

                return_since_entry = (
                    (current_price - entry_price) / entry_price
                ) * 100

                review_required = int(return_since_entry <= -30)

                # 5. Insert a new snapshot, preserving older history
                conn.execute("""
                    INSERT INTO price_history (
                        investment_id,
                        recorded_at,
                        price,
                        change_since_previous,
                        return_since_entry,
                        review_required
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    investment_id,
                    datetime.now(timezone.utc).isoformat(),
                    current_price,
                    change_since_previous,
                    return_since_entry,
                    review_required
                ))

                conn.commit()

                print(
                    f"{ticker}: {current_price:.2f} | "
                    f"Previous change: "
                    f"{change_since_previous if change_since_previous is not None else 'N/A'} | "
                    f"Entry return: {return_since_entry:.2f}% | "
                    f"Review: {bool(review_required)}"
                )

            except Exception as error:
                print(f"Failed to update {ticker}: {error}")

    finally:
        conn.close()


if __name__ == "__main__":
    update_stock_prices()
