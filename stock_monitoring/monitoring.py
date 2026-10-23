import sqlite3
import yfinance as yf
from datetime import datetime, timezone

DB_PATH = "investments.db"
DROP_THRESHOLD = -30.0  # percentage

# ---------- Database creation ----------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS monitor_investments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            original_verdict TEXT DEFAULT 'BUY'
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investment_id INTEGER NOT NULL,
            recorded_at TEXT NOT NULL,
            price REAL NOT NULL,
            change_since_previous REAL,
            return_since_entry REAL,
            review_required INTEGER DEFAULT 0,
            FOREIGN KEY (investment_id)
                REFERENCES monitor_investments(id)
        );

        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investment_id INTEGER NOT NULL,
            analyzed_at TEXT NOT NULL,
            market_price REAL,
            sentiment_label TEXT,
            sentiment_score REAL,
            predicted_volatility REAL,
            technical_signal TEXT,
            combined_verdict TEXT,
            FOREIGN KEY (investment_id)
                REFERENCES monitor_investments(id)
        );

        CREATE TABLE IF NOT EXISTS evaluation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investment_id INTEGER NOT NULL,
            evaluated_at TEXT NOT NULL,
            entry_price REAL,
            evaluation_price REAL,
            actual_return_pct REAL,
            original_verdict TEXT,
            notes TEXT,
            FOREIGN KEY (investment_id)
                REFERENCES monitor_investments(id)
        );
        """)


# ---------- Get the latest stock price ----------

def get_latest_price(ticker):
    history = yf.Ticker(ticker).history(period="5d")

    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No price data available for {ticker}")

    closes = history["Close"].dropna()

    if closes.empty:
        raise ValueError(f"No valid closing price for {ticker}")

    return float(closes.iloc[-1])


# ---------- Add a human-approved BUY investment ----------

def add_investment(ticker, entry_price=None):
    ticker = ticker.strip().upper()

    # If no entry price is provided, use the latest available close.
    if entry_price is None:
        entry_price = get_latest_price(ticker)

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO monitor_investments
            (ticker, entry_price, entry_date, status, original_verdict)
            VALUES (?, ?, ?, 'active', 'BUY')
        """, (
            ticker,
            entry_price,
            datetime.now(timezone.utc).isoformat()
        ))

        investment_id = cursor.lastrowid

    print(f"Investment added: {ticker}, ID: {investment_id}")
    return investment_id


# ---------- Monitor active investments ----------

def monitor_investments():
    with get_connection() as conn:
        investments = conn.execute("""
            SELECT id, ticker, entry_price
            FROM investments
            WHERE UPPER(decision) = 'BUY'
        """).fetchall()

    for investment_id, ticker, entry_price in investments:
        try:
            current_price = get_latest_price(ticker)
            recorded_at = datetime.now(timezone.utc).isoformat()

            with get_connection() as conn:
                previous = conn.execute("""
                    SELECT price
                    FROM price_history
                    WHERE investment_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (investment_id,)).fetchone()

                # Change since the previous monitoring snapshot
                previous_change = None
                if previous and previous[0] > 0:
                    previous_change = ((current_price - previous[0]) / previous[0]) * 100

                # Return relative to original entry price
                entry_return = ((current_price - entry_price) / entry_price) * 100

                # Flag for human review if price falls 30% from entry
                review_required = int(
                    entry_return <= DROP_THRESHOLD
                )

                conn.execute("""
                    INSERT INTO price_history (
                        investment_id, recorded_at, price,
                        change_since_previous, return_since_entry,
                        review_required
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    investment_id,
                    recorded_at,
                    current_price,
                    previous_change,
                    entry_return,
                    review_required
                ))

                conn.commit()

            print(
                f"{ticker}: Price={current_price:.2f}, "
                f"Since entry={entry_return:.2f}%, "
                f"Since previous check="
                f"{previous_change if previous_change is not None else 'N/A'}, "
                f"Review required={bool(review_required)}"
            )

            if review_required:
                print(
                    f"REVIEW ALERT: {ticker} has fallen "
                    f"at least 30% from its entry price."
                )

        except Exception as error:
            print(f"Monitoring failed for {ticker}: {error}")


# ---------- Save a new model analysis ----------

def save_analysis(
    investment_id,
    market_price,
    sentiment_label,
    sentiment_score,
    predicted_volatility,
    technical_signal,
    combined_verdict
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO analysis_history (
                investment_id, analyzed_at, market_price,
                sentiment_label, sentiment_score,
                predicted_volatility, technical_signal,
                combined_verdict
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            investment_id,
            datetime.now(timezone.utc).isoformat(),
            market_price,
            sentiment_label,
            sentiment_score,
            predicted_volatility,
            technical_signal,
            combined_verdict
        ))


if __name__ == "__main__":
    # create_tables()

    # Uncomment to register a BUY-approved investment:
    # add_investment("RELIANCE.NS")

    # Run the monitoring check:
    # monitor_investments()

    conn = sqlite3.connect("investments.db")
    # '''conn.execute("""DROP TABLE price_history;""")
    # conn.execute("""
    #         CREATE TABLE price_history (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             investment_id INTEGER NOT NULL,
    #             recorded_at TEXT NOT NULL,
    #             price REAL NOT NULL,
    #             change_since_previous REAL,
    #             return_since_entry REAL,
    #             review_required INTEGER DEFAULT 0,
    #             FOREIGN KEY (investment_id) REFERENCES investments(id)
    #         );
    # """)'''

    result = conn.execute(
        """
            select sql
            from sqlite_master
            where type = 'table'
            and name = 'evalutation_history'
        """
    )

    rows = result.fetchall()

    for row in rows:
        print(row)
