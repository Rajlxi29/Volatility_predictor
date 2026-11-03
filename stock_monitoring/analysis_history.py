import sqlite3
from datetime import datetime, timezone

def update_analysis_history(investment_id, ticker, 
                            market_price, 
                            sentiment_label, 
                            sentiment_score, 
                            predicted_volatility, 
                            technical_signal,
                            combined_verdict
                        ):
    """Store model evaluation scores for ticker 
       consistent scores between 0-100
    """
    try:
        with sqlite3.connect("investments.db") as conn:
            conn.execute(
                """Insert into analysis_history (investment_id, analyzed_at, market_price, sentiment_label, 
                    sentiment_score, predicted_volatility, technical_signal, combined_verdict)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,(
                    investment_id, 
                    datetime.now(timezone.utc).isoformat(),
                    market_price,
                    sentiment_label,
                    sentiment_score,
                    predicted_volatility,
                    technical_signal,
                    combined_verdict
                )
            )

        return True

    except sqlite3.Error as e:
        print(e)
        return False
