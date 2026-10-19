import sqlite3
from datetime import datetime, timezone

def save_model_evaluation(
    investment_id,
    entry_price,
    evaluation_price,
    predicted_volatility=None,
    actual_volatility=None,
    predicted_direction=None,
    actual_direction=None,
    sentiment_score=None,
    evaluation_period_days=14
):
    actual_return = (
        (evaluation_price - entry_price) / entry_price
    ) * 100

    # Directional correctness: 1 = correct, 0 = incorrect
    technical_correct = None

    if predicted_direction and actual_direction:
        technical_correct = int(
            predicted_direction.upper()
            == actual_direction.upper()
        )

    # Absolute error between predicted and actual volatility
    volatility_mae = None

    if (
        predicted_volatility is not None
        and actual_volatility is not None
    ):
        volatility_mae = abs(
            predicted_volatility - actual_volatility
        )

    with sqlite3.connect("investments.db") as conn:
        conn.execute("""
            INSERT INTO evaluation_history (
                investment_id,
                evaluated_at,
                entry_price,
                evaluation_price,
                actual_return_pct,
                original_verdict,
                notes,
                sentiment_score,
                volatility_mae,
                technical_correct,
                evaluation_period_days
            )
            SELECT ?, ?, ?, ?, ?, original_verdict, ?, ?, ?, ?, ?
            FROM monitor_investments
            WHERE id = ?
        """, (
            investment_id,
            datetime.now(timezone.utc).isoformat(),
            entry_price,
            evaluation_price,
            actual_return,
            "Model evaluation completed",
            sentiment_score,
            volatility_mae,
            technical_correct,
            evaluation_period_days,
            investment_id
        ))

    print(f"Actual return: {actual_return:.2f}%")
    print(f"Volatility MAE: {volatility_mae}")
    print(f"Technical signal correct: {technical_correct}")
