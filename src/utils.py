from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from model.technical.technical import technical_init, summary
from model.sentiment.sentiment import sentiment_init, Sentiment_result
from config.settings import VOLATILITY_MODEL, DB_PATH
from stock_monitoring.evaluation import save_model_evaluation
from stock_monitoring.monitoring import monitor_investments, get_connection
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import sqlite3
import json
import pickle

with open(VOLATILITY_MODEL, "rb") as file:
     model = pickle.load(file)

class MarketState(TypedDict):
    ticker: str
    verdict: summary
    volatility: int
    sentiment: Sentiment_result
    human_decsion: Literal["Hold", "Buy", "Sold"]

class EvaluationState(TypedDict, total=False):
    status: str
    evaluation_count: int

class Parent(TypedDict):
    task: Literal["analyzed", "evaluate"]
    ticker: str
    status: str
    evaluation_count: int

#Analysis Graph model
def human_decision_node(state: MarketState):
    try:
        ticker = state["ticker"].upper()
        decision = input("Enter your decision in (Hold, Buy, Sell) ").strip().upper()

        if decision not in ("HOLD", "BUY", "SELL"):
            raise ValueError("choose from the give option")

        if decision == "BUY":
            history = yf.Ticker(ticker).history(period="5d")

            if history.empty or history["Close"].dropna().empty:
                raise ValueError(f"No stock price data available for {ticker}")

            price = float(history["Close"].dropna().iloc[-1])
            reason = input("Enter your reason to BUY: ")

            conn = sqlite3.connect(DB_PATH)

            conn.execute("""
                INSERT INTO investments
                (ticker, decision, entry_price, entry_date, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticker,
                decision,
                price,
                datetime.now().isoformat(),
                reason
            ))

            conn.commit()
            conn.close()
            
        return {"decision": decision}

    except ValueError as e:
        print(e)

    except ConnectionError as e:
        print(f"cannot connect to the database")

def save_model_analysis(conn, state: MarketState, investment_id, decision):
    technical_data = state["verdict"]
    sentiment_data = state["sentiment"]

    conn.execute(
        """Insert into analysis_history(
            investment_id,
            ticker,
            analysed_at,
            sentiment,
            sentiment_label,
            sentiment_score,
            predicted_volatility,
            technical_analysis,
            technical_signal,
            combined_verdict,
            human_decision
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,(
            investment_id,
            state["ticker"],
            datetime.now(timezone.utc).isoformat(),

            json.dumps(
                sentiment_data.model_dump()
                if hasattr(sentiment_data, "model_dump")
                else sentiment_data
            ),

            0,

            json.dumps(
                technical_data.model_dump()
                if hasattr(technical_data, "model_dump")
                else technical_data
            ),

            json.dumps(
                technical_data.model_dump()
                if hasattr(technical_data, "model_dump")
                else technical_data
            ),

            decision
        )
    )

def collection(state: MarketState):
    try:    
        ticker = input("Enter the Stock")
        if ticker == None:
            raise ValueError("The stock is not present")
        
        results = yf.Search(ticker, max_results=10)
        for stocks in results.quotes:
            stock = stocks["symbol"]
            break
        return {"ticker": stock}
    except Exception as e:
        print(e.message)

def technode(state: MarketState):
    ticker = state["ticker"]
    result = technical_init(ticker)
    return {"verdict": result}

def sentimentnode(state: MarketState):
    ticker = state["ticker"]
    result = sentiment_init(ticker)
    return {"sentiment": result}

def output(state: MarketState):
    ans = state["verdict"].Position
    print(ans)
    return {"result": ans}

#Analysis Graph
graph = StateGraph(MarketState)

graph.add_node("tech", technode)
graph.add_node("sentiment", sentimentnode)
graph.add_node("collect", collection)
graph.add_node("output", output)
graph.add_node("human_decision", human_decision_node)

graph.add_edge(START, "collect")
graph.add_edge("collect", "sentiment")
graph.add_edge("collect", "tech")

graph.add_edge("tech", "output")
graph.add_edge("sentiment", "output")
graph.add_edge("output", "human_decision")
graph.add_edge("human_decision", END)

workflow1 = graph.compile()

#Evaluataion Graph Model
def monitoring(state: EvaluationState):
    monitor_investments()
    return {"status": "monitoring_completed"}

def evaluation_nod(state: EvaluationState):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                m.id,
                m.ticker,
                m.entry_price,
                p.price,
                a.predicted_volatility,
                a.technical_signal
            FROM investments AS m

            JOIN (
                SELECT investment_id, MAX(id) AS latest_id
                FROM price_history
                GROUP BY investment_id
            ) AS latest
                ON latest.investment_id = m.id

            JOIN price_history AS p
                ON p.id = latest.latest_id

            LEFT JOIN analysis_history AS a
                ON a.id = (
                    SELECT MAX(a2.id)
                    FROM analysis_history AS a2
                    WHERE a2.investment_id = m.id
                )
        """).fetchall()

    for ticker, investment_id, entry_price, evaluation_price, predicted_volatility, predicted_direction in rows:

        print(predicted_volatility, predicted_direction)

        save_model_evaluation(
            ticker=ticker,
            investment_id=investment_id,
            entry_price=entry_price,
            evaluation_price=evaluation_price,
            predicted_volatility=predicted_volatility,
            actual_volatility=None,
            predicted_direction=predicted_direction,
            actual_direction=None
        )

    return {"evaluation_count": len(rows)}

#Evaluation Graph
graph2 = StateGraph(EvaluationState)

graph2.add_node("monitor", monitoring)
graph2.add_node("evaluation", evaluation_nod)

graph2.add_edge(START, "monitor")
graph2.add_edge("monitor", "evaluation")
graph2.add_edge("evaluation", END)

workflow2 = graph2.compile()

#Parent Graph Model
def run_analysis(state: Parent):
    result = workflow1.invoke({})
    return result

def run_evalute(state: Parent):

    result = workflow2.invoke({
        "status": "started", 
        "evaluation_count": 0
    })

    return{
        "status": result.get("status", "monitoring_completed"),
        "evaluation_count": result.get("evaluation_count", 0)
    }

def task_route(state: Parent):
    return state["task"]

#Parent Graph
parent = StateGraph(Parent)
parent.add_node("analysis", run_analysis)
parent.add_node("evaluated", run_evalute)

parent.add_conditional_edges(
    START, 
    task_route,
    {
        "analyzed": "analysis",
        "evaluate": "evaluated",
    },
)

parent.add_edge("analysis", END)
parent.add_edge("evaluated", END)

workflow = parent.compile()

def finalflow():
    result = workflow.invoke({"task": "evaluate"})
    print(result)