from huggingface_hub import InferenceClient
from config.settings import SERAPI, OPENAI_API, OPENAI_BASE, HF_TOKEN
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from transformers import pipeline
import trafilatura
import serpapi

class Sentiment_result(BaseModel):
    positive: int
    negativ: int
    neutral: int
    overall: str

def summarise(text: str) -> str:
    prompt = f"""
        You are a financial news summarization assistant.

        Summarize the following news article in approximately 50 words.

        Focus only on:
        - The main event or development
        - The company, stock, or market involved
        - Important financial or business impact
        - Any significant positive or negative development

        Do not add information that is not present in the article.
        Do not give investment advice.
        Return only the summary.

        ARTICLE:
        {text}
        """
    
    client = ChatOpenAI(
        model = "openrouter/free",
        openai_api_base = OPENAI_BASE,
        openai_api_key = OPENAI_API,
        timeout=20,
        max_retries=2
    )   

    try:
        response = client.invoke(prompt)
        return response.content

    except ConnectionError as e:
        print(f"Failed to connect with the Openrouter model: {e}")
        return ""

    except TimeoutError as e:
        print(f"Response Timeout error from the. Try Again")
        return ""

    except Exception as e:
        print(f"The model is raising an issue: {e}")
        return ""


def news_extracter(name: str) -> list:

    try:
        sclient = serpapi.Client(api_key = SERAPI)

        news = sclient.search({
            "engine": "google_news",
            "q": name,
            "hl": "en",
            "gl": "in",
            "num": 3
        })

        news_results = news.get("news_results", [])

        content = []

        for news in news_results:
            link = news.get("link", [])

            if not link:
                continue

            download = trafilatura.fetch_url(link)
            context = trafilatura.extract(download)

            content.append(context)

        return content
    
    except ValueError as e:
        print(f"Invalid value format {e}")
        return []

    except Exception as e:
        print(f"news_extraction failed {e}")
        return []

    except Exception as e:
        print(f"Article failed to be extracted {e}")
        return []

def senitment_analyser(content: list):
    try:
        Client = InferenceClient(
            provider="hf-inference",
            api_key = HF_TOKEN
        )
        pipe = pipeline("text-classification", model="9mark9/finbert-minilm-sentiment")

        if not content:
            raise ValueError("The value should be a string")

        positive = 0
        negative = 0
        neutral = 0

        i = 0
        for news in content[0:3]:
            print(f"summarising news {i}")
            i += 1
            summary = summarise(news)
            sentiment = pipe(summary)

            for item in sentiment:
                if item["label"] == "positive":
                    positive += item["score"]
                elif item["label"] == "negative":
                    negative += item["score"]
                else:
                    neutral += item["score"]

        total = positive + negative + neutral
        positive /= total
        negative /= total
        neutral /= total

        Sentiment_result.positive = positive
        Sentiment_result.negative = negative
        Sentiment_result.neutral = neutral

        if positive > negative and positive > neutral:
            Sentiment_result.overall = "positive"
        elif negative > neutral:
            Sentiment_result.overall = "negative"
        else:
            Sentiment_result.overall = "neutral"

    except ValueError as e:
        print(f"Invalid Input: {e}") 
        return None

    except Exception as e:
        print(f"Sentiment Aanlyser failed {e}")
        return None

def sentiment_init(company: str) -> str:

    print(f"Searching news for {company}... \n")
    news = news_extracter(company)
    print(f"Finding the final sentiment \n")
    senitment_analyser(news) 
    print(Sentiment_result.positive)
    print(Sentiment_result.negative)
    print(Sentiment_result.neutral)
    print(Sentiment_result.overall)

if __name__ == "__main__":
    company = "RELAINCE"
    sentiment = sentiment_init(company)
    print(sentiment)
