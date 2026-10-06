from huggingface_hub import InferenceClient
from config.settings import SERAPI, OPENAI_API, OPENAI_BASE, HF_TOKEN
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import trafilatura
import requests
import serpapi

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

    except Exception as e:
        print(f"The model is raising an issue: {e}")
        return ""


def news_extracter(name: str) -> str:

    try:
        sclient = serpapi.Client(api_key = SERAPI)

        news = sclient.search({
            "engine": "google_news",
            "q": name,
            "hl": "en",
            "gl": "in",
            "num": 3
        })

        news_results = news.get("news_result", [])

        content = ""

        for news in news_results:
            link = news.get("links", [])

            if not link:
                continue

            # download = trafilatura.fetch_url(link)
            # context = trafilatura.extract(download)

            context = requests.get(
                link,
                timeout=10,
                headers={ "User-Agent": "Mozilla/5.0" }
            )

            context.raise_for_status()

            soup = BeautifulSoup(context, "html.parser")
            soup.get_text(
                separator=" ",
                strip=True
            )

            content += "\n" + context 

        return content
    
    except ValueError as e:
        print(f"Invalid value format {e}")
        return ""

    except Exception as e:
        print(f"news_extraction failed {e}")
        return ""

def senitment_analyser(content: str) -> str:
    try:
        Client = InferenceClient(
            provider="hf-inference",
            api_key = HF_TOKEN
        )

        if not content or not content.strip():
            raise ValueError("The value should be a string")
        
        sentiment = Client.text_classification(
            content,
            model="ProsusAI/finbert"
        )

        context = {}

        for emotion in sentiment:
            context.update(emotion)

        positive = context.get("positive", 0)
        negative = context.get("negative", 0)

        res = positive - negative

        if res > 0.15:
            return "positive"
        elif res < -0.15:
            return "negative"
        else:
            return "nuetral"

    except ValueError as e:
        print(f"Invalid Input: {e}") 
        return "neutral"

    except Exception as e:
        print(f"Sentiment Aanlyser failed {e}")
        return "neutral"


if __name__ == "__main__":
    company = "RELAINCE"
    news = news_extracter(company)
    summary = summarise(news)
    sentiment = senitment_analyser(summary) 
    print(sentiment)

