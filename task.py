import schedule
import time
from src.utils import finalflow

# def monitoring():
#     try:
#         result = workflow()
#         print("The final result of the score is", result)
#     except Exception as e:
#         print("The monitoring isn't completed", e)

# monitoring()

# Run every 2 weeks (14 days)
schedule.every(1).minutes.do(finalflow)

while True:
    print("\nNext will be visible here...\n")
    schedule.run_pending()
    time.sleep(60)
