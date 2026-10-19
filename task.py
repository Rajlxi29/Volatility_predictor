import schedule
import time
from stock_monitoring.monitoring import monitor_stocks

# Run every 2 weeks (14 days)
schedule.every(14).days.do(monitor_stocks)

while True:
    schedule.run_pending()
    time.sleep(60)