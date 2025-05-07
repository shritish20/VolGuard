import re
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_5paisa_date_string(date_string):
    """Parses the /Date(1234567890000+0000)/ format to a timestamp (milliseconds)."""
    if not isinstance(date_string, str):
        return None
    match = re.search(r'/Date\((\d+)[+-]\d+\)/', date_string)
    if match:
        return int(match.group(1))
    return None

def format_timestamp_to_date_str(timestamp_ms):
    """Converts a timestamp (milliseconds) to YYYY-MM-DD string."""
    if timestamp_ms is None:
        return "N/A"
    try:
        timestamp_s = timestamp_ms / 1000
        dt_object = datetime.fromtimestamp(timestamp_s)
        return dt_object.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"
