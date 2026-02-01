import json
from datetime import datetime


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles additional data types, like datetime.
    """
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)
