import json
from dataclasses import asdict, is_dataclass
from datetime import datetime


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, 'value'):
            return o.value
        return super().default(o)
