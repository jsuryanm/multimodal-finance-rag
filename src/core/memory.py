from pathlib import Path 
import json

from langchain_core.messages import messages_to_dict,messages_from_dict
from src.settings.config import settings 

class MemoryManager:

    def __init__(self,session_id: str):
        self.path = settings.DATA_DIR / session_id / "chat.json"
    
    def load(self):
        if not self.path.exists():
            return []
        
        with open(self.path) as f:
            data = json.load(f)

        return messages_from_dict(data)
    
    def save(self,messages):
        self.path.parent.mkdir(exist_ok=True,parents=True)
        
        with open(self.path,"w") as f:
            json.dump(messages_to_dict(messages),
                      f,indent=2)
        