import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")


list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/settings/__init__.py",
    "src/settings/config.py",
    "src/core/__init__.py",
    "src/core/pdf_processor.py",
    "src/core/memory.py",
    "src/core/vector_store.py",
    "src/exceptions/__init__.py",
    "src/exceptions/custom_exceptions.py",
    "src/agents/__init__.py",
    "src/agents/comparision_agent.py",
    "src/agents/chart_agent.py",
    "src/agents/orchestrator_agent.py",
    "src/agents/summary_agent.py",
    "src/agents/state.py",
    "src/tools/__init__.py",
    "src/tools/stock_price.py",
    "src/mcp_server/__init__.py",
    "src/mcp_server/server.py",
    "src/logger/__init__.py",
    "src/logger/custom_logger.py",
    "backend/__init__.py",
    "backend/app.py",
    "frontend/__init__.py",
    "frontend/app.py",
    "requirements.txt",
    ".env",]


for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")