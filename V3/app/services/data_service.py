# app/services/data_service.py
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()

class DataService:
    @staticmethod
    def load_etrm_data(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load ETRM data from Excel file with better error handling"""
        try:
            # Try both environment variable and hardcoded path
            path = file_path or os.getenv("ETRM_DATA_PATH") or r"C:\Users\RachanaBaldania\OneDrive - RandomTrees\Rachana_Code\ETRM_AI_Matching_RB\V3\data\ETRM_Data.xlsx"
            
            if not path:
                raise ValueError("No ETRM data path provided")
                
            path = Path(path).absolute()
            print(f"Loading ETRM data from: {path}")
            
            if not path.exists():
                raise FileNotFoundError(f"ETRM file not found at {path}")
                
            df = pd.read_excel(path)
            
            # Clean the DataFrame - replace NaN with None and convert numpy types
            df = df.where(pd.notnull(df), None)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            print("ETRM data loaded successfully")
            return df
            
        except Exception as e:
            print(f"CRITICAL ERROR loading ETRM data: {str(e)}")
            return None