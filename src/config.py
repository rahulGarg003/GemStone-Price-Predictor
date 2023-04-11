import os
import sys
from dataclasses import dataclass

@dataclass
class Config:
    z_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    z_artifacts_path = os.path.join(z_base_path, 'artifacts')
    
    os.makedirs(z_artifacts_path, exist_ok=True)
