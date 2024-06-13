from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional

    
class DatasetConfig(BaseModel):
    defects_folder: Path
    splits: List[float]

class ModelConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: str
    only_test: bool
    pretrained: Optional[Path] = None

class LoggerConfig(BaseModel):
    log_dir: Path
    experiment_name: str
    version: int

class CheckpointConfig(BaseModel):
    monitor: str
    save_top_k: int
    mode: str

class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    logger: LoggerConfig
    checkpoint: CheckpointConfig
