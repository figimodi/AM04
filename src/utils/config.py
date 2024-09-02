from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union

    
class DatasetConfig(BaseModel):
    defects_folder: Optional[Path] = None
    defects_masks_folder: Optional[Path] = None
    synthetized_defects_folder: Optional[Path] = None
    synthetized_no_defects_folder: Optional[Path] = None
    splits: List[float]

class ModelConfig(BaseModel):
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: str
    only_test: bool
    save_images: Optional[Path] = None
    pretrained: Optional[Path] = None
    annotations: Optional[Path] = None

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
