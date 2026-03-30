
from pathlib import Path
import numpy as np
from typing import Optional, Union
from dataloaders.utils import TrajectorySlice
import json

LABEL_FILENAME = "labels.json"
IMAGE_SIZE = 256

class ObjectPosLoader:
    def __init__(
        self, 
        object_pose_dim: int = 3,
        pos_labels_path: Optional[Union[str, Path]] = None,
    ):
        self.object_pose_dim = object_pose_dim
        self.pos_labels_path = Path(pos_labels_path)
        self._load_object_pos_data()

    def _load_object_pos_data(self):
        json_path = self.pos_labels_path / LABEL_FILENAME
        with json_path.open("r") as f:
            object_pos_data = json.load(f)
        object_pos = []
        for index, data in object_pos_data.items():
            if self.object_pose_dim == 2:
                object_pos.append(data["object_pos_2d"])
            elif self.object_pose_dim == 3:
                object_pos.append(data["object_pos_3d"])
            else:
                raise ValueError("Invalid object_pose_dim")
        self._object_pos = np.array(object_pos, dtype=np.float32)
        self._len = len(self._object_pos)

    def get_batch(self, indices: Union[np.ndarray, TrajectorySlice]):
        if isinstance(indices, TrajectorySlice):
            indices = np.arange(
                indices.start_index, indices.end_index, indices.skip + 1
            )
        return self._object_pos[indices] # T, D
