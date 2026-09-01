from typing import Protocol, runtime_checkable
from pathlib import Path
import json
import os
import logging
import shutil

from qewton.config.saving.schema_keys import SavingKeys
from qewton.config.saving.loading import Deserializer


class Serializer:
    def __init__(self, path: str | Path, replace: bool = False) -> None:
        self.original_path = Path(path)
        if self.original_path.exists() and not replace:
            raise FileExistsError(f"The path {path} already exists. \
                    Use replace=True to allow to overwrite it.")

        self.path = Path(str(self.original_path) + "_temp")
        self.parameter_path = self.path / SavingKeys.FILE_PARAMETERS.value

        self.file_counter = 0
        self.id_dictionary = {}
        self._id_idx_mapping = {}

    def add_objects(self, parent_obj, objects: list) -> list[int]:
        index_mapping = []

        if id(parent_obj) not in self.id_dictionary:
            self.id_dictionary[id(parent_obj)] = parent_obj
            self._id_idx_mapping[id(parent_obj)] = len(self._id_idx_mapping)

        for obj in objects:
            obj_id = id(obj)
            # Object was already added, so we just append its index to the mapping
            if obj_id in self.id_dictionary:
                index_mapping.append(self._id_idx_mapping[obj_id])
                continue

            # Add the object to the dictionary and assign it an index
            self.id_dictionary[obj_id] = obj
            self._id_idx_mapping[obj_id] = len(self._id_idx_mapping)
            index_mapping.append(self._id_idx_mapping[obj_id])

            if isinstance(obj, (float, int, str, bool, type(None))):
                continue  # Skip serialization for primitive types
            if isinstance(obj, Serializable):
                obj.save(self)  # Serialize the object using its save method
            elif isinstance(obj, list):
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_LIST,
                    SavingKeys.KEY_VALUES: self.add_objects(obj, obj),
                }
            elif isinstance(obj, tuple):
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_TUPLE,
                    SavingKeys.KEY_VALUES: self.add_objects(obj, list(obj)),
                }
            elif isinstance(obj, set):
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_SET,
                    SavingKeys.KEY_VALUES: self.add_objects(obj, list(obj)),
                }
            elif isinstance(obj, dict):
                serialize_dict = {}
                for key, value in obj.items():
                    k_serialized = self.add_objects(obj, [key])
                    v_serialized = self.add_objects(obj, [value])
                    serialize_dict[k_serialized] = v_serialized
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_DICT,
                    SavingKeys.KEY_VALUES: serialize_dict,
                }
            else:
                raise TypeError(f"Object of type {type(obj)} is not serializable.")

        return index_mapping

    def set_serialization_data(self, obj_id: int, data: dict) -> None:
        if not obj_id in self.id_dictionary:
            raise ValueError(f"Object with id {obj_id} not found in the serializer.")
        self.id_dictionary[obj_id] = data

    def save(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Saving to %s", self.path)

        # Save all the serialized data to a JSON file in the temporary directory
        saving_dict = {}
        for k, v in self.id_dictionary.items():
            saving_dict[self._id_idx_mapping[k]] = v

        with open(self.path / SavingKeys.FILE_DATA.value, "w", encoding="utf-8") as f:
            f.write(json.dumps(saving_dict, indent=4))

        # Save some general configuration data to a separate JSON file in the
        # temporary directory
        config_data = {
            SavingKeys.VERSION.value: SavingKeys.KEY_VERSION.value,
        }
        with open(self.path / SavingKeys.FILE_CONFIG.value, "w", encoding="utf-8") as f:
            f.write(json.dumps(config_data, indent=4))

        # Move the temporary directory to the original path,
        # replacing it if it exists
        if Path(self.original_path).exists():
            shutil.rmtree(self.original_path)
        # rename the temporary save directory to the original path
        os.rename(self.path, self.original_path)

        logger.info("Saving completed")


@runtime_checkable
class Serializable(Protocol):

    def save(self, serializer: Serializer) -> None: ...

    def ready_to_load(self, serializer: Deserializer, data_config: dict) -> bool:
        return False

    def load(self, serializer: Deserializer, data_config: dict) -> None: ...
