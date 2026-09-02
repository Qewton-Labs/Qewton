from enum import Enum
from typing import Any, Protocol, runtime_checkable
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
        self.parameter_path = self.path / SavingKeys.FILE_PARAMETERS

        # create the temporary directory for saving
        if not self.path.exists():
            os.makedirs(self.path)
        if not self.parameter_path.exists():
            os.makedirs(self.parameter_path)

        self.file_counter = 0
        self.id_dictionary = {}
        self._id_idx_mapping = {}

        from qewton.backends import BACKEND_DICT

        self.backend_dict = BACKEND_DICT

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
            if isinstance(obj, type):
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_CLASS_OBJ,
                    SavingKeys.KEY_CLASS: obj.__name__,
                    SavingKeys.KEY_MODULE: obj.__module__,
                }
            elif isinstance(obj, Serializable):
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
                    k_serialized = self.add_objects(obj, [key])[0]
                    v_serialized = self.add_objects(obj, [value])[0]
                    serialize_dict[k_serialized] = v_serialized
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_DICT,
                    SavingKeys.KEY_VALUES: serialize_dict,
                }
            elif isinstance(obj, Enum):
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_ENUM,
                    SavingKeys.KEY_CLASS: obj.__class__.__name__,
                    SavingKeys.KEY_MODULE: obj.__class__.__module__,
                    SavingKeys.KEY_VALUES: obj.value,
                }
            elif any(
                isinstance(obj, cls.default_dtype) for cls in self.backend_dict.values()
            ):
                # If the object is a backend tensor, save it as a parameter file
                param_extenstion = f"parameter_{self.file_counter}"
                param_path = self.parameter_path / param_extenstion
                self.file_counter += 1
                # Save the tensor using the appropriate backend
                backend_key = None
                for backend_key, backend_cls in self.backend_dict.items():
                    if isinstance(obj, backend_cls.default_dtype):
                        backend_cls.save_data(obj, path=param_path)  # type: ignore
                        break
                self.id_dictionary[obj_id] = {
                    SavingKeys.KEY_TYPE: SavingKeys.KEY_BACKEND_PARAMETER,
                    SavingKeys.KEY_CLASS: backend_key,
                    SavingKeys.KEY_VALUES: param_extenstion,
                }

            if obj_id not in self.id_dictionary:
                raise TypeError(f"Object of type {type(obj)} is not serializable. \
                    This object belongs to {type(parent_obj)}.")

        return index_mapping

    def set_serialization_data(self, obj_id: int, data: dict) -> None:
        self.id_dictionary[obj_id] = data

    def save(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Saving to %s", self.path)

        # Save all the serialized data to a JSON file in the temporary directory
        saving_dict = {}
        for k, v in self.id_dictionary.items():
            saving_dict[self._id_idx_mapping[k]] = v

        def find_bad_values(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_bad_values(v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    find_bad_values(v)
            elif not isinstance(obj, (str, int, float, bool, type(None))):
                print(f"Non-serializable: {type(obj)} -> {obj!r}")

        find_bad_values(saving_dict)

        with open(self.path / SavingKeys.FILE_DATA, "w", encoding="utf-8") as f:
            f.write(json.dumps(saving_dict, indent=4))

        # Save some general configuration data to a separate JSON file in the
        # temporary directory
        config_data = {
            SavingKeys.VERSION: SavingKeys.KEY_VERSION,
        }
        with open(self.path / SavingKeys.FILE_CONFIG, "w", encoding="utf-8") as f:
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

    def save(self, serializer: Serializer) -> None:
        self_args = []
        self_keys = []
        for k, v in self.__dict__.items():
            self_args.append(v)
            self_keys.append(k)
        idx_mapping = serializer.add_objects(self, self_args)
        # TODO: Build one big dictionary with class names for qewton
        # modules, so file changes dont break saving
        node_config = {
            SavingKeys.KEY_TYPE: SavingKeys.KEY_SERIALIZABLE,
            SavingKeys.KEY_CLASS: self.__class__.__name__,
            SavingKeys.KEY_MODULE: self.__class__.__module__,
            SavingKeys.KEY_SELF_ARGS: dict(zip(self_keys, idx_mapping)),
        }
        serializer.set_serialization_data(id(self), node_config)

    @classmethod
    def construct_new_object(cls, serializer: Deserializer, data_config: dict) -> Any:
        return cls.__new__(cls)  # Create a new instance without calling __init__

    def ready_to_load(self, serializer: Deserializer, data_config: dict) -> bool:
        if SavingKeys.KEY_SELF_ARGS not in data_config:
            return True  # No attributes to set, so it's ready to load
        return all(
            serializer.ready_to_reference.get(v_id, False)
            for v_id in data_config[SavingKeys.KEY_SELF_ARGS].values()
        )

    def load(self, serializer: Deserializer, data_config: dict) -> None:
        if SavingKeys.KEY_SELF_ARGS not in data_config:
            return  # No attributes to set

        for k, v in data_config[SavingKeys.KEY_SELF_ARGS].items():
            setattr(self, k, serializer.id_to_obj[v])
