from typing import Protocol, runtime_checkable
from pathlib import Path

from qewton.config.saving.schema_keys import SavingKeys


class Serializer:
    def __init__(self, path: Path) -> None:
        self.path = path
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

    def set_serialization_data(self, id: int, data: dict) -> None:
        if not id in self.id_dictionary:
            raise ValueError(f"Object with id {id} not found in the serializer.")
        self.id_dictionary[id] = data


class Deserializer:
    def __init__(self) -> None:
        super().__init__()
        self._data = {}

    def set_data(self, data: dict) -> None:
        self._data = data

    def get_data(self) -> dict:
        return self._data


@runtime_checkable
class Serializable(Protocol):

    def save(self, serializer: Serializer) -> None: ...

    @classmethod
    def load(cls, serializer: Deserializer) -> None: ...
