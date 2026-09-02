from typing import Any
from pathlib import Path
import json
import importlib

from qewton.config.saving.schema_keys import SavingKeys, ALLOWED_MODULE_PREFIXES


class Deserializer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.parameter_path = self.path / SavingKeys.FILE_PARAMETERS

        assert self.path.exists(), f"The path {path} does not exist."
        assert (
            self.path / SavingKeys.FILE_DATA
        ).exists(), f"The data file {SavingKeys.FILE_DATA} \
            does not exist in the path {path}."
        config_path = self.path / SavingKeys.FILE_CONFIG
        assert config_path.exists(), f"The config file {SavingKeys.FILE_CONFIG} \
            does not exist in the path {path}."

        # Check for version mismatch
        with config_path.open("r", encoding="utf-8") as f:
            config_data: dict = json.load(f)
        assert config_data.get(SavingKeys.KEY_VERSION) != SavingKeys.VERSION, (
            f"Schema version mismatch: expected {SavingKeys.VERSION}, "
            f"found {config_data.get(SavingKeys.KEY_VERSION)}"
        )

        self.obj_finished_loading: dict[int, bool] = {}
        self.ready_to_reference: dict[int, bool] = {}
        self.id_to_obj: dict[int, Any] = {}

        from qewton.backends import BACKEND_DICT

        self.backend_dict = BACKEND_DICT

    def load(self):
        from qewton.config.saving.saving import Serializable

        with open(self.path / SavingKeys.FILE_DATA, "r", encoding="utf-8") as f:
            data: dict = json.load(f)

        # First pass: create all objects without setting their attributes
        for obj_id, obj_data in data.items():
            obj_id = int(obj_id)
            if isinstance(obj_data, dict):
                self.obj_finished_loading[obj_id] = False
                self.id_to_obj[obj_id] = self._create_object(obj_data, obj_id)
            else:
                self.obj_finished_loading[obj_id] = True
                self.ready_to_reference[obj_id] = True
                self.id_to_obj[obj_id] = obj_data  # For primitive types

        # Now create a queue of objects that are not finished loading
        queue = [
            obj_id
            for obj_id, finished in self.obj_finished_loading.items()
            if not finished
        ]
        while len(queue) > 0:
            remaining_queue = []
            for obj_id in queue:
                obj_data = data[str(obj_id)]
                obj = self.id_to_obj[obj_id]
                if isinstance(obj, Serializable):
                    if obj.ready_to_load(self, obj_data):
                        obj.load(self, obj_data)
                        self.obj_finished_loading[obj_id] = True
                elif isinstance(obj, list):
                    ids = obj_data[SavingKeys.KEY_VALUES]
                    if self._check_dependence_ready(ids):
                        obj.extend(self.id_to_obj[v_id] for v_id in ids)
                        self.obj_finished_loading[obj_id] = True
                elif isinstance(obj, tuple):
                    ids = obj_data[SavingKeys.KEY_VALUES]
                    if self._check_dependence_ready(ids):
                        self.id_to_obj[obj_id] = tuple(
                            self.id_to_obj[v_id] for v_id in ids
                        )
                        self.obj_finished_loading[obj_id] = True
                        self.ready_to_reference[obj_id] = True
                elif isinstance(obj, set):
                    ids = obj_data[SavingKeys.KEY_VALUES]
                    if self._check_dependence_ready(ids):
                        for v_id in ids:
                            obj.add(self.id_to_obj[v_id])
                        self.obj_finished_loading[obj_id] = True
                elif isinstance(obj, dict):
                    serialize_dict: dict = obj_data[SavingKeys.KEY_VALUES]
                    keys_ready = self._check_dependence_ready(
                        [int(k) for k in serialize_dict.keys()]
                    )
                    values_ready = self._check_dependence_ready(
                        [int(v) for v in serialize_dict.values()]
                    )
                    if keys_ready and values_ready:
                        for k_serialized, v_serialized in serialize_dict.items():
                            k_id = int(k_serialized)
                            v_id = int(v_serialized)
                            obj[self.id_to_obj[k_id]] = self.id_to_obj[v_id]
                        self.obj_finished_loading[obj_id] = True

                # If we could not finish loading this object, add it back
                # to the queue
                if not self.obj_finished_loading[obj_id]:
                    remaining_queue.append(obj_id)

            if len(remaining_queue) == len(queue):
                raise RuntimeError(
                    "Circular dependency detected or missing dependencies. "
                    "Some objects could not be fully loaded. Items remaining "
                    f"in the queue: {remaining_queue}"
                )
            queue = remaining_queue

    def _check_dependence_ready(self, obj_ids: list[int]) -> bool:
        return all(self.ready_to_reference.get(v_id, False) for v_id in obj_ids)

    def _create_object(self, obj_data: dict, obj_id: int) -> Any:
        assert (
            SavingKeys.KEY_TYPE in obj_data
        ), "Serialized object data must contain a 'type' key."
        # Internal method to create an object based on its serialized data
        obj_type = obj_data[SavingKeys.KEY_TYPE]
        self.ready_to_reference[obj_id] = True

        # Check what type of object we are dealing with and create it accordingly
        if obj_type == SavingKeys.KEY_CLASS_OBJ:
            cls = self._load_module_and_class(
                obj_data[SavingKeys.KEY_MODULE], obj_data[SavingKeys.KEY_CLASS]
            )
            self.obj_finished_loading[obj_id] = True
            return cls
        if obj_type == SavingKeys.KEY_SERIALIZABLE:
            cls = self._load_module_and_class(
                obj_data[SavingKeys.KEY_MODULE], obj_data[SavingKeys.KEY_CLASS]
            )
            # Create instance without calling __init__
            return cls.construct_new_object(self, obj_data)

        if obj_type == SavingKeys.KEY_LIST:
            return []
        if obj_type == SavingKeys.KEY_TUPLE:
            self.ready_to_reference[obj_id] = False
            return ()
        if obj_type == SavingKeys.KEY_SET:
            return set()
        if obj_type == SavingKeys.KEY_DICT:
            return {}
        if obj_type == SavingKeys.KEY_ENUM:
            cls = self._load_module_and_class(
                obj_data[SavingKeys.KEY_MODULE], obj_data[SavingKeys.KEY_CLASS]
            )
            self.obj_finished_loading[obj_id] = True
            return cls(obj_data[SavingKeys.KEY_VALUES])
        if obj_type == SavingKeys.KEY_CLASS_OBJ:
            cls = self._load_module_and_class(
                obj_data[SavingKeys.KEY_MODULE], obj_data[SavingKeys.KEY_CLASS]
            )
            return cls
        if obj_type == SavingKeys.KEY_BACKEND_PARAMETER:
            backend_class = self.backend_dict[obj_data[SavingKeys.KEY_CLASS]]
            param_path = self.parameter_path / obj_data[SavingKeys.KEY_VALUES]
            self.obj_finished_loading[obj_id] = True
            return backend_class.load_data(param_path)  # type: ignore
        raise ValueError(f"Unknown object type: {obj_type}")

    def _load_module_and_class(self, module_name: str, class_name: str) -> Any:
        # Check if the module name starts with an allowed prefix
        if not any(module_name.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
            raise ValueError(
                f"Module {module_name} is not allowed. "
                f"Allowed prefixes are: {ALLOWED_MODULE_PREFIXES}, add them to \
                the _ALLOWED_MODULE_PREFIXES list in saving.py if you \
                want to allow this module."
            )

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
