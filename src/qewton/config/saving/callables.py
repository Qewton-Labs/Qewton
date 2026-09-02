from pathlib import Path
from typing import Any

from qewton.config.saving.saving import Serializable, Serializer
from qewton.config.saving.loading import Deserializer


def save(obj: Serializable, path: str | Path, replace: bool = False) -> None:
    """Saves a Serializable object to the specified path.

    Args:
        obj (Serializable): The object to be saved.
        path (str | Path): The path where the object will be saved.
        replace (bool, optional): If True, replaces existing files. Defaults to False.
    """
    serializer = Serializer(path, replace)
    obj.save(serializer)
    serializer.save()


def load(path: str | Path) -> Any:
    """Loads a Serializable object from the specified path.

    Args:
        path (str | Path): The path from which the object will be loaded.

    Returns:
        Serializable: The loaded object.
    """
    deserializer = Deserializer(path)
    deserializer.load()
    return deserializer.id_to_obj[0]  # Assuming the root object has ID 0
