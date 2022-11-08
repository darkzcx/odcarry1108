import importlib
import inspect
from typing import Text, Optional, Type


def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Type:

    klass = None
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        klass = getattr(m, class_name, None)
    elif lookup_path:
        # try to import the class from the lookup path
        m = importlib.import_module(lookup_path)
        klass = getattr(m, module_path, None)

    if klass is None:
        raise ImportError(f"Cannot retrieve class from path {module_path}.")

    return klass