from typing import Any

class Registry:
    """
    Registries that can be used across the project.
    
    Usage:
    ```
    registry = Registry('registry')
    
    @registry.register('foo')
    def foo():
        print('hello')
    
    registry.get('foo')()
    # hello
    ```
    """

    _all_registries = {}
    initialized = False

    def __new__(cls, registry_name: str):
        if registry_name not in cls._all_registries:
            cls._all_registries[registry_name] = super().__new__(cls)
        return cls._all_registries[registry_name]

    def __init__(self, registry_name: str):
        self.registry_name = registry_name
        if not self.initialized:
            self._dict = {}
            self.initialized = True

    def register(self, key: str):
        assert key not in self._dict, \
            f'"{key}" is already registered as {self._dict[key]} in Registry <{self.registry_name}>.'

        def wrapper(obj):
            self._dict[key] = obj
            return obj

        return wrapper

    def get(self, key: str, default=None) -> Any:
        return self._dict.get(key, default)
