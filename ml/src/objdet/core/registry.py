"""Plugin registry pattern for extensible architecture.

This module provides a generic registry system that allows for easy
registration and retrieval of plugins (models, datasets, transforms, etc.)
without tight coupling between components.

Example:
    >>> from objdet.core.registry import Registry
    >>>
    >>> # Create a registry for models
    >>> MODEL_REGISTRY = Registry("models")
    >>>
    >>> # Register a model using decorator
    >>> @MODEL_REGISTRY.register("faster_rcnn")
    ... class FasterRCNN:
    ...     pass
    >>>
    >>> # Or register directly
    >>> MODEL_REGISTRY.register("retinanet")(RetinaNet)
    >>>
    >>> # Retrieve registered class
    >>> model_cls = MODEL_REGISTRY.get("faster_rcnn")
    >>> model = model_cls(num_classes=80)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from objdet.core.exceptions import ConfigurationError
from objdet.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class Registry[T]:
    """Generic registry for plugin-style component management.

    This class provides a centralized registry where components can be
    registered by name and later retrieved. It supports both decorator-style
    and direct registration.

    Args:
        name: Name of this registry (for logging/error messages).

    Attributes:
        name: Registry name.
        _registry: Internal dictionary mapping names to registered items.

    Example:
        >>> registry = Registry[nn.Module]("models")
        >>> registry.register("my_model")(MyModelClass)
        >>> model_cls = registry.get("my_model")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[T]] = {}

    def register(
        self,
        name: str,
        *,
        aliases: list[str] | None = None,
        override: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """Register a class with the given name.

        This method can be used as a decorator or called directly.

        Args:
            name: Primary name to register the class under.
            aliases: Optional list of alternative names.
            override: If True, allow overriding existing registrations.

        Returns:
            Decorator function that registers the class.

        Raises:
            ConfigurationError: If name already registered and override=False.

        Example:
            >>> @registry.register("faster_rcnn", aliases=["frcnn"])
            ... class FasterRCNN:
            ...     pass
        """

        def decorator(cls: type[T]) -> type[T]:
            all_names = [name] + (aliases or [])

            for reg_name in all_names:
                if reg_name in self._registry and not override:
                    raise ConfigurationError(
                        f"'{reg_name}' is already registered in {self.name} registry. "
                        f"Use override=True to replace it.",
                        details={
                            "registry": self.name,
                            "name": reg_name,
                            "existing": self._registry[reg_name].__name__,
                            "new": cls.__name__,
                        },
                    )

                self._registry[reg_name] = cls
                logger.debug(
                    f"Registered '{reg_name}' in {self.name} registry",
                    registry=self.name,
                    name=reg_name,
                    cls=cls.__name__,
                )

            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Retrieve a registered class by name.

        Args:
            name: Name of the registered class.

        Returns:
            The registered class.

        Raises:
            ConfigurationError: If name is not registered.

        Example:
            >>> model_cls = registry.get("faster_rcnn")
            >>> model = model_cls(num_classes=80)
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ConfigurationError(
                f"'{name}' is not registered in {self.name} registry.",
                details={
                    "registry": self.name,
                    "requested": name,
                    "available": available,
                },
            )
        return self._registry[name]

    def get_or_none(self, name: str) -> type[T] | None:
        """Retrieve a registered class, returning None if not found.

        Args:
            name: Name of the registered class.

        Returns:
            The registered class or None if not found.
        """
        return self._registry.get(name)

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Build an instance of a registered class.

        This is a convenience method that retrieves the class and
        instantiates it with the provided arguments.

        Args:
            name: Name of the registered class.
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.

        Returns:
            Instance of the registered class.

        Example:
            >>> model = registry.build("faster_rcnn", num_classes=80)
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list_registered(self) -> list[str]:
        """List all registered names.

        Returns:
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: Name to check.

        Returns:
            True if name is registered.
        """
        return name in self._registry

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for checking registration."""
        return self.is_registered(name)

    def __len__(self) -> int:
        """Return number of registered items."""
        return len(self._registry)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Registry(name='{self.name}', items={len(self._registry)})"


def create_registry(name: str) -> Registry[Any]:
    """Factory function to create a new registry.

    Args:
        name: Name for the registry.

    Returns:
        New Registry instance.

    Example:
        >>> MODEL_REGISTRY = create_registry("models")
        >>> TRANSFORM_REGISTRY = create_registry("transforms")
    """
    return Registry(name)
