"""Unit tests for core registry module."""

from __future__ import annotations

import pytest

from objdet.core.exceptions import ConfigurationError
from objdet.core.registry import Registry, create_registry


class TestRegistry:
    """Tests for the Registry class."""

    def test_register_and_get(self) -> None:
        """Test basic registration and retrieval."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        retrieved = registry.get("my_class")
        assert retrieved is MyClass

    def test_register_with_aliases(self) -> None:
        """Test registration with aliases."""
        registry: Registry[type] = Registry("test")

        @registry.register("primary", aliases=["alias1", "alias2"])
        class MyClass:
            pass

        assert registry.get("primary") is MyClass
        assert registry.get("alias1") is MyClass
        assert registry.get("alias2") is MyClass

    def test_register_duplicate_raises_error(self) -> None:
        """Test that duplicate registration raises ConfigurationError."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_class")
        class MyClass1:
            pass

        with pytest.raises(ConfigurationError) as exc_info:

            @registry.register("my_class")
            class MyClass2:
                pass

        assert "already registered" in str(exc_info.value)

    def test_register_with_override(self) -> None:
        """Test that override=True allows replacing registrations."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_class")
        class MyClass1:
            pass

        @registry.register("my_class", override=True)
        class MyClass2:
            pass

        assert registry.get("my_class") is MyClass2

    def test_get_unregistered_raises_error(self) -> None:
        """Test that getting unregistered name raises ConfigurationError."""
        registry: Registry[type] = Registry("test")

        with pytest.raises(ConfigurationError) as exc_info:
            registry.get("nonexistent")

        assert "not registered" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_or_none(self) -> None:
        """Test get_or_none returns None for unregistered names."""
        registry: Registry[type] = Registry("test")

        @registry.register("exists")
        class MyClass:
            pass

        assert registry.get_or_none("exists") is MyClass
        assert registry.get_or_none("not_exists") is None

    def test_build(self) -> None:
        """Test building instances from registered classes."""
        registry: Registry[object] = Registry("test")

        @registry.register("my_class")
        class MyClass:
            def __init__(self, value: int) -> None:
                self.value = value

        instance = registry.build("my_class", value=42)
        assert isinstance(instance, MyClass)
        assert instance.value == 42

    def test_list_registered(self) -> None:
        """Test listing all registered names."""
        registry: Registry[type] = Registry("test")

        @registry.register("class_b")
        class ClassB:
            pass

        @registry.register("class_a")
        class ClassA:
            pass

        registered = registry.list_registered()
        assert registered == ["class_a", "class_b"]  # Sorted

    def test_is_registered(self) -> None:
        """Test checking if a name is registered."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        assert registry.is_registered("my_class") is True
        assert registry.is_registered("other") is False

    def test_contains(self) -> None:
        """Test 'in' operator support."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        assert "my_class" in registry
        assert "other" not in registry

    def test_len(self) -> None:
        """Test len() returns number of registered items."""
        registry: Registry[type] = Registry("test")

        assert len(registry) == 0

        @registry.register("class1")
        class Class1:
            pass

        assert len(registry) == 1

        @registry.register("class2")
        class Class2:
            pass

        assert len(registry) == 2

    def test_repr(self) -> None:
        """Test string representation."""
        registry: Registry[type] = Registry("models")
        assert "Registry" in repr(registry)
        assert "models" in repr(registry)


class TestCreateRegistry:
    """Tests for the create_registry factory function."""

    def test_creates_registry(self) -> None:
        """Test that create_registry returns a Registry instance."""
        registry = create_registry("my_registry")
        assert isinstance(registry, Registry)
        assert registry.name == "my_registry"

    def test_multiple_registries_are_independent(self) -> None:
        """Test that multiple registries don't share state."""
        registry1 = create_registry("r1")
        registry2 = create_registry("r2")

        @registry1.register("item")
        class Item1:
            pass

        @registry2.register("item")
        class Item2:
            pass

        assert registry1.get("item") is Item1
        assert registry2.get("item") is Item2
