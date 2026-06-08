from typing import ClassVar, Self


class Registry:
    """Class-as-registry mixin.

    A registry *root* is declared with ``class Foo(Registry, ..., root=True)``; it
    owns a fresh ``{kind: subclass}`` map. Every concrete descendant that sets a
    non-empty ``kind`` class attribute registers itself into the nearest root's map
    at class-creation time. Lookups and enumeration live on the root, so the whole
    tree is discoverable without a parallel module-level dict.
    """

    kind: ClassVar[str] = ""
    _members: ClassVar[dict[str, type]]

    def __init_subclass__(cls, root: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if root:
            cls._members = {}
        elif cls.kind:
            if cls.kind in cls._members:
                raise ValueError(f"duplicate registry kind {cls.kind!r} for {cls.__name__}")
            cls._members[cls.kind] = cls

    @classmethod
    def by_name(cls, kind: str) -> type[Self]:
        if kind not in cls._members:
            raise KeyError(f"unknown {cls.__name__} kind {kind!r}; known: {cls.names()}")
        return cls._members[kind]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._members)
