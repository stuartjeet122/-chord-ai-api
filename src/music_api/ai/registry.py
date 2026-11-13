from __future__ import annotations

from collections.abc import Iterable

from music_api.ai.schemas import ModelDefinition


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self, definitions: Iterable[ModelDefinition] = ()) -> None:
        self._definitions: dict[str, ModelDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ModelDefinition) -> None:
        if definition.model_id in self._definitions:
            raise ValueError(f"Model ID is already registered: {definition.model_id}.")
        self._definitions[definition.model_id] = definition

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._definitions[model_id]
        except KeyError as exc:
            raise ModelNotFoundError(f"Model is not registered: {model_id}.") from exc

    def list(self) -> tuple[ModelDefinition, ...]:
        return tuple(self._definitions.values())
