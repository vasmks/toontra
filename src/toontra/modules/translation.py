"""Small offline translators used by the core package and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from toontra.errors import ModelContractError


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ModelContractError("texts must be a sequence of strings")
    result = list(texts)
    for index, text in enumerate(result):
        if not isinstance(text, str):
            raise ModelContractError(f"text item {index} must be a string")
    return result


class IdentityTranslator:
    """Return source strings unchanged while preserving batch order."""

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        del source_language, target_language
        return _validate_texts(texts)


class DictionaryTranslator:
    """Translate exact strings from an in-memory, user-supplied glossary."""

    def __init__(self, translations: Mapping[str, str], *, keep_unknown: bool = True) -> None:
        if not isinstance(translations, Mapping):
            raise TypeError("translations must be a mapping")
        copied: dict[str, str] = {}
        for source, target in translations.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise TypeError("translation keys and values must be strings")
            copied[source] = target
        self._translations = copied
        self.keep_unknown = keep_unknown

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
    ) -> list[str]:
        del source_language, target_language
        source_texts = _validate_texts(texts)
        translated: list[str] = []
        for text in source_texts:
            if text in self._translations:
                translated.append(self._translations[text])
            elif self.keep_unknown:
                translated.append(text)
            else:
                raise KeyError(f"no dictionary translation for {text!r}")
        return translated
