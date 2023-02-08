from enum import Enum
from typing import Optional, Union

class ExtendedEnum(str, Enum):
    """
    From https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/enums.py#L19
    Added list() from https://stackoverflow.com/a/54919285
    """
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_str(cls, value: str) -> Optional['ExtendedEnum']:
        statuses = [status for status in dir(cls) if not status.startswith('_')]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        return None

    def __eq__(self, other: Union[str, Enum]) -> bool:  # type: ignore
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())
