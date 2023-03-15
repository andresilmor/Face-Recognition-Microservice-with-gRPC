from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Box(_message.Message):
    __slots__ = ["x1", "x2", "y1", "y2"]
    X1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: int
    x2: int
    y1: int
    y2: int
    def __init__(self, x1: _Optional[int] = ..., y1: _Optional[int] = ..., x2: _Optional[int] = ..., y2: _Optional[int] = ...) -> None: ...

class FaceRecognitionInferenceReply(_message.Message):
    __slots__ = ["box", "uuid"]
    BOX_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    box: Box
    uuid: str
    def __init__(self, uuid: _Optional[str] = ..., box: _Optional[_Union[Box, _Mapping]] = ...) -> None: ...

class FaceRecognitionRequest(_message.Message):
    __slots__ = ["image", "personBox"]
    class PersonBoxEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    PERSONBOX_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    personBox: _containers.ScalarMap[str, int]
    def __init__(self, image: _Optional[bytes] = ..., personBox: _Optional[_Mapping[str, int]] = ...) -> None: ...
