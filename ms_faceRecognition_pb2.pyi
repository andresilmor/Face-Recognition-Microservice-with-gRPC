from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FaceRecognitionInferenceReply(_message.Message):
    __slots__ = ["faceCenterX", "faceCenterY", "uuid"]
    FACECENTERX_FIELD_NUMBER: _ClassVar[int]
    FACECENTERY_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    faceCenterX: int
    faceCenterY: int
    uuid: str
    def __init__(self, uuid: _Optional[str] = ..., faceCenterX: _Optional[int] = ..., faceCenterY: _Optional[int] = ...) -> None: ...

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
