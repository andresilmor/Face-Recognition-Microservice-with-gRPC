from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FaceRecognitionInferenceReply(_message.Message):
    __slots__ = ["recognitions"]
    RECOGNITIONS_FIELD_NUMBER: _ClassVar[int]
    recognitions: _containers.RepeatedCompositeFieldContainer[FaceRecognized]
    def __init__(self, recognitions: _Optional[_Iterable[_Union[FaceRecognized, _Mapping]]] = ...) -> None: ...

class FaceRecognitionRequest(_message.Message):
    __slots__ = ["collections", "image"]
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedScalarFieldContainer[str]
    image: bytes
    def __init__(self, image: _Optional[bytes] = ..., collections: _Optional[_Iterable[str]] = ...) -> None: ...

class FaceRecognitionWithRectListRequest(_message.Message):
    __slots__ = ["collections", "image", "rectList"]
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    RECTLIST_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedScalarFieldContainer[str]
    image: bytes
    rectList: _containers.RepeatedCompositeFieldContainer[FaceRect]
    def __init__(self, image: _Optional[bytes] = ..., collections: _Optional[_Iterable[str]] = ..., rectList: _Optional[_Iterable[_Union[FaceRect, _Mapping]]] = ...) -> None: ...

class FaceRecognized(_message.Message):
    __slots__ = ["faceRect", "uuid"]
    FACERECT_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    faceRect: FaceRect
    uuid: str
    def __init__(self, uuid: _Optional[str] = ..., faceRect: _Optional[_Union[FaceRect, _Mapping]] = ...) -> None: ...

class FaceRect(_message.Message):
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
