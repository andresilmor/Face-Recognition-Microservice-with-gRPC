# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ms_faceRecognition.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18ms_faceRecognition.proto\x12\x0f\x66\x61\x63\x65Recognition\"u\n\"FaceRecognitionWithRectListRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x13\n\x0b\x63ollections\x18\x02 \x03(\t\x12+\n\x08rectList\x18\x03 \x03(\x0b\x32\x19.faceRecognition.FaceRect\"<\n\x16\x46\x61\x63\x65RecognitionRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x13\n\x0b\x63ollections\x18\x02 \x03(\t\"V\n\x1d\x46\x61\x63\x65RecognitionInferenceReply\x12\x35\n\x0crecognitions\x18\x01 \x03(\x0b\x32\x1f.faceRecognition.FaceRecognized\"k\n\x0e\x46\x61\x63\x65Recognized\x12\x11\n\x04uuid\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x30\n\x08\x66\x61\x63\x65Rect\x18\x02 \x01(\x0b\x32\x19.faceRecognition.FaceRectH\x01\x88\x01\x01\x42\x07\n\x05_uuidB\x0b\n\t_faceRect\":\n\x08\x46\x61\x63\x65Rect\x12\n\n\x02x1\x18\x01 \x01(\r\x12\n\n\x02y1\x18\x02 \x01(\r\x12\n\n\x02x2\x18\x03 \x01(\r\x12\n\n\x02y2\x18\x04 \x01(\r2\xfb\x03\n\x16\x46\x61\x63\x65RecognitionService\x12\x66\n\tInference\x12\'.faceRecognition.FaceRecognitionRequest\x1a..faceRecognition.FaceRecognitionInferenceReply\"\x00\x12\x82\x01\n\x19InferenceWithoutDetection\x12\x33.faceRecognition.FaceRecognitionWithRectListRequest\x1a..faceRecognition.FaceRecognitionInferenceReply\"\x00\x12w\n\x1a\x46\x61stInferenceWithDetection\x12\'.faceRecognition.FaceRecognitionRequest\x1a..faceRecognition.FaceRecognitionInferenceReply\"\x00\x12{\n\x1e\x41\x63\x63urateInferenceWithDetection\x12\'.faceRecognition.FaceRecognitionRequest\x1a..faceRecognition.FaceRecognitionInferenceReply\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ms_faceRecognition_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FACERECOGNITIONWITHRECTLISTREQUEST._serialized_start=45
  _FACERECOGNITIONWITHRECTLISTREQUEST._serialized_end=162
  _FACERECOGNITIONREQUEST._serialized_start=164
  _FACERECOGNITIONREQUEST._serialized_end=224
  _FACERECOGNITIONINFERENCEREPLY._serialized_start=226
  _FACERECOGNITIONINFERENCEREPLY._serialized_end=312
  _FACERECOGNIZED._serialized_start=314
  _FACERECOGNIZED._serialized_end=421
  _FACERECT._serialized_start=423
  _FACERECT._serialized_end=481
  _FACERECOGNITIONSERVICE._serialized_start=484
  _FACERECOGNITIONSERVICE._serialized_end=991
# @@protoc_insertion_point(module_scope)
