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




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18ms_faceRecognition.proto\x12\x0f\x66\x61\x63\x65Recognition\"\xa4\x01\n\x16\x46\x61\x63\x65RecognitionRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\x12I\n\tpersonBox\x18\x02 \x03(\x0b\x32\x36.faceRecognition.FaceRecognitionRequest.PersonBoxEntry\x1a\x30\n\x0ePersonBoxEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\"k\n\x1d\x46\x61\x63\x65RecognitionInferenceReply\x12\x11\n\x04uuid\x18\x01 \x01(\tH\x00\x88\x01\x01\x12&\n\x03\x62ox\x18\x02 \x01(\x0b\x32\x14.faceRecognition.BoxH\x01\x88\x01\x01\x42\x07\n\x05_uuidB\x06\n\x04_box\"5\n\x03\x42ox\x12\n\n\x02x1\x18\x01 \x01(\r\x12\n\n\x02y1\x18\x02 \x01(\r\x12\n\n\x02x2\x18\x03 \x01(\r\x12\n\n\x02y2\x18\x04 \x01(\r2\x80\x01\n\x16\x46\x61\x63\x65RecognitionService\x12\x66\n\tInference\x12\'.faceRecognition.FaceRecognitionRequest\x1a..faceRecognition.FaceRecognitionInferenceReply\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ms_faceRecognition_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FACERECOGNITIONREQUEST_PERSONBOXENTRY._options = None
  _FACERECOGNITIONREQUEST_PERSONBOXENTRY._serialized_options = b'8\001'
  _FACERECOGNITIONREQUEST._serialized_start=46
  _FACERECOGNITIONREQUEST._serialized_end=210
  _FACERECOGNITIONREQUEST_PERSONBOXENTRY._serialized_start=162
  _FACERECOGNITIONREQUEST_PERSONBOXENTRY._serialized_end=210
  _FACERECOGNITIONINFERENCEREPLY._serialized_start=212
  _FACERECOGNITIONINFERENCEREPLY._serialized_end=319
  _BOX._serialized_start=321
  _BOX._serialized_end=374
  _FACERECOGNITIONSERVICE._serialized_start=377
  _FACERECOGNITIONSERVICE._serialized_end=505
# @@protoc_insertion_point(module_scope)
