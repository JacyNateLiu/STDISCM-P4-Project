"""Lightweight protobuf definitions for the dashboard gRPC contract."""

from __future__ import annotations

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

_file_proto = descriptor_pb2.FileDescriptorProto()
_file_proto.name = "dashboard.proto"
_file_proto.package = "dashboard"
_file_proto.syntax = "proto3"

_sample_entry = _file_proto.message_type.add()
_sample_entry.name = "SampleEntry"
_field = _sample_entry.field.add()
_field.name = "sample_id"
_field.number = 1
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_field = _sample_entry.field.add()
_field.name = "image_b64"
_field.number = 2
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_field = _sample_entry.field.add()
_field.name = "prediction"
_field.number = 3
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_field = _sample_entry.field.add()
_field.name = "ground_truth"
_field.number = 4
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_field = _sample_entry.field.add()
_field.name = "confidence"
_field.number = 5
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

_batch_request = _file_proto.message_type.add()
_batch_request.name = "BatchUpdateRequest"
_field = _batch_request.field.add()
_field.name = "iteration"
_field.number = 1
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
_field = _batch_request.field.add()
_field.name = "loss"
_field.number = 2
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
_field = _batch_request.field.add()
_field.name = "batch_size"
_field.number = 3
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
_field = _batch_request.field.add()
_field.name = "samples"
_field.number = 4
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
_field.type_name = ".dashboard.SampleEntry"
_field = _batch_request.field.add()
_field.name = "took_ms"
_field.number = 5
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
_field = _batch_request.field.add()
_field.name = "delay_ms"
_field.number = 6
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32

_batch_response = _file_proto.message_type.add()
_batch_response.name = "BatchUpdateResponse"
_field = _batch_response.field.add()
_field.name = "status"
_field.number = 1
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_field = _batch_response.field.add()
_field.name = "iteration"
_field.number = 2
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
_field = _batch_response.field.add()
_field.name = "loss"
_field.number = 3
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
_field = _batch_response.field.add()
_field.name = "tiles_ready"
_field.number = 4
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
_field = _batch_response.field.add()
_field.name = "samples_available"
_field.number = 5
_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32

_descriptor = _descriptor_pool.Default().AddSerializedFile(
    _file_proto.SerializeToString()
)
DESCRIPTOR = _descriptor

SampleEntry = _reflection.GeneratedProtocolMessageType(
    "SampleEntry",
    (_message.Message,),
    {
        "DESCRIPTOR": DESCRIPTOR.message_types_by_name["SampleEntry"],
        "__module__": "dashboard.server.proto.dashboard_pb2",
    },
)
_sym_db.RegisterMessage(SampleEntry)

BatchUpdateRequest = _reflection.GeneratedProtocolMessageType(
    "BatchUpdateRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": DESCRIPTOR.message_types_by_name["BatchUpdateRequest"],
        "__module__": "dashboard.server.proto.dashboard_pb2",
    },
)
_sym_db.RegisterMessage(BatchUpdateRequest)

BatchUpdateResponse = _reflection.GeneratedProtocolMessageType(
    "BatchUpdateResponse",
    (_message.Message,),
    {
        "DESCRIPTOR": DESCRIPTOR.message_types_by_name["BatchUpdateResponse"],
        "__module__": "dashboard.server.proto.dashboard_pb2",
    },
)
_sym_db.RegisterMessage(BatchUpdateResponse)

__all__ = [
    "BatchUpdateRequest",
    "BatchUpdateResponse",
    "SampleEntry",
]
