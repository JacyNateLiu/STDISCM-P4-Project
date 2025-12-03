"""Async gRPC bindings for the dashboard service."""

from __future__ import annotations

import grpc

from . import dashboard_pb2 as dashboard__pb2


class TrainingDashboardStub:
    """Client stub that talks to the dashboard ingestion service."""

    def __init__(self, channel: grpc.aio.Channel) -> None:
        self.BatchUpdate = channel.unary_unary(
            "/dashboard.TrainingDashboard/BatchUpdate",
            request_serializer=dashboard__pb2.BatchUpdateRequest.SerializeToString,
            response_deserializer=dashboard__pb2.BatchUpdateResponse.FromString,
        )


class TrainingDashboardServicer:
    """Base class for implementing the service on the server side."""

    def BatchUpdate(self, request, context):  # pragma: no cover - interface method
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("BatchUpdate not implemented")


def add_TrainingDashboardServicer_to_server(servicer, server) -> None:
    rpc_method_handlers = {
        "BatchUpdate": grpc.unary_unary_rpc_method_handler(
            servicer.BatchUpdate,
            request_deserializer=dashboard__pb2.BatchUpdateRequest.FromString,
            response_serializer=dashboard__pb2.BatchUpdateResponse.SerializeToString,
        )
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "dashboard.TrainingDashboard",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))


__all__ = [
    "TrainingDashboardStub",
    "TrainingDashboardServicer",
    "add_TrainingDashboardServicer_to_server",
]
