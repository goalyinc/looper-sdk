"""Raw-result expiry preserves typed, non-retryable retrieval errors."""

from looper_sdk import errors, wire
from looper_sdk.proto.v1 import operations_pb2, sampling_pb2


def test_expiry_is_distinct_from_computation_failure():
    error_type = getattr(errors, "ResultExpired", None)
    assert error_type is not None
    error = error_type("The original result expired")
    decoded = wire.error_from_proto(wire.error_to_proto(error))
    assert isinstance(decoded, error_type)
    assert decoded.code == "RESULT_EXPIRED"
    assert decoded.retryable is False


def test_successful_operation_can_report_expired_raw_data():
    for cls in (operations_pb2.Operation, sampling_pb2.SampleOperation):
        message = cls()
        assert "raw_data_expired_at" in message.DESCRIPTOR.fields_by_name
        message.raw_data_expired_at.FromJsonString("2026-11-09T00:00:00Z")
        restored = cls.FromString(message.SerializeToString())
        assert restored.HasField("raw_data_expired_at")
