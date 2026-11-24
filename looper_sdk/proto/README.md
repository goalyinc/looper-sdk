# proto/

`looper.v1`'s messages: HTTP request and response bodies, binary protobuf or
protobuf's JSON mapping.

| File | Origin |
|---|---|
| `v1/*.proto` | **Hand-written**: the only files to edit here |
| `v1/*_pb2.py` | Generated: message classes |
| `v1/*_pb2.pyi` | Generated: type stubs |

Edit a `.proto`, never a generated file: the generator writes the
`_pb2` modules from it, and only message modules, never gRPC stubs.

Compatibility: a field is added only as an optional field, and a breaking
change is a `v2`, so neither encoding, a field's number nor its name ever
changes.
