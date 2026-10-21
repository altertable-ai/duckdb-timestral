# vcpkg overlay ports

These overlays are copied from [anofox_tabfm](https://github.com/DataZooDE/anofox-tabfm)
(commit `ff5af557ca31a1ef0cf63c3345789f68d62c4dd7`, MIT licence, © 2026 DataZoo GmbH). That community
extension already builds ONNX Runtime statically in the DuckDB extension CI, and these are the ports it uses.

| Port | Version | Why it is overlaid |
|---|---|---|
| `onnxruntime` | 1.29.0 | DuckDB's pinned vcpkg commit only has 1.23.2. The vcpkg-built `flatc` segfaults in DuckDB's AlmaLinux 8 build image, so this port uses the flatbuffers headers checked into ONNX Runtime and skips code generation. **Changed here:** on arm64 it enables KleidiAI (Arm's SME/NEON GEMM kernels), which upstream's `build.py` turns on but the CMake default leaves off. KleidiAI v1.20.0 is fetched by the portfile, because vcpkg runs FetchContent disconnected. |
| `onnx` | 1.22.0 | Matches ONNX Runtime 1.29.0. **Changed here:** it is built with `ONNX_DISABLE_STATIC_REGISTRATION=ON`, which the ONNX Runtime port requires. Without it, every first model load prints hundreds of "schema already registered" errors. |
| `flatbuffers` | 23.5.26 | Pinned to the version ONNX Runtime's checked-in headers were generated with. |

The triplets in `../toolchains/` come from the same repository. They make the Linux builds release-only,
which roughly halves the time it takes to build ONNX Runtime from source.
