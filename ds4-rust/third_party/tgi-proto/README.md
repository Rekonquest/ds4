# Vendored TGI gRPC Protocol Definitions

This directory contains the gRPC protocol definitions vendored from
[Hugging Face's text-generation-inference](https://github.com/huggingface/text-generation-inference)
for reference and future compilation by the DwarfStar (DS4) Rust rewrite.

## Version kept: **v3** only

The upstream `proto/` directory contains two API surfaces:

- `proto/generate.proto`         — legacy v2 API (`package generate.v2;`)
- `proto/v3/generate.proto`      — current v3 API (`package generate.v3;`)

Per the v1 vendoring plan, **only the v3 definitions are kept**. The legacy v2
`generate.proto` is intentionally omitted. A future v2 of DS4 will compile
these `.proto` files into Rust via `tonic`/`prost`; the v1 build does not link
against them.

## Layout

```
third_party/tgi-proto/
├── LICENSE                     # Apache-2.0, verbatim from upstream
└── proto/
    └── v3/
        └── generate.proto      # TGI v3 gRPC service & message definitions
```

## Upstream source

- Repository: `https://github.com/huggingface/text-generation-inference`
- Source path: `proto/v3/generate.proto`
- License: Apache-2.0 (see `LICENSE` in this directory)
- Upstream does **not** ship a separate `NOTICE` file.

## Constraints on vendored files

The `.proto` files in this directory must not be modified locally — they are
kept verbatim as a reference. Any DWARFStar-specific extensions belong in a
separate, non-vendored proto file that `import`s the vendored definitions.

The `third_party/tgi-proto/` directory is **not** a Cargo workspace member
and is not compiled in v1. It exists purely to preserve the protocol contract
for downstream v2 work.