# hf-fetch

A low-dependency, low-complexity way to download model files from the Hugging Face
Hub. It exists because **hf-hub 1.0 brings in far more than this job needs**:
tokio, reqwest, and hf-xet to fetch a few model files. hf-fetch uses a small,
blocking `ureq` client with rustls. It needs no async runtime or system OpenSSL.

The endpoint, token, and cache resolve as they do in Python's `huggingface_hub`,
so Rust and Python tools can share downloaded files.

```rust,no_run
use hf_fetch::Hub;

# fn main() -> Result<(), hf_fetch::HubError> {
let hub = Hub::from_env()?;
let repo = hub.model("avencera/scriptrs-models", "main");
let path = repo.get("config.json")?;
# Ok(())
# }
```

`Hub::with_cache_dir(path)` selects a cache directory. `Hub::model(id, revision)`
selects a model repository and a branch, tag, or commit. Use a commit hash to pin
the model version.

- `HubRepo::cached(file)` finds a local snapshot without a network request.
- `HubRepo::get(file)` uses the cached file, or downloads it if it is absent.
- `HubRepo::download(file)` asks the Hub for metadata even if a copy is cached.
- `HubError::status()` returns the HTTP status when the Hub rejects a request.

## Configuration

Empty environment variables count as unset. A leading `~` in a path expands to
the home directory.

| Setting | Resolution order |
| --- | --- |
| Endpoint | `HF_ENDPOINT`, then `https://huggingface.co` |
| Token | nonblank `HF_TOKEN`, then the file at `HF_TOKEN_PATH`, then `$HF_HOME/token` if no token path is set |
| Hub cache | `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `$HF_HOME/hub` |
| HF home | `HF_HOME`, then `$XDG_CACHE_HOME/huggingface`, then `~/.cache/huggingface` |

`hf auth login` writes the default token file. Line breaks and surrounding
whitespace are removed from tokens. A missing token file means no token; other
file read errors are returned. `default_hub_cache_dir()` exposes cache resolution
without constructing a client.

Connections have a 30-second timeout. Waiting for response headers and each body
read have a 60-second stall timeout. There is **no whole-download time limit**,
so large files can finish over a slow connection.

## Shared cache and scope

Files use `models--owner--repo/blobs`, `refs`, and `snapshots`. Downloads stream
through separate temporary files and move into place after completion. Unix
snapshots use relative symlinks; other platforms use copies. Tokens are sent to
the Hub and are not sent to redirected storage URLs.

A cached branch or tag is used without checking whether it moved. Call
`download()` to refresh it. Cached files can load without network access or a
token. Client construction still reads the configured token file.

This crate downloads model files only. It does not provide uploads, repository
management, dataset APIs, an async API, an Xet client, progress reporting, resume,
or automatic retries. Large LFS and Xet-backed files use the Hub's HTTP storage
redirects.

The default `online` feature enables downloads. Disable default features to use
only `default_hub_cache_dir()`, with no dependencies:

```toml
[dependencies]
hf-fetch = { git = "https://github.com/avencera/hf-fetch", default-features = false }
```

For downloads, keep default features enabled. The crate uses edition 2024 and
stable Rust.

## Development

Run `just check` for formatting, dependency scope, clippy, and tests. Run
`cargo deny --all-features check` for dependency policy. CI checks Linux, macOS,
and Windows. Tests use local HTTP servers and do not download model weights.

Licensed under Apache-2.0.
