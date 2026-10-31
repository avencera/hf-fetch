fmt:
    cargo fmt --all

clippy:
    cargo clippy --all-targets --all-features -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings

test:
    cargo test --all-features
    cargo test --no-default-features

# prevent the async stack this crate replaces from entering normal dependencies
small-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    deps=$(cargo tree --all-features --target all -e normal,build --prefix none)
    if rg '^(tokio|reqwest|hf-xet|hf-hub) ' <<<"$deps"; then
        echo "async Hub dependency found" >&2
        exit 1
    fi
    offline=$(cargo tree --no-default-features -e normal,build --prefix none)
    test "$(wc -l <<<"$offline")" -eq 1

check: fmt small-deps clippy test
