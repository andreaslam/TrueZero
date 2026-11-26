#!/usr/bin/env bash
# Build self-contained, single-file executables of tzrust binaries.
#
#   src/rust/bundle.sh [BIN...]        (default: ucimain)
#
# Each BIN is built in release mode and then wrapped by the launcher in bundle/, which
# embeds the binary and its libtorch libraries; the result is written to dist/<BIN> at the
# workspace root and runs on a machine with no libtorch, Python or net files installed
# (same OS and CPU architecture as the build machine).
#
# ucimain also has its default network compiled in: set TZ_DEFAULT_NET (absolute, or
# relative to the workspace root) to choose it; it defaults to tz_36.pt.
# TZ_BUNDLE_LEVEL sets the zstd level (default 12; 19 is smaller but much slower to build).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bins=("$@")
[ ${#bins[@]} -eq 0 ] && bins=(ucimain)

bin_args=()
for b in "${bins[@]}"; do bin_args+=(--bin "$b"); done

echo "==> building ${bins[*]}" >&2
build_json="$(cargo build --manifest-path "$here/Cargo.toml" --release "${bin_args[@]}" \
    --message-format=json-render-diagnostics)"

# the libtorch lib dir is the native link path of torch-sys that holds libtorch_cpu
libtorch=""
while IFS= read -r p; do
    if compgen -G "$p/*torch_cpu*" >/dev/null; then libtorch="$p"; break; fi
done < <(printf '%s\n' "$build_json" | grep '"reason":"build-script-executed"' \
    | grep 'torch-sys' | grep -o '"native=[^"]*"' | sed 's/^"native=//; s/"$//')
if [ -z "$libtorch" ]; then
    echo "bundle.sh: could not find the libtorch lib dir in the torch-sys build output" >&2
    exit 1
fi
echo "==> libtorch: $libtorch" >&2

target_dir="$(printf '%s\n' "$build_json" | grep -o '"executable":"[^"]*"' | head -1 \
    | sed 's/^"executable":"//; s/"$//' | xargs dirname)"
mkdir -p "$root/dist"

for b in "${bins[@]}"; do
    exe="$target_dir/$b"
    [ -f "$exe.exe" ] && exe="$exe.exe"
    echo "==> packing $b" >&2
    TZ_BUNDLE_EXE="$exe" TZ_BUNDLE_LIBS="$libtorch" \
        cargo build --manifest-path "$here/bundle/Cargo.toml" --release \
        --target-dir "$here/target/bundle"
    launcher="$here/target/bundle/release/tzbundle"
    [ -f "$launcher.exe" ] && launcher="$launcher.exe"
    out="$root/dist/$(basename "$exe")"
    cp "$launcher" "$out"
    echo "==> $out ($(du -h "$out" | cut -f1))" >&2
done
