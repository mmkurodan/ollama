#!/usr/bin/env bash
# ============================================================================
# build-local.sh — llama のハイブリッド・ローカルビルド
#
# 想定する 3 つの導線:
#   (1) ネイティブ(C/C++)を変更 → CI で native をコンパイル（push）→ 本スクリプト --fetch で
#       最新 .so を取得してローカルで APK 組立。
#   (2) ネイティブ未変更 → 取得済み .so を再利用（--no-fetch）してローカルで APK 組立（最速）。
#   (3) すべて CI → 本スクリプトは使わず GitHub Actions のみ。
#
# 仕組み: CI が artifact 出力する APK から arm64-v8a の .so を取り出し jniLibs へ配置、
#         externalNativeBuild を一時無効化して gradle assemble を回す（native 再コンパイル無し）。
#         x86_64 のビルドツールは QEMU_CPU=max（binfmt+qemu）で動かす。
#
# Usage: ./build-local.sh [--debug|--release] [--fetch|--no-fetch] [--run-id ID] [--artifact NAME]
#   --debug        debug APK（既定）
#   --release      署名付き release APK（keystore: KEYSTORE_PATH or ./release.keystore or ~/keystore.jks or /root/keystore.jks）
#   --fetch        最新の成功 CI run から .so を取得（既定）
#   --no-fetch     取得せず既存 jniLibs の .so を再利用
#   --run-id ID    取得元の CI run を明示
#   --artifact N   取得する artifact 名（既定: app-debug）
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
MODULE="app"
JNILIBS="$REPO_ROOT/$MODULE/src/main/jniLibs/arm64-v8a"
BUILD_GRADLE="$REPO_ROOT/$MODULE/build.gradle"

VARIANT="debug"; DO_FETCH=1; RUN_ID=""; ARTIFACT="app-debug"; TMP=""; BG_BAK=""; KS_INSTALLED=""; KS_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --debug) VARIANT="debug" ;;
    --release) VARIANT="release"; ARTIFACT="app-debug" ;;  # .so は debug/release 共通
    --fetch) DO_FETCH=1 ;;
    --no-fetch) DO_FETCH=0 ;;
    --run-id) RUN_ID="$2"; shift ;;
    --artifact) ARTIFACT="$2"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

# aarch64 ホストで x86_64 の aapt2/d8/zipalign 等を qemu で動かす（未対応命令対策）。
export QEMU_CPU="${QEMU_CPU:-max}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk}"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
GRADLE_RUNNER="$(command -v gradle || echo ./gradlew)"
REPO_SLUG="$(git remote get-url origin 2>/dev/null | sed -E 's|.*github\.com[:/]||; s|\.git$||' || true)"

# -------- 1) CI 成果物(.so) の取得 --------
if [ "$DO_FETCH" -eq 1 ]; then
  command -v gh >/dev/null || { echo "gh CLI が必要です（--no-fetch で回避可）"; exit 2; }
  if [ -z "$RUN_ID" ]; then
    RUN_ID="$(gh run list --repo "$REPO_SLUG" --status success --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
  fi
  [ -n "$RUN_ID" ] || { echo "成功した CI run が見つかりません"; exit 2; }
  echo "== fetch .so from CI run $RUN_ID (artifact=$ARTIFACT) =="
  TMP="$(mktemp -d)"
  gh run download "$RUN_ID" --repo "$REPO_SLUG" -n "$ARTIFACT" -D "$TMP"
  CI_APK="$(find "$TMP" -name '*.apk' | head -1)"
  [ -n "$CI_APK" ] || { echo "artifact に APK がありません"; exit 2; }
  mkdir -p "$JNILIBS"
  unzip -o -j "$CI_APK" 'lib/arm64-v8a/*.so' -d "$JNILIBS" >/dev/null
  echo "== 配置した .so =="; ls -la "$JNILIBS"
fi
[ -n "$(ls -A "$JNILIBS" 2>/dev/null | grep -i '\.so$' || true)" ] || { echo "jniLibs に .so がありません（--fetch してください）"; exit 2; }

# -------- 2) externalNativeBuild を一時無効化（native 再コンパイルを避ける）--------
BG_BAK="$(mktemp)"; cp "$BUILD_GRADLE" "$BG_BAK"
restore_gradle(){
  [ -n "$BG_BAK" ] && [ -f "$BG_BAK" ] && cp "$BG_BAK" "$BUILD_GRADLE" && rm -f "$BG_BAK"
  [ -n "$KS_INSTALLED" ] && [ -n "$KS_NAME" ] && rm -f "$REPO_ROOT/$MODULE/$KS_NAME" 2>/dev/null || true
  [ -n "$TMP" ] && rm -rf "$TMP" 2>/dev/null || true
}
trap 'restore_gradle' EXIT
# android.externalNativeBuild { cmake { path "..." } } の path 行をコメント化
sed -i -E 's#^([[:space:]]*)path[[:space:]]+"src/main/cpp/CMakeLists.txt"#\1// path "src/main/cpp/CMakeLists.txt" (build-local: prebuilt .so)#' "$BUILD_GRADLE"

# -------- 3) release 署名の準備 --------
if [ "$VARIANT" = "release" ]; then
  # build.gradle が参照する keystore ファイル名を検出（release.keystore or keystore.jks）
  KS_NAME="$(sed -nE 's/.*storeFile[[:space:]]+file\("([^"]+)"\).*/\1/p' "$BUILD_GRADLE" | head -1)"
  KS_NAME="${KS_NAME:-release.keystore}"
  KS_SRC="${KEYSTORE_PATH:-}"
  if [ -z "$KS_SRC" ]; then
    for c in "$REPO_ROOT/$KS_NAME" "$REPO_ROOT/release.keystore" "$HOME/keystore.jks" "/root/keystore.jks"; do
      [ -f "$c" ] && { KS_SRC="$c"; break; }
    done
  fi
  [ -f "${KS_SRC:-}" ] || { echo "release keystore が見つかりません（KEYSTORE_PATH を設定）"; exit 2; }
  cp -f "$KS_SRC" "$REPO_ROOT/$MODULE/$KS_NAME"; KS_INSTALLED=1
  export KEYSTORE_PASSWORD="${KEYSTORE_PASSWORD:-micklab}"
  export KEY_ALIAS="${KEY_ALIAS:-mykey}"
  export KEY_PASSWORD="${KEY_PASSWORD:-micklab}"
  echo "== release signing: $KS_SRC -> $MODULE/$KS_NAME =="
fi

# -------- 4) local.properties --------
grep -q "^sdk.dir=" "$REPO_ROOT/local.properties" 2>/dev/null || echo "sdk.dir=$ANDROID_SDK_ROOT" > "$REPO_ROOT/local.properties"

# -------- 5) assemble --------
TASK="assembleDebug"; OUT_DIR="$MODULE/build/outputs/apk/debug"
[ "$VARIANT" = "release" ] && { TASK="assembleRelease"; OUT_DIR="$MODULE/build/outputs/apk/release"; }
echo "== $GRADLE_RUNNER :$MODULE:$TASK (QEMU_CPU=$QEMU_CPU) =="
"$GRADLE_RUNNER" --no-daemon --console=plain ":$MODULE:$TASK"

# -------- 6) 配置 --------
APK="$(find "$REPO_ROOT/$OUT_DIR" -name '*.apk' | sort | tail -1)"
DEST_DIR="${APK_OUTPUT_DIR:-}"; [ -z "$DEST_DIR" ] && { [ -d /sdcard/Download ] && [ -w /sdcard/Download ] && DEST_DIR=/sdcard/Download || DEST_DIR="$HOME/downloads"; }
mkdir -p "$DEST_DIR"
PROJ="$(basename "$REPO_ROOT")"
cp -f "$APK" "$DEST_DIR/$PROJ-app-$VARIANT.apk"
echo "== DONE: $DEST_DIR/$PROJ-app-$VARIANT.apk =="
