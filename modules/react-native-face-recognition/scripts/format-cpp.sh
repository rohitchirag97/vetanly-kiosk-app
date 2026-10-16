#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: scripts/format-cpp.sh <check|write>" >&2
  exit 2
fi

mode="$1"
if [[ "$mode" != "check" && "$mode" != "write" ]]; then
  echo "Expected mode to be 'check' or 'write'." >&2
  exit 2
fi

found_file=0
while IFS= read -r -d '' file_path; do
  found_file=1
  if [[ "$mode" == "check" ]]; then
    clang-format --dry-run --Werror "$file_path"
  else
    clang-format -i "$file_path"
  fi
done < <(
  find cpp android/src/main/cpp ios \
    -type f \
    \( \
      -name '*.cc' \
      -o -name '*.cpp' \
      -o -name '*.cxx' \
      -o -name '*.h' \
      -o -name '*.hpp' \
      -o -name '*.mm' \
    \) \
    -print0
)

if [[ "$found_file" -eq 0 ]]; then
  echo "No C++ files found to format." >&2
  exit 1
fi
