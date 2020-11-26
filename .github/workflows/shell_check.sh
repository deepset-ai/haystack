#!/usr/bin/env bash

# Purpose : Make sure all *.sh scripts are good
# Author  : Ky-Anh Huynh

set +x
set +e

if [[ -z "${USE_CURRENT_DIRECTORY:-}" ]]; then
  cd "$(git rev-parse --show-toplevel)" || exit
fi

: "${SHELL_CHECK_EXCLUDE:=}"

# We are using shellcheck version ~ 0.7
if [[ "$OSTYPE" =~ linux.*  ]] && ! shellcheck --version 2>/dev/null | grep -qs 'version: 0.7'; then
  echo >&2 ":: Downloading shellcheck to $(pwd -P)..."
  wget --quiet -cO shellcheck.txz https://github.com/koalaman/shellcheck/releases/download/v0.7.1/shellcheck-v0.7.1.linux.x86_64.tar.xz
  tar xJf shellcheck.txz
  PATH="$(pwd -P)"/shellcheck-v0.7.1/:$PATH
  export PATH
fi

while read -r file; do
  echo >&2 ":: Checking $file..."
  shellcheck -e "${SHELL_CHECK_EXCLUDE}" "$file" || exit
done < <( \
  find . \
    -type f \
       -iname "*.sh"
)
