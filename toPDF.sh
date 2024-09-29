#!/usr/bin/env bash

nix-shell -p pandoc texliveSmall --command \
" \
pandoc -V geometry:paperwidth=8.5in \
       -V geometry:paperheight=11in \
       -V geometry:margin=0.8in \
       -V colorlinks=true \
       -V toccolor=gray \
       --from markdown-markdown_in_html_blocks+link_attributes \
       --toc \
       $1 \
       -o $2 \
"