#!/bin/bash

parse_extra_args() {
    local extra_args="${1:-}"
    if [ "${extra_args}" = "" ]; then
        return 0
    fi
    echo "${extra_args}" | awk -F, '{
        for (i = 1; i <= NF; i++) {
            split($i, arr, "=")
            if (arr[1] != "") {
                printf "--%s %s ", arr[1], arr[2]
            }
        }
    }'
}

parse_extra_flags() {
    local extra_flags="${1:-}"
    if [ "${extra_flags}" = "" ]; then
        return 0
    fi
    echo "${extra_flags}" | awk -F, '{
        for (i = 1; i <= NF; i++) {
            if ($i != "") {
                printf "--%s ", $i
            }
        }
    }'
}

