#!/bin/bash

parse_script_path_args() {
    SCRIPT_CLI_ARGS=()

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --suffix)
                if [ "$#" -lt 2 ]; then
                    echo "ERROR: --suffix requires a value."
                    exit 1
                fi
                suffix="$2"
                shift 2
                ;;
            --suffix=*)
                suffix="${1#--suffix=}"
                shift
                ;;
            *)
                SCRIPT_CLI_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

build_extra_cli_args() {
    EXTRA_CLI_ARGS=()

    local legacy_extra_args="${extra_args:-}"
    if [ "${legacy_extra_args}" != "" ]; then
        local pair key value
        IFS=',' read -ra extra_arg_pairs <<< "${legacy_extra_args}"
        for pair in "${extra_arg_pairs[@]}"; do
            if [[ "${pair}" == *=* ]]; then
                key="${pair%%=*}"
                value="${pair#*=}"
                if [ "${key}" != "" ]; then
                    EXTRA_CLI_ARGS+=("--${key}" "${value}")
                fi
            elif [ "${pair}" != "" ]; then
                key="${pair}"
                EXTRA_CLI_ARGS+=("--${key}")
            fi
        done
    fi

    local legacy_extra_flags="${extra_flags:-}"
    if [ "${legacy_extra_flags}" != "" ]; then
        local flag
        IFS=',' read -ra extra_flag_names <<< "${legacy_extra_flags}"
        for flag in "${extra_flag_names[@]}"; do
            if [ "${flag}" != "" ]; then
                EXTRA_CLI_ARGS+=("--${flag}")
            fi
        done
    fi

    EXTRA_CLI_ARGS+=("$@")
}

print_extra_cli_args() {
    if [ "${#EXTRA_CLI_ARGS[@]}" -eq 0 ]; then
        echo "Extra CLI arguments: "
    else
        printf "Extra CLI arguments:"
        printf " %q" "${EXTRA_CLI_ARGS[@]}"
        printf "\n"
    fi
}
