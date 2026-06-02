#!/bin/bash

resolve_s2s_backbone_arg() {
    local backbone="$1"
    case "${backbone}" in
        Qwen3Session2)
            echo "Qwen3Session"
            ;;
        Llama)
            echo "LlamaMulti"
            ;;
        Qwen3Multi*)
            echo "Qwen3Multi"
            ;;
        Qwen3TemporalHierarchical*)
            echo "Qwen3TemporalHierarchical"
            ;;
        *)
            echo "${backbone}"
            ;;
    esac
}

resolve_s2s_base_model() {
    local backbone="$1"
    case "${backbone}" in
        TIGER)
            echo "./config/s2s-models/TIGER"
            ;;
        PBATransformer)
            echo "./config/s2s-models/PBATransformer"
            ;;
        Qwen3|Qwen3Session)
            echo "./config/s2s-models/Qwen3-Light"
            ;;
        Qwen3Session2)
            echo "./config/s2s-models/Qwen3-Light-2"
            ;;
        Llama)
            echo "./config/s2s-models/Llama"
            ;;
        LlamaMulti)
            echo "./config/s2s-models/LlamaMulti"
            ;;
        Qwen3Multi*|Qwen3TemporalHierarchical*)
            echo "./config/s2s-models/${backbone}"
            ;;
        *)
            if [ -d "./config/s2s-models/${backbone}" ]; then
                echo "./config/s2s-models/${backbone}"
            else
                return 1
            fi
            ;;
    esac
}

