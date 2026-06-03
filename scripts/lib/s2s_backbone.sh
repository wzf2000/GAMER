#!/bin/bash

resolve_s2s_backbone_arg() {
    local backbone="$1"
    local python_bin="${PYTHON:-python}"
    "${python_bin}" -m SeqRec.models.generative.registry resolve-backbone "${backbone}"
}

resolve_s2s_base_model() {
    local backbone="$1"
    local python_bin="${PYTHON:-python}"
    "${python_bin}" -m SeqRec.models.generative.registry resolve-base-model "${backbone}"
}
