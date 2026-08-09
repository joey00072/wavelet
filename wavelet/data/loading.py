"""Compatibility wrapper; implementation lives in wavelet.data.sft."""

from wavelet.data.sft import (
    Example as Example,
    Stats as Stats,
    _coerce_messages as _coerce_messages,
    _deserialize_tool_calls as _deserialize_tool_calls,
    _strip_message_content as _strip_message_content,
    _merge_message_thinking as _merge_message_thinking,
    _split_messages_for_sft as _split_messages_for_sft,
    _prepend_system_prompt as _prepend_system_prompt,
    _paths as _paths,
    _load_payloads as _load_payloads,
    _load_local_payload_groups as _load_local_payload_groups,
    _load_hf_payload_groups as _load_hf_payload_groups,
    _load_fake_payload_groups as _load_fake_payload_groups,
    _hf_subsets_and_splits as _hf_subsets_and_splits,
    _mix_payload_groups as _mix_payload_groups,
    normalize_record as normalize_record,
    load_data_payloads as load_data_payloads,
    load_records as load_records,
)
