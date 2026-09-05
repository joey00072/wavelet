from __future__ import annotations

from typing import Any, NotRequired, Required, TypedDict


def apply_verifier_openai_patches() -> None:
    """Apply OpenAI SDK patches used by verifier rollout clients."""
    monkey_patch_oai_iterable_types()
    monkey_patch_chat_completion_logprobs()


def monkey_patch_oai_iterable_types() -> None:
    """Use concrete list types for OpenAI chat message payloads.

    Some OpenAI SDK chat message TypedDicts use lazy iterable annotations. Those
    can interact badly with downstream Pydantic validation in verifier clients,
    especially for tool and multimodal messages.
    """
    try:
        import openai.types.chat as chat_types
        from openai.types.chat.chat_completion_content_part_param import (
            ChatCompletionContentPartParam,
        )
        from openai.types.chat.chat_completion_content_part_text_param import (
            ChatCompletionContentPartTextParam,
        )
        from openai.types.chat.chat_completion_function_message_param import (
            ChatCompletionFunctionMessageParam,
        )
        from openai.types.chat.chat_completion_message import FunctionCall
        from openai.types.chat.chat_completion_message_tool_call_union_param import (
            ChatCompletionMessageToolCallUnionParam,
        )
    except ImportError:
        return

    class DeveloperMessageParam(TypedDict, total=False):
        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[str]
        name: NotRequired[str]

    class SystemMessageParam(TypedDict, total=False):
        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[str]
        name: NotRequired[str]

    class UserMessageParam(TypedDict, total=False):
        content: Required[str | list[ChatCompletionContentPartParam]]
        role: Required[str]
        name: NotRequired[str]

    class AssistantMessageParam(TypedDict, total=False):
        role: Required[str]
        audio: NotRequired[Any | None]
        content: NotRequired[str | list[Any] | None]
        function_call: NotRequired[FunctionCall | None]
        name: NotRequired[str]
        refusal: NotRequired[str | None]
        tool_calls: NotRequired[list[ChatCompletionMessageToolCallUnionParam]]

    class ToolMessageParam(TypedDict, total=False):
        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[str]
        tool_call_id: Required[str]

    chat_types.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam = DeveloperMessageParam
    chat_types.chat_completion_system_message_param.ChatCompletionSystemMessageParam = (
        SystemMessageParam
    )
    chat_types.chat_completion_user_message_param.ChatCompletionUserMessageParam = (
        UserMessageParam
    )
    chat_types.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam = AssistantMessageParam
    chat_types.chat_completion_tool_message_param.ChatCompletionToolMessageParam = (
        ToolMessageParam
    )
    chat_types.chat_completion_message_param.ChatCompletionMessageParam = (
        DeveloperMessageParam
        | SystemMessageParam
        | UserMessageParam
        | AssistantMessageParam
        | ToolMessageParam
        | ChatCompletionFunctionMessageParam
    )


def monkey_patch_chat_completion_logprobs() -> None:
    """Avoid expensive validation of large chat-completion logprob payloads."""
    try:
        from openai.types.chat import chat_completion
        from openai.types.chat.chat_completion import ChatCompletion, Choice
    except ImportError:
        return

    class ChoiceAny(Choice):
        logprobs: Any | None = None
        sampling_mask: list[list[int]] | None = None

    class ChatCompletionAnyLogprobs(ChatCompletion):
        choices: list[ChoiceAny]  # type: ignore[assignment]

    chat_completion.Choice = ChoiceAny
    chat_completion.ChatCompletion = ChatCompletionAnyLogprobs
