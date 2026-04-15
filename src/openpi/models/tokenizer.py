import logging

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download

END_OF_PREFIX_TOKEN = 257022
BEGIN_OF_ACTION = 257021
BEGIN_OF_REASONING = 257020
PALIGEMMA_EOS_TOKEN = 1

class FusePaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        path = download.maybe_download("/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self,
                 thought: list[str],
                 act_with_outdated_thought: bool,
                 think_with_outdated_thought: bool,
                 ) -> tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray, np.ndarray]:
        prefix = thought[0]
        prefix_tokens = (
            self._tokenizer.encode(prefix, add_bos=True) +
            [END_OF_PREFIX_TOKEN]
        )

        if len(thought) > 1:
            suffix = thought[1]
            suffix_tokens = [BEGIN_OF_REASONING] + self._tokenizer.encode(suffix, add_eos=True)
            diffusion_loss_mask = np.False_
        else:
            suffix_tokens = [BEGIN_OF_ACTION]
            diffusion_loss_mask = np.True_
        tokens = prefix_tokens + suffix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(suffix_tokens)
        if think_with_outdated_thought:
            text_loss_mask = [False] * len(prefix_tokens) + [True] + [False] * (len(suffix_tokens) - 1)
        else:
            text_loss_mask = (
                [False] * len(prefix_tokens) +
                # we should not supervise the <BEGIN_OF_ACTION> token in this case
                [not act_with_outdated_thought] +  
                [True] * (len(suffix_tokens) - 1)
            )
        
        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            text_loss_mask = text_loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            text_loss_mask = text_loss_mask[: self._max_len]
        
        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(text_loss_mask),
            diffusion_loss_mask,
        )
    
    def extract_thoughts(self, tokens: np.ndarray) -> str:
        tokens = tokens.tolist()
        filtered_tokens = []
        # skip the first token, which is the BOA/BOT token
        for t in tokens[1:]:
            filtered_tokens.append(t)
            if t == PALIGEMMA_EOS_TOKEN:
                break
        return self._tokenizer.decode(filtered_tokens)


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        path = download.maybe_download("/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


# class FuseFASTTokenizer:
#     """Combines FusePaligemmaTokenizer prefix/suffix structure with FAST action tokenization."""

#     def __init__(self, max_len: int = 400, fast_tokenizer_path: str = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/hf_cache/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4"):
#         self._max_len = max_len

#         path = download.maybe_download("/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model", gs={"token": "anon"})
#         with path.open("rb") as f:
#             self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

#         self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
#         self._fast_skip_tokens = 128

#     def tokenize(
#         self,
#         thought: list[str],
#         state: np.ndarray,
#         actions: np.ndarray | None,
#         act_with_outdated_thought: bool,
#         think_with_outdated_thought: bool,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         prefix = thought[0]
#         discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
#         state_str = " ".join(map(str, discretized_state))
#         prefix_with_state = f"{prefix} State: {state_str};\n"
#         prefix_tokens = (
#             self._paligemma_tokenizer.encode(prefix_with_state, add_bos=True)
#             + [END_OF_PREFIX_TOKEN]
#         )

#         if len(thought) > 1:
#             suffix = thought[1]
#             suffix_tokens = [BEGIN_OF_REASONING] + self._paligemma_tokenizer.encode(suffix, add_eos=True)
#             diffusion_loss_mask = np.False_
#         else:
#             if actions is not None:
#                 action_tokens = self._fast_tokenizer(actions[None])[0]
#                 action_tokens_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
#                 suffix_tokens = (
#                     [BEGIN_OF_ACTION]
#                     + self._paligemma_tokenizer.encode("Action: ")
#                     + action_tokens_pg.tolist()
#                     + self._paligemma_tokenizer.encode("|")
#                 )
#             else:
#                 suffix_tokens = [BEGIN_OF_ACTION]
#             diffusion_loss_mask = np.True_

#         tokens = prefix_tokens + suffix_tokens
#         token_mask = [True] * len(tokens)
#         ar_mask = [0] * len(prefix_tokens) + [1] * len(suffix_tokens)
#         if think_with_outdated_thought:
#             text_loss_mask = [False] * len(prefix_tokens) + [True] + [False] * (len(suffix_tokens) - 1)
#         else:
#             text_loss_mask = (
#                 [False] * len(prefix_tokens)
#                 + [not act_with_outdated_thought]
#                 + [True] * (len(suffix_tokens) - 1)
#             )

#         tokens_len = len(tokens)
#         if tokens_len < self._max_len:
#             padding = [False] * (self._max_len - tokens_len)
#             tokens = tokens + padding
#             token_mask = token_mask + padding
#             ar_mask = ar_mask + padding
#             text_loss_mask = text_loss_mask + padding
#         else:
#             if len(tokens) > self._max_len:
#                 logging.warning(
#                     f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
#                     "Consider increasing the `max_token_len` in your model config if this happens frequently."
#                 )
#             tokens = tokens[: self._max_len]
#             token_mask = token_mask[: self._max_len]
#             ar_mask = ar_mask[: self._max_len]
#             text_loss_mask = text_loss_mask[: self._max_len]

#         return (
#             np.asarray(tokens),
#             np.asarray(token_mask),
#             np.asarray(ar_mask),
#             np.asarray(text_loss_mask),
#             diffusion_loss_mask,
#         )

#     def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
#         decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
#         if "Action: " not in decoded_tokens:
#             return np.zeros((action_horizon, action_dim), dtype=np.float32)
#         raw_action_tokens = np.array(
#             self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
#         )
#         action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
#         return self._fast_tokenizer.decode(
#             [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
#         )[0]

#     def extract_thoughts(self, tokens: np.ndarray) -> str:
#         tokens = tokens.tolist()
#         filtered_tokens = []
#         for t in tokens[1:]:
#             filtered_tokens.append(t)
#             if t == PALIGEMMA_EOS_TOKEN:
#                 break
#         return self._paligemma_tokenizer.decode(filtered_tokens)

#     def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
#         if isinstance(tokens, list):
#             tokens = np.array(tokens)
#         return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens



# add for vqa tokenizer
class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "/inspire/hdd/global_user/gongjingjing-25039/sdzhang/hf_cache/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4",
    ):
        self._max_len = max_len

        path = download.maybe_download(
            "/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )
        self._fast_skip_tokens = 128

    def _get_pad_id(self) -> int:
        pad_id = self._paligemma_tokenizer.pad_id()
        if pad_id is None or pad_id < 0:
            pad_id = 0
        return pad_id

    def _clean_text(self, text: str) -> str:
        text = text.replace("<image>", " ").replace("<IMAGE>", " ")
        text = text.replace("_", " ").replace("\n", " ")
        text = " ".join(text.strip().split())
        return text.lower()

    def _discretize_state(self, state: np.ndarray) -> np.ndarray:
        return np.digitize(
            state, bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1

    def _pad_or_truncate(
        self,
        tokens: list[int],
        token_mask: list[bool],
        ar_mask: list[int],
        loss_mask: list[bool],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pad_id = self._get_pad_id()
        tokens_len = len(tokens)

        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            tokens = tokens + [pad_id] * pad_len
            token_mask = token_mask + [False] * pad_len
            ar_mask = ar_mask + [0] * pad_len
            loss_mask = loss_mask + [False] * pad_len
        else:
            if tokens_len > self._max_len:
                logging.warning(
                    f"Token length ({tokens_len}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing max_token_len if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(token_mask, dtype=bool),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask, dtype=bool),
        )

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = self._clean_text(prompt)

        discretized_state = self._discretize_state(state)
        state_str = " ".join(map(str, discretized_state))

        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|")
            )
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        return self._pad_or_truncate(tokens, token_mask, ar_mask, loss_mask)

    def tokenize_vqa(
        self,
        question: str,
        state: np.ndarray,
        answer: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_question = self._clean_text(question)
        cleaned_answer = " ".join(answer.strip().split())

        discretized_state = self._discretize_state(state)
        state_str = " ".join(map(str, discretized_state))

        # prefix
        prefix = f"Task: {cleaned_question}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        # answer template
        answer_prefix_tokens = self._paligemma_tokenizer.encode("Answer: ")
        answer_text_tokens = self._paligemma_tokenizer.encode(cleaned_answer)
        eos_tokens = [PALIGEMMA_EOS_TOKEN]

        postfix_tokens = answer_prefix_tokens + answer_text_tokens + eos_tokens

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)

        # 只对 answer 文本 + EOS 算 loss，不对 "Answer: " 算 loss
        loss_mask = (
            [False] * len(prefix_tokens)
            + [False] * len(answer_prefix_tokens)
            + [True] * len(answer_text_tokens)
            + [True] * len(eos_tokens)
        )

        return self._pad_or_truncate(tokens, token_mask, ar_mask, loss_mask)

    def extract_actions(
        self,
        tokens: np.ndarray,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split("Action: ")[1].split("|")[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )[0]

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens