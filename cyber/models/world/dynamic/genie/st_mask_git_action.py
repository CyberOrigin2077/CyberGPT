import math

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from transformers.utils import ModelOutput

from genie.factorization_utils import FactorizedEmbedding
from genie.config import ActionGenieConfig
from genie.st_mask_git import STMaskGIT


def cosine_schedule(u):
    """ u in [0, 1] """
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")

    return cls.cos(u * cls.pi / 2)


class STMaskGITAction(STMaskGIT):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    # Action-Token appened as done in https://arxiv.org/pdf/2209.00588.pdf
    def __init__(self, config: ActionGenieConfig):
        super().__init__(config)
        self.A = config.A  # action token number
        self.mask_action_id = config.action_vocab_size
        self.pos_embed_TSC = nn.Parameter(torch.randn(1, config.T, config.S + config.A, config.d_model))
        self.action_embed = FactorizedEmbedding(    # this is the trainable action embedding
            factored_vocab_size=config.factored_action_size,
            num_factored_vocabs=config.num_factored_actions,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        input_actions: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        min_new_tokens: int = None,
        return_logits: int = False,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Args designed to match the format of Llama.
        We ignore `attention_mask`, and use `max_new_tokens` to determine the number of frames to generate.

        Returns: `(sample_THW, factored_logits)` if `return_logits` else `sample_THW`
            sample_THW: size (B, num_new_frames * H * W) corresponding to autoregressively generated
                unfactorized token ids for future frames.
            Optionally, factored_logits: size (B, factored_vocab_size, num_factored_vocabs, num_new_frames, H, W).
        """
        assert min_new_tokens in (None, max_new_tokens), \
            "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."

        assert max_new_tokens % self.config.S == 0, "Expecting `max_new_tokens` to be a multiple of `self.config.S`."
        num_new_frames = max_new_tokens // self.config.S

        inputs_THW = rearrange(input_ids.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w)
        inputs_masked_THW = torch.cat([
            inputs_THW,
            torch.full((input_ids.size(0), num_new_frames, self.h, self.w),
                       self.mask_token_id, dtype=torch.long, device=input_ids.device)
        ], dim=1)
        inputs_TA = rearrange(input_actions.clone(), "b (t a) -> b t a", a=self.A)
        inputs_masked_TA = torch.cat([
            inputs_TA,
            torch.full((input_actions.size(0), num_new_frames, self.A),
                       self.mask_action_id, dtype=torch.long, device=input_actions.device)
        ], dim=1)

        all_factored_logits = []
        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            # could change sampling hparams
            sample_HW, factored_logits = self.maskgit_generate(
                inputs_masked_THW,
                inputs_masked_TA,
                timestep,
                maskgit_steps=maskgit_steps,
                temperature=temperature
            )
            inputs_masked_THW[:, timestep] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W -> B (T H W)")
        if return_logits:
            return predicted_tokens, torch.stack(all_factored_logits, dim=3)  # (b, factored_vocab_size, num_factored_vocabs, num_new_frames, h, w)
        else:
            return predicted_tokens

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.LongTensor,
        prompt_TA: torch.Tensor,  # TODO: determine specific type
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Performs MaskGIT-style inference to predict frame `out_t`.

        Args:
            prompt_THW: Unfactorized token ids, size (B, T, H, W)
            out_t: Will return predicted unfactorized token ids for this frame.
                Should be >= 1 as the 0th frame is assumed to be given.
                Expects all future frames to be fully masked.
            maskgit_steps: The number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.
                In the factorized case, sampling is performed for each factorized vocabulary independently.
                If temperature is <= 1e-8, will be greedy (i.e. argmax) instead of actual sampling.
            unmask_mode: The method to determine tokens to unmask during each step of MaskGIT inference.
                Options:
                    - "greedy" for unmasking the most confident tokens, which is matches the original MaskGIT
                    - "random" for randomly choosing tokens to unmask
                "greedy" tends to copy the previous frame, so we default to "random" instead.

        Returns: (sample_HW, factored_logits)
            sample_HW: size (B, H, W) corresponding to predicted unfactorized token ids for frame `out_t`.
            factored_logits: size (B, factored_vocab_size, num_factored_vocabs, H, W).
        """
        # assume we have pre-masked z{out_t}...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), \
            f"when generating z{out_t}, frames {out_t} and later must be masked"

        bs, t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)

        logits_CTHW = self.compute_logits(prompt_THW, prompt_TA)
        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = logits_CHW.clone()  # Return these original logits, not logits after partially sampling.
        for step in tqdm(range(maskgit_steps)):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                logits_CHW = self.compute_logits(prompt_THW, prompt_TA)[:, :, out_t]

            factored_logits = rearrange(logits_CHW, "b (num_vocabs vocab_size) h w -> b vocab_size num_vocabs h w",
                                        vocab_size=self.config.factored_vocab_size,
                                        num_vocabs=self.config.num_factored_vocabs)

            factored_probs = torch.nn.functional.softmax(factored_logits, dim=1)

            samples_HW = torch.zeros((bs, h, w), dtype=torch.long, device=prompt_THW.device)
            confidences_HW = torch.ones((bs, h, w), dtype=torch.float, device=prompt_THW.device)
            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:  # greedy sampling
                    sample = probs.argmax(dim=1)
                else:
                    # Categorical expects last dim to be channel dim
                    dist = torch.distributions.categorical.Categorical(
                        probs=rearrange(probs, "b vocab_size ... -> b ... vocab_size") / temperature
                    )
                    sample = dist.sample()
                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")

            samples_flat = samples_HW.reshape(bs, self.config.S)

            if step != maskgit_steps - 1:  # skip masking for last maskgit step
                # use cosine mask scheduling function, n is how many of frame out_t to mask
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * self.config.S)

                if unmask_mode == "greedy":
                    # set the n patches with the least confidence to mask_token
                    confidences_flat = confidences_HW.reshape(bs, self.config.S)
                elif unmask_mode == "random":
                    # randomize confidences, so that patches are randomly masked
                    confidences_flat = torch.rand_like(confidences_HW).reshape(bs, self.config.S)
                    # not probability distribution anymore, but only relative order matters
                else:
                    raise NotImplementedError(f"Expected `unmask_mode` to be one of ['greedy', 'random'], "
                                              f"got {unmask_mode}")

                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                # unmask the (self.config.S - n) most confident tokens
                unmasked.scatter_(1, least_confident_tokens[:, n:], True)
                samples_flat.scatter_(1, least_confident_tokens[:, :n], self.mask_token_id)

            # copy previously unmasked values from prompt input into sample
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.reshape(-1, h, w)

            # feed back to iteratively decode
            prompt_THW[:, out_t] = samples_HW

        # Return the final sample and logits
        return samples_HW, rearrange(
            orig_logits_CHW, "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
            vocab_size=self.config.factored_vocab_size, num_vocabs=self.config.num_factored_vocabs, H=h, W=w
        )

    def compute_logits(self, x_THW, x_TA):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")
        x_TSC = self.token_embed(x_TS)
        x_TAC = self.action_embed(x_TA)
        x_TSaC = torch.cat([x_TSC, x_TAC], dim=2)
        # additive embeddings, using the same vocab space
        x_TSaC = self.decoder(x_TSaC + self.pos_embed_TSC)
        # remove the action embeddings e.g. z0, a0, z1, a1, ..., zT, aT -> z0, z1, ..., zT
        x_TSC = x_TSaC[:, :, :-self.config.A]
        x_next_TSC = self.out_x_proj(x_TSC)
        logits_CTHW = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=self.h, W=self.w)
        return logits_CTHW

    def forward(self, input_ids, input_actions, labels):
        T, H, W, A = self.config.T, self.h, self.w, self.A
        x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)
        # concat actions to the input in the format of [z0, a0, z1, a1, ..., zT, aT]
        x_TA = rearrange(input_actions, "B (T A) -> B T A", T=T, A=A)

        logits_CTHW = self.compute_logits(x_THW, x_TA)

        labels = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        relevant_mask = x_THW[:, 1:] == self.mask_token_id  # could also get mask of corrupted tokens by uncommenting line in `get_maskgit_collator`
        relevant_loss, relevant_acc = self.compute_loss_and_acc(logits_CTHW, labels, relevant_mask)

        return ModelOutput(loss=relevant_loss, acc=relevant_acc, logits=logits_CTHW)
