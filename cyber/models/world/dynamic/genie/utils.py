import torch

from genie.config import GenieConfig
from genie.st_mask_git import STMaskGIT
from genie.factorization_utils import factorize_token_ids, unfactorize_token_ids
from genie.config import GenieConfig
from genie.st_mask_git import cosine_schedule


from einops import rearrange
import random
import math


def convert_lightning_checkpoint(lightning_checkpoint, num_layers, num_heads, d_model, save_dir):
    """
    v0.0.1 saved models in Lightning checkpoints, this can convert Lightning checkpoints to HF checkpoints.
    """
    config = GenieConfig(num_layers=num_layers, num_heads=num_heads, d_model=d_model)
    model = STMaskGIT(config)

    lightning_checkpoint = torch.load(lightning_checkpoint)
    model_state_dict = lightning_checkpoint["state_dict"]

    # Remove `model.` prefix
    model_state_dict = {name.replace("model.", ""): params for name, params in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model.save_pretrained(save_dir)

def get_maskgit_collator(config):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T,
                          h=h, w=w)
        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.clone()

        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(),
                                      dtype=torch.long, device=device)
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

    return collate_fn


def get_maskgit_collator_with_action(config: GenieConfig):
    '''
    this function does the same thing as get_maskgit_collator but also includes the action tokens
    '''
    image_token_collate_fn = get_maskgit_collator(config)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        image_collated = image_token_collate_fn(features)
        return {
            **image_collated,
            "input_actions": torch.stack([ex["actions"] for ex in features])
        }

    return collate_fn
