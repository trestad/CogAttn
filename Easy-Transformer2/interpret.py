import torch
import argparse
assert torch.cuda.device_count() == 1
from tqdm import tqdm
import torch
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
from tqdm import tqdm

from easy_transformer.ioi_dataset import (
    IOIDataset,
    NAMES,
)
from easy_transformer.ioi_utils import (
    path_patching,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)

from easy_transformer.ioi_utils import logit_diff


def main(args):

    ckpt = args.ckpt

    if 'cog' in ckpt:
        model_name = 'cog'
    elif 'vanilla':
        model_name = 'vanilla'
    else:
        raise NotImplemented

    model = EasyTransformer.from_pretrained(ckpt).cuda()
    model.set_use_headwise_qkv_input(True)
    model.set_use_attn_result(True)

    N = 100
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=True,
    )  # TODO make this a seeded dataset

    for i in range(N):
        io_idx = ioi_dataset.word_idx['IO'][i].item()
        s_idx = ioi_dataset.word_idx['S'][i].item()
        s2_idx = ioi_dataset.word_idx['S2'][i].item()
        
        names = ioi_dataset.tokenized_prompts[i].split('|')
        assert names[s_idx] in NAMES
        assert names[io_idx] in NAMES
        assert names[s_idx] == names[s2_idx], names
        assert names[s_idx] != names[io_idx], names
        
    print('data checked')
        
    print(f"Here are two of the prompts from the dataset: {ioi_dataset.sentences[:2]}")

    # we make the ABC dataset in order to knockout other model components
    abc_dataset = (  # TODO seeded
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
        .gen_flipped_prompts(("S", "RAND"))
        .gen_flipped_prompts(("S1", "RAND"))
    )


    def plot_path_patching(
        model,
        ioi_dataset,
        receiver_hooks,  # list of tuples (hook_name, idx). If idx is not None, then at dim 2 index in with idx (used for doing things for specific attention heads)
        position,
    ):
        # input(ckpt)
        model.reset_hooks()
        default_logit_diff = logit_diff(model, ioi_dataset)
        results = torch.zeros(size=(12, 12))
        for source_layer in tqdm(range(12)):
            for source_head_idx in tqdm(range(12)):
                model.reset_hooks()

                model = path_patching(
                    model=model,
                    D_new=abc_dataset,
                    D_orig=ioi_dataset,
                    sender_heads=[(source_layer, source_head_idx)],
                    receiver_hooks=receiver_hooks,
                    positions=[position],
                    return_hooks=False,
                    freeze_mlps=False,
                    have_internal_interactions=False,
                )
                cur_logit_diff = logit_diff(model, ioi_dataset)

                
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - default_logit_diff
                )

                # if source_layer == 1:
                assert not torch.allclose(results, 0.0 * results), results

                if source_layer == 11 and source_head_idx == 11:
                    results /= default_logit_diff
                    results *= 100
                    # print(results)
                    # show attention head results
                    fig = show_pp(
                        results,
                        title=f"Effect of patching (Heads->Final Residual Stream State) path",
                        return_fig=True,
                        show_fig=False,
                        bartitle="% change in logit difference",
                    )
                    fig.show()
                    fig.write_image(f"ioi_patch_heads_{model_name}.png") #, scale=6, width=1080, height=1080)

    with torch.no_grad():
        plot_path_patching(
            model,
            ioi_dataset,
            receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
            position="end",
        )


    layer_no = 9
    if model_name == 'cog':
        head_no=4
    else:
        head_no=11


    model.reset_hooks()

    show_attention_patterns(model, [(layer_no, head_no)], ioi_dataset[:1], mode='attn', title_suffix=model_name)
        
    model.reset_hooks()

    scatter_attention_and_contribution(
        model=model, layer_no=layer_no, head_no=head_no, ioi_dataset=ioi_dataset, suffix=model_name,
    )

    model.reset_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    main(args)