import argparse
import torch
from Cogformer.attention import generate_attn_masks
from Cogformer.model import create_model
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import types

def new_forward(
        self,
        tokens,
        mask = None,
        past_key_values = None,
    ):
        h = self.embedding(tokens)

        for layer in self.layers:
            h, _, _ = layer(h, self.freqs_cis, mask, past_key_values)

        return self.norm(h)


@torch.inference_mode(mode=True)
def main(cfg):

    torch.set_default_device('cuda')
    
    # You need to get access to Llama 2 first
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side='left')# Tokenizer(cfg.tokenizer_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    name = cfg.ckpt.split('_ckpt')[-2].split('/')[-1]
    model = create_model(cfg.ckpt)
    
    model.forward = types.MethodType(new_forward, model)
    
    torch.manual_seed(0)
    
    length = [200,400,600,800,1000,2000] # 
    
    prompts = []
    rep_diff = {}
    for idx, l in enumerate(length):
        if cfg.task == 'count_ones':
            prompt = "How many ones are in the following sequences?"
            prompts.append(prompt + '1' * l + '1')
            prompts.append(prompt + '1' * l)
        elif cfg.task == 'find_a_zero':
            prompt = "Is there a zero in the following sequences?"
            prompts.append(prompt + '0' + '1' * l)
            prompts.append(prompt + '1' + '1' * l)

    encoded_kv = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True).to('cuda')
    input_ids = encoded_kv['input_ids']
    mask = generate_attn_masks(input_ids == 1)
    hidden = model(input_ids, mask)
    
    for idx, l in enumerate(length):    
        rep_diff[l] = torch.norm((hidden[idx * 2][-1] - hidden[idx * 2 + 1][-1]).view(12,64), p=float('inf'), dim=-1).mean()
    
    norm_dominator = rep_diff[length[0]]
    
    for key in rep_diff.keys():
        rep_diff[key] /= norm_dominator
    
    custom_labels = [str(i) for i in rep_diff.keys()]
    
    plt.xticks([x for x in range(len(rep_diff.keys()))], custom_labels)

    for x, t in enumerate(rep_diff.keys()):
        plt.bar(x, rep_diff[t].cpu(), width=0.8, color='tab:red' if 'cog' in name else 'tab:blue', edgecolor='black', linewidth=0.7)

    
    plt.xlabel('n')
    plt.ylabel('acc')
    plt.yticks([0.2 * i for i in range(6)])
    plt.legend()
    plt.savefig(f'rep-diff-{name}-{cfg.task}.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="ckpt")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--tokenizer_model_path', type=str, default="llama/tokenizer.model")
    args = parser.parse_args()

    main(args)