import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from Cogformer.model import create_model
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from Cogformer.attention import generate_attn_masks

@torch.inference_mode(mode=True)
def main(cfg):

    torch.set_default_device('cuda')

    # You need to get access to Llama 2 first
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side='left', add_bos_token=False)# Tokenizer(cfg.tokenizer_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    ckpt = cfg.ckpt
    name = ckpt.split('_ckpt')[-2].split('/')[-1]
    print(name)
            
    fig, axs = plt.subplots(12, 12, figsize=(48, 48,))

    model = create_model(ckpt)
        
    encoded_kv = tokenizer.batch_encode_plus(['The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.'], 
                                                return_tensors="pt").to('cuda')
    
    input_ids = encoded_kv['input_ids'][:, :1000]
    mask = generate_attn_masks(input_ids == 1)
    
    _, all_attn_w, _ = model(input_ids, mask,)
    
    for layer_idx in range(12):
        for head_idx in range(12):
            attn_weights = all_attn_w[layer_idx][0, head_idx, -1].cpu().numpy()
            
            # sorted_indices = np.argsort(attn_weights)
            # attn_weights = attn_weights[sorted_indices]

            x = np.arange(len(attn_weights))
            axs[layer_idx, head_idx].bar(x, attn_weights, color='blue')
            axs[layer_idx, head_idx].set_title(f'L{layer_idx}H{head_idx}')
            
    plt.yticks(rotation=45)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.savefig(f'attn-{name}.png')
    plt.cla()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    main(args)