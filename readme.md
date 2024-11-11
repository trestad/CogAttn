## Official PyTorch implementation of "More Expressive Attention with Negative Weights"

TL;DR: We propose a novel attention mechanism, named Cog Attention, that enables attention weights to be negative for enhanced expressiveness.

Why named cog attention?
> 1. The attention pattern looks like cogs.
> 2. The transformation cog ("T-cog") and the living metal of each transformer's body allows them to change from their natural robotic body into an "alternate mode" based on something a form of technology or life form that they've observed and scanned. —— [Wikipedia](https://en.wikipedia.org/wiki/Autobot). In summary, the cog enhances the expressiveness of Transformers :)

## Dependencies
Create a conda environment and install required packages:

    conda create -n cog python=3.11.9
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
    pip install -e .
    cd ..
    
    pip install sentencepiece==0.2.0, transformers==4.44.2, rich==13.9.4, matplotlib, einops, plotly, fancy_einsum 
    pip install -U kaleido

    cd lm-evaluation-harness/
    pip install -e .
    cd ..

## Checkpoints

We are training larger language models and will update the paper and release checkpoints ASAP.

| Models | Link |
|--|--|
|Cogformer (141M LM)|[Link](https://drive.google.com/file/d/1Ph3d-Yc1SED2JrYdWZw4bJPWeQPU_7iR/view?usp=drive_link)|
|Transformer (141M LM)|[Link](https://drive.google.com/file/d/1soPHh4SePXwRjBAVrcZjO8q8QYBBtqNY/view?usp=drive_link)|
|U-ViC-S/2 MS-COCO|[Link](https://drive.google.com/file/d/1zl-57yGaY_qC6iC_iWdDIgNqkIuQyZlo/view?usp=drive_link)|
|U-ViT-S/2 MS-COCO|[Link](https://drive.google.com/file/d/1Fo6_9xns5ScJitHLELXgHniGnTBpDwuB/view?usp=drive_link)|



## Reproducing Experiments

Place downloaded language model checkpoints in the `CogAttn/checkpoints` folder.

### To reproduce figure 1
    cd Easy-Transformer
    python interpret.py --ckpt ../checkpoints/vanilla_ckpt_100.00_Bi.pt
    python interpret.py --ckpt ../checkpoints/cog_ckpt_100.00_Bi.pt
    cd ..

Plotted figures will be saved in the Easy-Transformer folder.

### To reproduce figure 5

    python draw_rep_dif.py --ckpt checkpoints/vanilla_ckpt_100.00_Bi.pt --task count_ones
    python draw_rep_dif.py --ckpt checkpoints/vanilla_ckpt_100.00_Bi.pt --task find_a_zero
    python draw_rep_dif.py --ckpt checkpoints/cog_ckpt_100.00_Bi.pt --task count_ones
    python draw_rep_dif.py --ckpt checkpoints/cog_ckpt_100.00_Bi.pt --task find_a_zero

### To reproduce figure 9 and 10

    python draw_attn.py --ckpt checkpoints/cog_ckpt_100.00_Bi.pt
    python draw_attn.py --ckpt checkpoints/vanilla_ckpt_100.00_Bi.pt

### Language modeling

    lm_eval --model my --model_args ckpt=checkpoints/vanilla_ckpt_100.00_Bi.pt --tasks arc_easy,arc_challenge,piqa,mnli,mrpc,qqp,rte --device cuda:0 --batch_size 1 --trust_remote_code &> vanilla-100B.output &
    lm_eval --model my --model_args ckpt=checkpoints/cog_ckpt_100.00_Bi.pt --tasks arc_easy,arc_challenge,piqa,mnli,mrpc,qqp,rte --device cuda:0 --batch_size 1 --trust_remote_code &> cog-100B.output &

 
 ### Image Generation

1. Clone the [U-ViT repository](https://github.com/baofff/U-ViT), place files in the `U-ViC` folder accordingly.
2. Download U-ViC checkpoints. 
3. Download "reference statistics for FID", preprocess MS-COCO following U-ViT's readme.
4. Then run:

    
    CUDA_VISIBLE_DEVICES=0,1,2,3, accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small_np_0_last.py --nnet_path=YOUR_PATH_SAVE_MY_CKPT/cog_nnet_ema.pth &> cfg_mscoco_sample_cog.log &

Two generation log files have been available in the U-ViC folder.


### Acknowledgements
Our code is based on
* [U-ViT](https://github.com/baofff/U-ViT)
* [LM-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [Easy-Transformer](https://github.com/redwoodresearch/Easy-Transformer) and [Some of my previous code](https://github.com/trestad/Factual-Recall-Mechanism)
* [fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2/fairseq)

We thank all these open-source projects for their contributions.