import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="ViT-B/16"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# parser = argparse.ArgumentParser()
# parser.add_argument("fpath", type=str, help="Path to the learned prompt")
# parser.add_argument("topk", type=int, help="Select top-k similar words")
# args = parser.parse_args()

# fpath = "C:/Jupyter/MyConda/Experiment/1_CoOp_/Augmented/output/0508/base2new/train_base/eurosat/shots_16/CoCoOp/vit_b16_c16_ep10_batch1/seed2/prompt_learner/model.pth.tar-10"  # args.fpath
fpath = "C:/Jupyter/MyConda/Experiment/1_CoOp_/Multi_View/output/all/train/fgvc_aircraft/CoCoOp/vit_b16_c16_ep10_batch1_ctxv1_16shots/seed3/prompt_learner/model.pth.tar-10"
topk = 1 # args.topk

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m+1}: {words} {dist}")

elif ctx.dim() == 3:
    # Class-specific context
    raise NotImplementedError

# import torch
#
# # 尝试加载文件
# try:
#     checkpoint = torch.load(fpath, map_location='cpu')
#     print("Checkpoint keys:", checkpoint.keys())
# except Exception as e:
#     print("Error loading checkpoint:", e)
