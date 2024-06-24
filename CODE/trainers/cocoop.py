import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# For Augmented
from tqdm import tqdm
from torchvision.transforms import functional as F_v2
from torchvision.transforms import v2 as T_v2
import random
from ultralytics import YOLO

# Set the transformation strength hyperparameter
INTENSITY = 0.1
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE', 'SHEAR', 'RANDOM_PERSPECTIVE',
#                              'RANDOM_EQUALIZE', 'RANDOM_ERASING', 'COLOR_JITTER', 'ANIME_GAN']  # caltech101
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE', 'SHEAR', 'RANDOM_PERSPECTIVE',
#                              'RANDOM_EQUALIZE', 'COLOR_JITTER']  # oxford_pets food101
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE', 'RANDOM_EQUALIZE', 'COLOR_JITTER']  # else(except cars)
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'TRANSLATE', 'RANDOM_EQUALIZE', 'COLOR_JITTER']  # 0504
# AUGMENTED_TRANSFORMATIONS = [] # 0505
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'COLOR_JITTER']  # 0506
# AUGMENTED_TRANSFORMATIONS = ['ROTATION', 'TRANSLATE', 'RANDOM_EQUALIZE', 'COLOR_JITTER', 'HORIZONTAL_FLIP']  # 0508
# AUGMENTED_TRANSFORMATIONS = ['ROTATION', 'RANDOM_PERSPECTIVE', 'RANDOM_ERASING']  # 0510
# AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'RANDOM_EQUALIZE', 'COLOR_JITTER', 'ANIME_GAN']  # 0513
# ['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE', 'SHEAR', 'RANDOM_PERSPECTIVE', 'CROP', 'RANDOM_ERASING',
#                              'RANDOM_EQUALIZE', 'COLOR_JITTER', 'ANIME_GAN']

AUGMENTED_TRANSFORMATIONS = ['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE']

'''
    transform_functions = {
        'HORIZONTAL_FLIP': self.horizontal_flip,
        'ROTATION': self.rotate,
        'TRANSLATE': self.translate,
        'SHEAR': self.shear,
        'RANDOM_PERSPECTIVE': self.random_perspective,
        'CROP': self.crop,
        'RANDOM_EQUALIZE': self.random_equalize,
        'RANDOM_ERASING': self.random_erasing,
        'COLOR_JITTER': self.color_jitter,
        'ANIME_GAN': self.anime_gan
    }
'''


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # For Augmented
        x = prompts + self.positional_embedding.type(self.dtype)[:prompts.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # For Augmented
        self.cfg = cfg
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # For Augmented
        # random initialization
        n_aid_ctx = len(AUGMENTED_TRANSFORMATIONS) + 1  # 1=original_input
        aid_ctx_vectors = torch.empty(n_aid_ctx, ctx_dim, dtype=dtype)  # can be deleted
        nn.init.normal_(aid_ctx_vectors, std=0.02)  # can be deleted
        aid_prompt_prefix = " ".join(["X"] * n_aid_ctx)

        print(f'Aid context: "{aid_prompt_prefix}"')
        print(f"Number of aid context words (tokens): {n_aid_ctx}")

        aid_prompts = [aid_prompt_prefix + " " + name + "." for name in classnames]

        aid_tokenized_prompts = torch.cat([clip.tokenize(p) for p in aid_prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            aid_embedding = clip_model.token_embedding(aid_tokenized_prompts).type(dtype)

        self.register_buffer("aid_token_prefix", aid_embedding[:, :1, :])  # SOS
        self.register_buffer("aid_token_suffix", aid_embedding[:, 1 + n_aid_ctx:, :])  # CLS, EOS

        self.aid_tokenized_prompts = aid_tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, features_list):
        prefix = self.token_prefix  # torch.Size([number_cls, 1, 512])
        suffix = self.token_suffix  # torch.Size([number_cls, *, 512])
        ctx = self.ctx  # (n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx.expand(self.cfg.DATALOADER.TEST.BATCH_SIZE, -1, -1)  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        # For Augmented
        aid_prefix = self.aid_token_prefix  # torch.Size([number_cls, 1, 512])
        aid_suffix = self.aid_token_suffix  # torch.Size([number_cls, *, 512])

        aid_ctx_list = []
        for feature in features_list:
            aid_ctx = self.meta_net(feature)  # (batch, ctx_dim)
            aid_ctx = aid_ctx.unsqueeze(1)  # (batch, 1, ctx_dim)
            aid_ctx_list.append(aid_ctx)

        aid_ctx = torch.cat(aid_ctx_list, dim=1)  # (batch, num_features, ctx_dim)

        aid_prompts = []
        for aid_ctx_i in aid_ctx:
            ctx_i_expanded = aid_ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)  # (n_cls, 2, ctx_dim)
            aid_pts_i = self.construct_prompts(ctx_i_expanded, aid_prefix, aid_suffix)  # (n_cls, n_tkn, ctx_dim)
            aid_prompts.append(aid_pts_i)
        aid_prompts = torch.stack(aid_prompts)

        return prompts, aid_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.aid_tokeinzed_prompts = self.prompt_learner.aid_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_list, label=None):
        tokenized_prompts = self.tokenized_prompts
        aid_tokenized_prompts = self.aid_tokeinzed_prompts
        logit_scale = self.logit_scale.exp()

        # Process all images in the list
        image_features_list = []
        for img in image_list:
            img_features = self.image_encoder(img.type(self.dtype))
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            image_features_list.append(img_features)

        # Assuming the number of images is always 2 for simplicity (image and yolo_view_image)
        # This part should be adapted if the list can have a different size or structure.
        prompts, aid_prompts = self.prompt_learner(image_features_list)

        logits = []
        for pts_i, aid_pts_i, imf_i in zip(prompts, aid_prompts, image_features_list[0]):
            origin_text_features = self.text_encoder(pts_i, tokenized_prompts)
            origin_text_features = origin_text_features / origin_text_features.norm(dim=-1, keepdim=True)

            # For Augmented
            aid_text_features = self.text_encoder(aid_pts_i, aid_tokenized_prompts)
            aid_text_features = aid_text_features / aid_text_features.norm(dim=-1, keepdim=True)

            alpha = 1.0
            text_features = alpha * origin_text_features + alpha * aid_text_features  # alpha * origin_text_features + (1-alpha) * aid_text_features

            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    # animeGAN = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
    yolov8_model = YOLO('yolov8n.pt')

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image_list, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                # For Augmented
                loss = model(image_list, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            # For Augmented
            loss = model(image_list, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        # For Augmented
        transform_functions = {
            'HORIZONTAL_FLIP': self.horizontal_flip,
            'ROTATION': self.rotate,
            'TRANSLATE': self.translate,
            'SHEAR': self.shear,
            'RANDOM_PERSPECTIVE': self.random_perspective,
            'RANDOM_EQUALIZE': self.random_equalize,
            'RANDOM_ERASING': self.random_erasing,
            'COLOR_JITTER': self.color_jitter,
            'ANIME_GAN': self.anime_gan,
            'CROP': self.crop
        }

        image_list = []
        input = batch["img"]  # [batch_size, RGB_chanel, height, width] = [1, 3, 224, 224]
        input = self.normalize_batch_images(input)
        input = input.to(self.device)
        image_list.append(input)
        for transformation in AUGMENTED_TRANSFORMATIONS:
            if transformation in transform_functions:
                if transformation == 'ANIME_GAN' or 'CROP':
                    transformed_images = transform_functions[transformation](batch["img"])
                else:
                    transformed_images = torch.stack([transform_functions[transformation](img) for img in batch["img"]])
                transformed_images = self.normalize_batch_images(transformed_images)
                transformed_images = transformed_images.to(self.device)

                image_list.append(transformed_images)

        label = batch["label"]
        label = label.to(self.device)

        return image_list, label

    def model_inference(self, input_list):
        # For Augmented
        return self.model(input_list)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # For Augmented
            input_list, label = self.parse_batch_test(batch)
            output = self.model_inference(input_list)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def parse_batch_test(self, batch):
        # For Augmented
        transform_functions = {
            'HORIZONTAL_FLIP': self.horizontal_flip,
            'ROTATION': self.rotate,
            'TRANSLATE': self.translate,
            'SHEAR': self.shear,
            'RANDOM_PERSPECTIVE': self.random_perspective,
            'RANDOM_EQUALIZE': self.random_equalize,
            'RANDOM_ERASING': self.random_erasing,
            'COLOR_JITTER': self.color_jitter,
            'ANIME_GAN': self.anime_gan,
            'CROP': self.crop
        }

        image_list = []
        input = batch["img"]  # [batch_size, RGB_chanel, height, width] = [1, 3, 224, 224]
        input = self.normalize_batch_images(input)
        input = input.to(self.device)
        image_list.append(input)
        for transformation in AUGMENTED_TRANSFORMATIONS:
            if transformation in transform_functions:
                if transformation == 'ANIME_GAN' or 'CROP':
                    transformed_images = transform_functions[transformation](batch["img"])
                else:
                    transformed_images = torch.stack([transform_functions[transformation](img) for img in batch["img"]])
                transformed_images = self.normalize_batch_images(transformed_images)
                transformed_images = transformed_images.to(self.device)

                image_list.append(transformed_images)

        label = batch["label"]
        label = label.to(self.device)

        return image_list, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            # Ignore fixed token vectors
            if "aid_token_prefix" in state_dict:
                del state_dict["aid_token_prefix"]

            if "aid_token_suffix" in state_dict:
                del state_dict["aid_token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def normalize_batch_images(self, images_tensor):
        """
        A batch of images is normalized.

        Parameters:
        -images_tensor: image tensor of shape (batch_size, channels, height, width)
        -pixel_mean: A list of channel means in the shape [mean_red, mean_green, mean_blue]
        -pixel_std: List of standard deviations of channels in the shape [std_red, std_green, std_blue]

        Returns:
        - Normalized image tensor
        """
        # For Augmented
        PIXEL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        PIXEL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

        normalized_images = (images_tensor - PIXEL_MEAN) / PIXEL_STD
        return normalized_images

    '''
        For Augmented
        Data Augmented Function
    '''

    def horizontal_flip(self, image_tensor):
        return T_v2.RandomHorizontalFlip(p=1.0)(image_tensor)  # The flip probability is 1.0

    def translate(self, image_tensor, intensity=INTENSITY):
        max_translation = int(intensity * 224)  # Max shear Angle
        translate_x = int(random.uniform(-max_translation, max_translation))
        translate_y = int(random.uniform(-max_translation, max_translation))
        return F_v2.affine(image_tensor, angle=0, translate=[translate_x, translate_y], scale=1.0, shear=[0.0, 0.0],
                           fill=[0, ])

    def rotate(self, image_tensor, intensity=INTENSITY):
        max_angle = intensity * 180  # max rotation degree
        return T_v2.RandomRotation(degrees=(-max_angle, max_angle))(image_tensor)

    def shear(self, image_tensor, intensity=INTENSITY):
        max_shear = intensity * 30
        shear_x = random.uniform(-max_shear, max_shear)
        shear_y = random.uniform(-max_shear, max_shear)
        return F_v2.affine(image_tensor, angle=0, translate=[0, 0], scale=1.0, shear=[shear_x, shear_y], fill=[0, ])

    def random_perspective(self, image_tensor, intensity=INTENSITY):
        distortion_scale = intensity  # Maximum distortion ratio
        return T_v2.RandomPerspective(distortion_scale=distortion_scale, p=1.0)(image_tensor)

    def crop(self, input_tensor):
        """
        Processes the input image and returns the modified image if any regions are detected;
        otherwise, returns the input image.

        :param input_tensor: Input image as a torch.Tensor with shape [1, 3, 224, 224].
        :return: Modified image as a torch.Tensor with the same shape if detections are made;
                 otherwise, the input tensor.

        Refer to https://docs.ultralytics.com/zh/modes/predict/#__tabbed_2_7
        """
        # Add for Multi_View
        # Perform inference on the batch
        results = self.yolov8_model(input_tensor)

        processed_images = []
        for i in range(len(input_tensor)):
            max_area = 0
            max_box = None

            boxes = results[i].boxes.xyxy  # Get all the bounding boxes detected
            for box in boxes:
                # box=[x1, y1, x2, y2, confidence, class]
                x1, y1, x2, y2 = map(float, box[:4])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_box = box[:4]  # Save the box with the largest area

            if max_box is not None:
                # Crop and resize the image based on the largest detected box
                box = [int(coord) for coord in max_box]
                cropped_img = input_tensor[i:i + 1, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                resized_img = F.interpolate(cropped_img, size=(224, 224), mode='bicubic', align_corners=False)
                processed_images.append(resized_img)
            else:
                processed_images.append(input_tensor[i:i + 1])

        # Concatenate all processed images into a single tensor
        return torch.cat(processed_images, dim=0)

    def random_erasing(self, image_tensor, intensity=INTENSITY):
        return T_v2.RandomErasing(p=1.0, scale=(0.02, INTENSITY), value="random")(image_tensor)

    def random_equalize(self, image_tensor):
        return T_v2.RandomEqualize(p=1.0)(image_tensor)

    def color_jitter(self, image_tensor, intensity=INTENSITY):
        # Adjust brightness, contrast, saturation, and hue
        brightness = contrast = saturation = hue = intensity
        return T_v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(image_tensor)

    def anime_gan(self, image_tensor):
        return self.animeGAN(image_tensor)  # BCHW tensor



