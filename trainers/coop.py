import os.path as osp

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
import json

_tokenizer = _Tokenizer()


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
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextEncoder2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
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
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        print('classnamesclassnames:',len(classnames)) #1000
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print('prompt_prefix',prompt_prefix) # X X X X X X X X X X X X X X X X

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        #print('prompts:',prompts.size()) # torch.Size([100, 77, 512])
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.encode_text = TextEncoder2(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        #print(image_features.size()) #torch.Size([32, 1024])
        #text_features_ori = self.encode_text(text_tensor)
        #print('text_features_ori:',text_features_ori.size()) #torch.Size([32, 1024])

        prompts = self.prompt_learner()
        #print('prompts',prompts.size()) #torch.Size([100, 77, 512])
        tokenized_prompts = self.tokenized_prompts 
        #print('tokenized_prompts',tokenized_prompts.size()) #tokenized_prompts torch.Size([100, 77])
        text_features = self.text_encoder(prompts, tokenized_prompts)
        #print('text_features',text_features.size()) #text_features torch.Size([100, 1024])
        #torch.save(text_features, './mytensor2_3.pt')
        #print('image_features before:',image_features.size()) #torch.Size([25, 1024])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print('image_features after:',image_features.size()) #torch.Size([25, 1024]) caltech torch.Size([8, 1024])
        # print('text_features.size():',text_features.size()) #caltech torch.Size([100, 1024])
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
 

        # text_features_ori = text_features_ori / text_features_ori.norm(dim=-1, keepdim=True)
        # logits_per_image_ori = logit_scale * image_features @ text_features_ori.t()
        # logits_per_text_ori = logits_per_image_ori.t()


        return logits#, logits_per_image_ori, logits_per_text_ori


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    


    def forward_backward(self, batch):
        image, label= self.parse_batch_train(batch)
        #print('len(batch)len(batch)len(batch):',batch['image'].size())
        
        prec = self.cfg.TRAINER.COOP.PREC
        loss_set = self.cfg.TRAINER.COOP.LOSS
        #print('precprecprecprec:',prec) #fp16
        if prec == "amp":
            with autocast():
                output, logits_text = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if loss_set == "cross_entropy":
                output, logits_per_image_ori, logits_per_text_ori = self.model(image, text_tensor)
                # output, logits_text = self.model(image)
                #print('output.size():',output.size()) # torch.Size([32, 100])
                #print('logits_text.size():',logits_text.size()) # torch.Size([100, 32])
                #print('label.size()',label) #torch.Size([32])
                loss1 = F.cross_entropy(output, label)
                loss2 = loss1
                loss = loss1
                self.model_backward_and_update(loss)

            elif loss_set == "cross_contrastive":
                output, logits_per_image_ori, logits_per_text_ori = self.model(image, text_tensor)
                # output, logits_text = self.model(image)
                #print('output.size():',output.size()) # torch.Size([32, 100])
                #print('logits_text.size():',logits_text.size()) # torch.Size([100, 32])
                #print('label.size()',label) #torch.Size([32])
                loss1 = F.cross_entropy(output, label)

                ground_truth = torch.arange(32,dtype=torch.long,device=self.device)
                loss2 =  (F.cross_entropy(logits_per_image_ori,ground_truth) +  F.cross_entropy(logits_per_text_ori,ground_truth))/2
                #loss2 =  F.cross_entropy(output,ground_truth)
                loss = loss1 + loss2
                self.model_backward_and_update(loss)

            elif loss_set == "unicl_loss":
                # #print(label)
                # output, logits_per_image_ori, logits_per_text_ori = self.model(image, text_tensor)
                # logits_per_image_ori, logits_per_text_ori = logits_per_image_ori.type(torch.DoubleTensor), logits_per_text_ori.type(torch.DoubleTensor)
                # print(logits_per_image_ori.size(),logits_per_text_ori.size()) #torch.Size([32, 32]) torch.Size([32, 32])
                # target = self.targetM(label.cpu())#.to(self.device)
                # #print('target',target)

                # i2t = self.SoftCE(logits_per_image_ori, target)
                # #print(logits_per_image_ori)
                # t2i = self.SoftCE(logits_per_text_ori, target.T)
                # #print(t2i)
                # loss = (i2t + t2i) / 2
                # print('loss:',loss)
                # self.model_backward_and_update(loss)

                output, logits_per_image_ori, logits_per_text_ori = self.model(image, text_tensor)
                print('output',output.requires_grad)


                logits_per_image_ori, logits_per_text_ori = logits_per_image_ori.type(torch.FloatTensor).to(self.device), logits_per_text_ori.type(torch.FloatTensor).to(self.device)
                print(logits_per_image_ori.requires_grad,logits_per_text_ori.requires_grad) #torch.Size([32, 32]) torch.Size([32, 32])
                target = self.targetM(label.cpu()).to(self.device)
                print('target',target.requires_grad)

                i2t = self.SoftCE(logits_per_image_ori, target)
                #print(logits_per_image_ori)
                t2i = self.SoftCE(logits_per_text_ori, target.T)
                #print(t2i)
                loss = (i2t + t2i) / 2
                print('loss:',loss)
                self.model_backward_and_update(loss)
            elif loss_set == "cross_contrastive_loss":
                output= self.model(image)
                # output, logits_per_image_ori, logits_per_text_ori = self.model(image, text_tensor)
                # print('output',output.shape)
                # print('output0',len(output))
                # print('output1',len(output[1]))

                # target = self.targetM(label.cpu()).to(self.device)
                ground_truth_i = torch.arange(len(output),dtype=torch.long,device=self.device)
                ground_truth_t = torch.arange(len(output[0]),dtype=torch.long,device=self.device)
                output = output.type(torch.FloatTensor).cuda()
                # print(output.size(),output.T.size())
                # ground_truth_i = torch.arange(32,dtype=torch.long,device='cpu')
                # ground_truth_t = torch.arange(100,dtype=torch.long,device='cpu')
                loss1 = F.cross_entropy(output, ground_truth_i)
                
                # print('loss1:',loss1)
                output_t = output.T
                x = torch.zeros(len(output[0]), (len(output[0])-len(output))).cuda()
                z = torch.cat((x,output_t),1).cuda()
                #print(z.size())#torch.Size([100, 100])
                loss2 = F.cross_entropy(z, ground_truth_t)

                loss3 = F.cross_entropy(output, label) 

                loss = ((loss1+loss2) / 2) + loss3
                #loss = loss1 + loss3
                self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss1": loss1.item(),
            "loss2": loss2.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def targetM(self, y):
        print('y:',y)
        cap_m = (y == 0).sum()
        cls_m = y[y>0].max()
        y[y==0] = torch.arange(0, cap_m) + cls_m + 1
        return y.view(-1, 1) == y.view(1, -1)

    def SoftCE(self, s, t):
        s = torch.softmax(s, dim=-1)
        loss = - (t * s.log()).sum(dim = -1)
        return (loss/t.sum(dim=-1)).mean()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        # label2 = label.numpy()
        input = input.to(self.device)
        #print('input.size()',input.size()) #torch.Size([32, 3, 224, 224])
        
        #print('label.size()',label) #label.size() torch.Size([32])

        # label_text=[]
        # with open('/home/jason/coop/caltech.json', 'r') as fcc_file:
        #     fcc_data = json.load(fcc_file)
        #     #print(fcc_data)
        #     for i in range(len(label)):
        #         #print(str(label2[i]))
        #         if str(label2[i]) in fcc_data.keys():
        #             label_promt = 'This is a photo of ' + fcc_data[str(label2[i])]
        #             #print(label_promt)
        #             label_text.append(label_promt)
        #print(label_text)
        label = label.to(self.device)
        # print('label:',label)
        # text_tensor = clip.tokenize(label_text).to(self.device)
        #print('text_tensor.size():',text_tensor.size())

        return input, label#, text_tensor

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
            print('model_pathmodel_pathmodel_pathmodel_path',model_path)

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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
