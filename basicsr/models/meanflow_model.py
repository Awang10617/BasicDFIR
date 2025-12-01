import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from functools import partial
from einops import rearrange
from .flow_model import FlowModel
import numpy as np
'''
def stopgrad(x):
    return x.detach()
def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    # 计算每个样本的均方误差 (B,)
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    # 计算自适应权重：误差越大的样本权重越小，防止离群点干扰
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()
'''
@MODEL_REGISTRY.register()

class MeanFlowModel(FlowModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(MeanFlowModel, self).__init__(opt)
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        # define losses
        if train_opt.get('flow_opt'):
            self.flow_loss = build_loss(train_opt['flow_opt']).to(self.device)
        else:
            self.flow_loss = None
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()



    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    '''
    define the interpolation processing of flow-based method. Should be instanced for differnet flow-based method.
    input:time step t
    '''
    def flow_interpolation(self, t, r=None):
        self.xt= (1-t)*self.gt + self.lq * t

        return self.xt


    def sample_t_r(self, batch_size, device):
        # 简单的均匀分布采样 t，并且设 r=t
        t = torch.rand(batch_size, device=device)
        r = t.clone()
        return t, r


    '''
    define the timestep sampling method.
    '''
    def sample_timestep(self):
        #self.normer = Normalizer.from_list(normalizer)
        batch_size = self.lq.shape[0]
        device = self.device

        t, r = self.sample_t_r(batch_size, device)
        self.t = t
        self.r = r

        self.t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        self.r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        _,_,h,w, = self.gt.shape
        self.lq = F.interpolate(self.lq,size=(h,w),mode='bicubic',align_corners=False)

        self.v_hat = self.lq - self.gt


        '''
    main processing of flow.
    '''
    def flow_process(self):
        self.sample_timestep()
        self.xt = self.flow_interpolation(self.t_)
        jvp_args = (
            lambda z, t, r: self.net_g((z, t, r)),
            (self.xt, self.t, self.r),
            (self.v_hat, torch.ones_like(self.t), torch.zeros_like(self.r)),
        )
        u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)
        u_tgt = self.v_hat - (self.t_ - self.r_) * dudt
        #error = u - stopgrad(u_tgt)
        #loss = adaptive_l2_loss(error)

        error = u - u_tgt.detach()
        loss = torch.nn.functional.l1_loss(error, torch.zeros_like(error))
        return loss

    '''
    sample image with flow-based ODE.
    '''

    @torch.no_grad()
    def sample_image(self, lq, model=None):  # <--- 修改了参数定义
        # 如果没传 model，就用默认的 self.net_g
        if model is None:
            model = self.net_g

        # 1. 先放大 LQ 到目标尺寸 (128x128)
        scale = self.opt.get('scale', 4)
        h, w = lq.shape[-2:]
        target_size = (h * scale, w * scale)
        x = F.interpolate(lq, size=target_size, mode='bicubic', align_corners=False)

        # 2. 准备循环参数
        batch_size = x.shape[0]
        device = x.device
        num_steps = 10
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        # 3. Euler 积分循环 (从 LQ 逐步推导到 GT)
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next

            t_input = torch.full((batch_size,), t_curr, device=device)
            r_input = t_input.clone()

            # 预测 v
            v_pred = model((x, t_input, r_input))

            # 更新 x
            x = x - dt * v_pred

        return x

    '''
    Add flow-based loss function.
    '''

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        flow_loss = self.flow_process()
        self.vt_pre = self.net_g((self.xt, self.t, self.r))
        self.output = self.xt - self.t_ * self.vt_pre
        l_total = flow_loss
        loss_dict = OrderedDict()

        # flow loss
        if self.flow_loss:
            l_flow = self.flow_loss(self.output, self.gt)
            l_total += l_flow
            loss_dict['l_flow'] = l_flow

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    '''
    self.output is reconstructed image from sample_image method.
    '''
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        with torch.no_grad():
            scale = self.opt.get('scale', 4)
            h,w = self.lq.shape[-2:]
            target_size = (h * scale, w * scale)
            z = F.interpolate(self.lq, size=target_size, mode='bicubic', align_corners=False)
            model_input_size = self.opt['network_g']['input_size']
            if z.shape[2] != model_input_size or z.shape[3] != model_input_size:
                start_h = (z.shape[2] - model_input_size) // 2
                start_w = (z.shape[3] - model_input_size) // 2
                end_h = start_h + model_input_size
                end_w = start_w + model_input_size
                z = z[:, :, start_h:end_h, start_w:end_w]
                if hasattr(self, 'gt'):
                    self.gt = self.gt[:, :, start_h:end_h, start_w:end_w]

            batch_size = z.shape[0]
            device = self.device
            num_steps = 10  # 采样步数，建议 10-20

            # 生成时间步: [1.0, 0.9, ..., 0.0]
            time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]

                # 构造时间张量 [B]
                t_tensor = torch.full((batch_size,), t_cur, device=device)
                # 你的sampler里写 r=t，我们照搬这个逻辑
                r_tensor = torch.full((batch_size,), t_cur, device=device)

                # 【关键】调用模型预测速度 v
                # 注意：必须打包成 tuple ((z, t, r)) 传进去
                model_input = (z, t_tensor, r_tensor)
                v = net(model_input)

                # Euler 更新公式: z_{next} = z_{curr} - dt * v
                dt = t_cur - t_next
                z = z - dt * v

            # =================================================
            # Step 3: 保存结果
            # =================================================
            self.output = z
        self.net_g.train()

    # def test_selfensemble(self):
    #     # TODO: to be tested
    #     # 8 augmentations
    #     # modified from https://github.com/thstkdgus35/EDSR-PyTorch
    #
    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == 'v':
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #         elif op == 'h':
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #         elif op == 't':
    #             tfnp = v2np.transpose((0, 1, 3, 2)).copy()
    #
    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()
    #
    #         return ret
    #
    #     # prepare augmented data
    #     lq_list = [self.lq]
    #     for tf in 'v', 'h', 't':
    #         lq_list.extend([_transform(t, tf) for t in lq_list])
    #
    #     # inference
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #         self.net_g.train()
    #
    #     # merge results
    #     for i in range(len(out_list)):
    #         if i > 3:
    #             out_list[i] = _transform(out_list[i], 't')
    #         if i % 4 > 1:
    #             out_list[i] = _transform(out_list[i], 'h')
    #         if (i % 4) % 2 == 1:
    #             out_list[i] = _transform(out_list[i], 'v')
    #     output = torch.cat(out_list, dim=0)
    #
    #     self.output = output.mean(dim=0, keepdim=True)

    '''
    added code to delete flow-based attribution self.v_pred et.al.
    '''
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            if hasattr(self,'xt'):
                del self.xt
            if hasattr(self, 'vt_pre'):
                del self.vt_pre
            if hasattr(self, 'lq'):
                del self.lq
            if hasattr(self, 'output'):
                del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)