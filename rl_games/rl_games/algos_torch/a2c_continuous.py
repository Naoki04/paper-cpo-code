import os
from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch
import torch.distributed as dist 


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        
        # アクターモデルの構築
        ## self.intr_reward_coef_embd は内発的報酬の係数のリスト(n_envs,1)であり、同じblockでは同じ値を持つ。(これはcommonの方で生成される)
        if self.intr_reward_coef_embd is not None and not (self.expl_type.startswith('mixed_expl') and 'disjoint' in self.expl_type):
            # 内的報酬がONかつ、(探索タイプがmixed_expl & disjointでない)場合、入力を内的報酬の係数で条件づける。
            input_shape = (self.obs_shape[0] + self.intr_reward_coef_embd.shape[1],)
        else:
            # それ以外では、入力は観測値のみ。
            input_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : input_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'type' : 'simple' if 'learn_param' not in self.expl_type else 'extra_param',
        }
        
        if self.expl_type.startswith('mixed_expl'): 
            # self.intr_coef_block_sizeは各ブロックの環境数
            build_config['coef_ids'] = self.intr_reward_coef_embd[::self.intr_coef_block_size,0] # 各ブロックの内的報酬の係数のリスト
            build_config['coef_id_idx'] = self.obs_shape[0]
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        # 共通価値ネットワークモデルの構築
        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True) # 共通価値ネットワークを使用するかどうかのフラグ
        
        # ロールアウトバッファの構築
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        # 300環境, 6エージェントの場合: batch_size=4800, minibatch_size=1200, is_discrete=False, is_rnn=True, ppo_device=cuda:0, seq_length=16(seq_lengthはRNN関連のパラメータ)
        
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn, override_state=None):
        state = {self.global_rank: self.get_full_state_weights()}
        if override_state is not None:
            state = override_state
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint[self.global_rank] if self.global_rank in checkpoint else checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    # ロスを計算する関数
    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values'] # データ収集時の価値関数の予測値、データ拡張時に変更したobs_embで再計算されている。
        old_action_log_probs_batch = input_dict['old_logp_actions'] # データ収集時の方策の対数確率
        advantage = input_dict['advantages'] # アドバンテージ、データ拡張時に変更したobs_embで再計算されている。
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        off_policy_mask = input_dict.get('off_policy_mask', None)
        awac_mask = input_dict.get('awac_mask', None)
        leader_online_mask = input_dict.get('leader_online_mask', None)
        follower_online_mask = input_dict.get('follower_online_mask', None)
        
        critic_mask = torch.logical_or(leader_online_mask, follower_online_mask) # onlineデータ

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp'] # 現在のポリシーそのactionを生成する対数確率
            values = res_dict['values'] # 現在のモデルで予測した状態価値
            entropy = res_dict['entropy'] # [1200] # 現在のポリシーのエントロピー
            mu = res_dict['mus'] # [1200,23]
            sigma = res_dict['sigmas']
            
            # 現在のリーダーでの行動確率を計算
            leader_batch_dict = batch_dict.copy()
            leader_batch_dict['obs'][:,-1] = 0
            leader_res_dict = self.model(leader_batch_dict)
            leader_action_log_probs = leader_res_dict['prev_neglogp']
            #print("leader_action_log_probs: ", leader_action_log_probs.shape)

            # アクターのロスを計算する部分(普通のPPOロス)
            if self.sapg2:
                a_loss, a_loss_info = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, leader_action_log_probs, advantage, self.ppo, curr_e_clip, off_policy_mask, awac_mask, leader_online_mask, follower_online_mask, self.awac_lambda, self.awac_max, self.awac_alpha, self.awac_beta, self.awac_gamma, critic_mask, self.enable_w)
            else:
                a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip, off_policy_mask)
            
            if self.has_value_loss:
                # クリッピング付きのクリティックロス
                if self.sapg2: # critic_maskを使用する場合(厳密にオンラインデータのみで学習する)
                    if self.vanilla_sapg:
                        c_loss = common_losses.critic_loss_sapg(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value, critic_mask, off_policy_mask, self.enable_w)
                    else:
                        c_loss = common_losses.critic_loss_sapg2(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value, critic_mask, off_policy_mask, self.enable_w)
                else:
                    c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                # なし
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            
            # アクターの出力範囲を制限するためのboundロス
            if self.bound_loss_type == 'regularisation':
                # muの二乗和をロスとして追加
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                # muの上下限を超えた分の二乗和をロスを追加
                b_loss = self.bound_loss(mu)
            else:
                # なし
                b_loss = torch.zeros(len(mu), device=self.ppo_device)
            
            # オンラインデータでのみ学習するようにマスクを適用した後、正規化する
            if self.sapg2:
                if self.vanilla_sapg:
                    b_mask = torch.logical_or(critic_mask, off_policy_mask)
                else:
                    b_mask = critic_mask
                assert b_loss.shape == b_mask.shape, "b_loss shape: {}, b_mask shape: {}".format(b_loss.shape, b_mask.shape)
                b_loss = b_loss * b_mask
                
                if self.enable_w:
                    w = b_mask.sum() / b_mask.numel()
                    b_loss = b_loss / w            
                    
                
                
            # エントロピーについてもマスクを適用してから、正規化する
            if self.sapg2:
                if self.vanilla_sapg:
                    entropy_mask = torch.logical_or(critic_mask, off_policy_mask)   
                else:
                    entropy_mask = critic_mask
                assert entropy.shape == entropy_mask.shape, "entropy shape: {}, entropy_mask shape: {}".format(entropy.shape, entropy_mask.shape)
                entropy = entropy * entropy_mask
            
                if self.enable_w:
                    w = entropy_mask.sum() / entropy_mask.numel()
                    entropy = entropy / w
                    
            
            
            if self.expl_type.startswith('mixed_expl') and self.config.get('expl_reward_type') == 'entropy':
                ec_candidates = self.intr_reward_coef[::self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
                ec_indices = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            elif self.expl_type.startswith('simple') and self.config.get('expl_reward_type') == 'entropy':
                entropy_coef = self.intr_reward_coef
            else:
                entropy_coef = self.entropy_coef
            
            if os.getenv('LOG_OFF_POLICY_GRADS'):
                loss_arr = a_loss.unsqueeze(1) + 0.5 * c_loss * self.critic_coef - (entropy_coef*entropy).unsqueeze(1) + b_loss.unsqueeze(1) * self.bounds_loss_coef
                off_policy_loss = torch.masked_select(loss_arr, input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                on_policy_loss = torch.masked_select(loss_arr, ~input_dict['off_policy_mask'].unsqueeze(1)).sum() / len(input_dict['off_policy_mask'])
                grads_off = self.get_grads(off_policy_loss)
                grads_on = self.get_grads(on_policy_loss)


            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , (entropy_coef*entropy).unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy_loss, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef
            
            print("=========")
            print("a_loss: ", a_loss)
            print("c_loss: ", c_loss* self.critic_coef*0.5)
            print("entropy_loss: ", entropy_loss*self.entropy_coef)
            print("b_loss: ", b_loss* self.bounds_loss_coef)
            print("critic_coef: ", self.critic_coef)
            print("entropy_coef: ", entropy_coef)
            print("bounds_loss_coef: ", self.bounds_loss_coef)
            

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        all_grads = self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0) 

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        if os.getenv('LOG_OFF_POLICY_GRADS'):
            contrib_off = torch.masked_select(contrib, input_dict['off_policy_mask'])
            contrib_on = torch.masked_select(contrib, ~input_dict['off_policy_mask'])
            contrib_off = torch.nan_to_num(contrib_off.mean())
            contrib_on = torch.nan_to_num(contrib_on.mean())
        
            extras = {
                "off_policy_contrib" : contrib_off.item(),
                "on_policy_contrib" : contrib_on.item(),
                "off_policy_grads" : grads_off.detach().cpu(),
                "on_policy_grads" : grads_on.detach().cpu(),
            }
        else:
            extras = {
                "on_policy_contrib" : contrib.mean().item(),
                "off_policy_contrib" : 0,
                "on_policy_grads" : all_grads.detach().cpu(),
                "off_policy_grads" : torch.zeros_like(all_grads).cpu(),
            }     
        if self.expl_type.startswith('mixed_expl'):
            bl_ids = self.intr_reward_coef_embd[::self.intr_coef_block_size, 0].reshape(-1,1)
            bl_idxs = torch.argmax((obs_batch[:,-self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item() for i in range(self.num_actors // self.intr_coef_block_size)]
        # ログの記録
        a_loss_dict = {
            "total": a_loss,
            "ppo": a_loss_info["ppo"],
            "awac": a_loss_info["awac"],
            "klppo": a_loss_info["klppo"],
        }
        self.train_result = (a_loss_dict, c_loss, torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0], \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, extras)

    # ロスの計算
    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
    def get_grads(self, loss, retain_graph=True):
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None

        loss.backward(retain_graph=retain_graph)

        all_grads_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads_list.append(param.grad.view(-1))

        all_grads = torch.cat(all_grads_list)
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
        
        return all_grads

