from torch import nn
import torch
import math


def critic_loss(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    return default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    #return model.get_value_layer().loss(value_preds_batch=value_preds_batch, values=values, curr_e_clip=curr_e_clip, return_batch=return_batch, clip_value=clip_value)


def critic_loss_sapg2_double_critic(model, value_preds_batch, value_preds_batch1, value_preds_batch2, values, values1, values2, curr_e_clip, return_batch, clip_value, critic_mask, off_policy_mask, enable_w):
    # 各criticについて損失を計算
    c_loss1 = default_critic_loss(value_preds_batch1, values1, curr_e_clip, return_batch, clip_value)
    c_loss2 = default_critic_loss(value_preds_batch2, values2, curr_e_clip, return_batch, clip_value)
    c_loss = c_loss1 + c_loss2
    
    # あとは同じ。マスクをかけて、統計量を計算する。
    mask = critic_mask.unsqueeze(1)
    assert mask.shape == c_loss.shape, "mask shape is {}, c_loss shape is {}".format(mask.shape, c_loss.shape)
    
    c_loss_masked = c_loss * mask
    
    # 使うデータの割合で正規化
    if enable_w:
        w = mask.count_nonzero().item()/c_loss.shape[0] # .itemの時点でdetachされるので、detach()は不要
        c_loss_masked = c_loss_masked / w 
    
    # マスクされた部分のQ値とV値の平均を計算
    masked_values = values * mask
    masked_value_preds = value_preds_batch * mask
    masked_returns = return_batch * mask
    num_valid = mask.sum()
    mean_v = masked_values.sum() / (num_valid + 1e-8)
    mean_v_pred = masked_value_preds.sum() / (num_valid + 1e-8)
    mean_q = masked_returns.sum() / (num_valid + 1e-8)
    
    return c_loss_masked, mean_v.detach().cpu().numpy(), mean_q.detach().cpu().numpy(), mean_v_pred.detach().cpu().numpy()
    

def critic_loss_sapg2(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value, critic_mask, off_policy_mask, enable_w):
    c_loss = default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    # critic_maskをかけてから、大きさを正規化する
    mask = critic_mask.unsqueeze(1)
    assert mask.shape == c_loss.shape, "mask shape is {}, c_loss shape is {}".format(mask.shape, c_loss.shape)
    
    c_loss_masked = c_loss * mask
    
    # 使うデータの割合で正規化
    if enable_w:
        w = mask.count_nonzero().item()/c_loss.shape[0] # .itemの時点でdetachされるので、detach()は不要
        c_loss_masked = c_loss_masked / w 
    
    # マスクされた部分のQ値とV値の平均を計算
    masked_values = values * mask
    masked_value_preds = value_preds_batch * mask
    masked_returns = return_batch * mask
    num_valid = mask.sum()
    mean_v = masked_values.sum() / (num_valid + 1e-8)
    mean_v_pred = masked_value_preds.sum() / (num_valid + 1e-8)
    mean_q = masked_returns.sum() / (num_valid + 1e-8)
    
    return c_loss_masked, mean_v.detach().cpu().numpy(), mean_q.detach().cpu().numpy(), mean_v_pred.detach().cpu().numpy()


def critic_loss_sapg(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value, critic_mask, off_policy_mask, enable_w):
    c_loss = default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    
    mask = torch.logical_or(critic_mask, off_policy_mask).unsqueeze(1)
    
    assert mask.shape == c_loss.shape, "mask shape is {}, c_loss shape is {}".format(mask.shape, c_loss.shape)
    c_loss_masked = c_loss * mask
    
    # 使うデータの割合で正規化
    if enable_w:
        w = mask.count_nonzero().item()/c_loss.shape[0] # .itemの時点でdetachされるので、detach()は不要
        c_loss_masked = c_loss_masked / w 
    
    # マスクされた部分のQ値とV値の平均を計算
    masked_values = values * mask
    masked_value_preds = value_preds_batch * mask
    masked_returns = return_batch * mask
    num_valid = mask.sum()
    mean_v = masked_values.sum() / (num_valid + 1e-8)
    mean_v_pred = masked_value_preds.sum() / (num_valid + 1e-8)
    mean_q = masked_returns.sum() / (num_valid + 1e-8)
    
    return c_loss_masked, mean_v, mean_q, mean_v_pred

  


def default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    if clip_value:
        value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
        value_losses = (values - return_batch)**2
        value_losses_clipped = (value_pred_clipped - return_batch)**2
        c_loss = torch.max(value_losses,
                                         value_losses_clipped)
    else:
        c_loss = (return_batch - values)**2
    return c_loss

def smooth_clamp(x, mi, mx):
    return 1/(1 + torch.exp((-(x-mi)/(mx-mi)+0.5)*4)) * (mx-mi) + mi


def smoothed_actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs) # pnew/poldの計算
        surr1 = advantage * ratio
        surr2 = advantage * smooth_clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)
    return a_loss


def actor_loss(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, off_policy_mask):    
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)
    return a_loss

def actor_loss_with_awac(old_action_neglog_probs_batch, action_neglog_probs, leader_action_log_probs, advantage, is_ppo, curr_e_clip, off_policy_mask, awac_mask, leader_online_mask, follower_online_mask, lambda_awac, lambda_ppo, awac_max, awac_alpha, awac_beta, awac_gamma, critic_mask, enable_w):
    """
    # 1. leader_online_mask(リーダーのオンライン学習), off_policy_mask(リーダーのオンライン学習)のデータはPPOで学習する。
    """
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2) # マイナスのmaxを取ることで、最小化問題に変換
    else:
        ppo_loss = (action_neglog_probs * advantage)
        print("??????????? not PPO ????????????")
        
    ppo_loss = ppo_loss * torch.logical_or(leader_online_mask, off_policy_mask)
    
    
    """
    # 2. awac_mask(フォロワーのオフライン学習)のデータはAWACで学習する。
    """
    # AWACロスの計算(expが爆発するのでadvantageをクリップする)
    offline_awac_loss = - torch.clamp(torch.exp(advantage / lambda_awac), max=awac_max)*(-action_neglog_probs) # 先頭のマイナスで、最小化問題に変換
    offline_awac_loss = offline_awac_loss * awac_mask 
    
    
    """
    # 3. follower_online_mask(フォロワーのオンライン学習)のデータは、KL拘束付きのPPOで学習する。
    """
    ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
    surr1 = (-(leader_action_log_probs - action_neglog_probs)*lambda_ppo + advantage) *ratio # これは最大化したい値
    surr2 = (-(leader_action_log_probs - action_neglog_probs)*lambda_ppo + advantage) * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip) # これは最大化したい値
    online_awac_loss = torch.max(-surr1, -surr2) # マイナスのmaxを取ることで、最小化問題に変換
    online_awac_loss = online_awac_loss * follower_online_mask
    """
    print("DEBUG")
    print("advantage: {}".format((advantage*follower_online_mask).sum()/follower_online_mask.sum()))
    print("lim:{}".format(((leader_action_log_probs - action_neglog_probs)*follower_online_mask).sum()/follower_online_mask.sum()))
    """

    """
    # 4. 各ロスのバランスを取って、最終的なロスを計算する。
    """
    if enable_w:
        # PPO, KL-PPOのデータ数で正規化する。
        ppo_klppo_mask = torch.logical_or(leader_online_mask, follower_online_mask)
        ppo_klppo_mask = torch.logical_or(ppo_klppo_mask, off_policy_mask)
        w = (ppo_klppo_mask.numel() / ppo_klppo_mask.sum()).detach()
        ppo_loss = ppo_loss * w
        online_awac_loss = online_awac_loss * w
        
    # スケーリング
    ppo_loss = awac_alpha * ppo_loss
    offline_awac_loss = awac_beta * offline_awac_loss
    online_awac_loss = awac_gamma * online_awac_loss
    
    
    # 合計ロス
    a_loss = ppo_loss + offline_awac_loss + online_awac_loss
    
    a_loss_info = {
        "ppo": ppo_loss.mean(), 
        "awac": offline_awac_loss.mean(),
        "klppo": online_awac_loss.mean()
        }
    """
    print("--- LOSS INFO ---")
    print("  ppo_loss: {}".format(ppo_loss.abs().mean()))
    print("  awac_loss: {}".format(offline_awac_loss.abs().mean()))
    print("  klppo_loss: {}".format(online_awac_loss.abs().mean()))
    """
    
     
    return a_loss, a_loss_info



def actor_loss_sapg(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, off_policy_mask, awac_mask, awac_lambda, awac_max, awac_alpha, critic_mask, enable_w):
    # PPOロスを計算
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2)
    else:
        ppo_loss = (action_neglog_probs * advantage)
    
    # critic_mask or off_policy_maskのデータのみを使う。
    mask = torch.logical_or(critic_mask, off_policy_mask)
    
    assert mask.shape == ppo_loss.shape, "mask shape is {}, ppo_loss shape is {}".format(mask.shape, ppo_loss.shape)
    a_loss = ppo_loss * mask 
    
    if enable_w:
        w = mask.count_nonzero().item()/mask.shape[0] # .itemの時点でdetachされるので、detach()は不要
        a_loss = a_loss / w

    
    #print("==========PPO loss is used for critic_mask or off_policy_mask data, for debug.===============")
    
    return a_loss


def decoupled_actor_loss(behavior_action_neglog_probs, action_neglog_probs, proxy_neglog_probs, advantage, curr_e_clip):
    logratio = proxy_neglog_probs - action_neglog_probs
    #neglogp_adj = -torch.max(-behavior_action_neglog_probs, -action_neglog_probs.detach() - math.log(100))
    pg_losses1 = -advantage * torch.exp(behavior_action_neglog_probs - action_neglog_probs)
    clipped_logratio = torch.clamp(logratio, math.log(1.0 - curr_e_clip), math.log(1.0 + curr_e_clip))
    pg_losses2 = -advantage * torch.exp(clipped_logratio - proxy_neglog_probs + behavior_action_neglog_probs)
    pg_losses = torch.max(pg_losses1,pg_losses2)

    return pg_losses
