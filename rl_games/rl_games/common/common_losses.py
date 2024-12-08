from torch import nn
import torch
import math


def critic_loss(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
    return default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    #return model.get_value_layer().loss(value_preds_batch=value_preds_batch, values=values, curr_e_clip=curr_e_clip, return_batch=return_batch, clip_value=clip_value)

def critic_loss_sapg2(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value, critic_mask, off_policy_mask):
    c_loss = default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    # critic_maskをかけてから、大きさを正規化する
    mask = torch.logical_or(critic_mask, off_policy_mask).unsqueeze(1)
    assert mask.shape == c_loss.shape, "mask shape is {}, c_loss shape is {}".format(mask.shape, c_loss.shape)
    
    c_loss_masked = c_loss * mask
    
    # 使うデータの割合で正規化
    w = mask.count_nonzero().item()/c_loss.shape[0]
    c_loss = c_loss / w
    print("critic_w", w)
    
    return c_loss_masked

def critic_loss_sapg(model, value_preds_batch, values, curr_e_clip, return_batch, clip_value, critic_mask, off_policy_mask):
    c_loss = default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
    # critic_maskをかけてから、大きさを正規化する
    mask = torch.logical_or(critic_mask, off_policy_mask).unsqueeze(1)
    assert mask.shape == c_loss.shape, "mask shape is {}, c_loss shape is {}".format(mask.shape, c_loss.shape)
    
    c_loss_masked = c_loss * mask
    
    # 使うデータの割合で正規化
    w = mask.count_nonzero().item()/c_loss.shape[0]
    c_loss = c_loss / w
    
    return c_loss_masked

  


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

def actor_loss_with_awac(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, off_policy_mask, awac_mask, awac_lambda, awac_max, awac_alpha, critic_mask):
    # PPOロスを計算
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2)
        # awac maskでマスキングする
    else:
        ppo_loss = (action_neglog_probs * advantage)
    
    # critic用データとawac用データは取り除く
    #ppo_loss = ppo_loss * torch.logical_not(awac_mask) * torch.logical_not(critic_mask)
    
    """
    # AWACロスの計算(expが爆発するのでadvantageをクリップする)
    awac_loss = torch.clamp(torch.exp(1/awac_lambda * advantage), max=awac_max)*old_action_neglog_probs_batch
    
    # critic用データは取り除く
    awac_loss = awac_loss * awac_mask * torch.logical_not(critic_mask)
    # バランス調整のため、awac_lossを正規化する。
    w = awac_alpha / awac_loss.abs().sum()
    # ロスを足し合わせる
    a_loss = ppo_loss + w*awac_loss 
    # critic用データ以外の割合で正規化
    a_loss = a_loss / (torch.logical_not(critic_mask).count_nonzero().item()/critic_mask.shape[0])
    """
    """
    #print("=======")
    #print("w",w)
    print("PPO_loss:", ppo_loss.sum()/ppo_loss.count_nonzero().item())
    print("AWAC(バランス後):", awac_loss.sum()/awac_loss.count_nonzero().item()*w)
    print("lambda", awac_lambda)
    print("alpha", awac_alpha)
    """
    
    # デバッグ用: 全てPPOで学習する。(1リーダーオンライン, 2フォロワーオンライン, 3リーダーオフライン, 4フォロワーAWACのうち、123をPPOで学習する。)
    # つまり、(critic_maskと,awac_maskがないもの) = AWAC_maskがないものをPPOで学習する。
    #w = awac_mask.count_nonzero().item()/awac_mask.shape[0]
    mask = torch.logical_or(critic_mask, off_policy_mask)
    assert mask.shape == ppo_loss.shape, "mask shape is {}, ppo_loss shape is {}".format(mask.shape, ppo_loss.shape)
    a_loss = ppo_loss * mask #/ w
    
    w = mask.count_nonzero().item()/mask.shape[0]
    a_loss = a_loss / w
    
    print("actor_w", w)
    
    
    print("==========PPO loss is used for not awac data, for debug.===============")
    
    return a_loss

def actor_loss_sapg(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, off_policy_mask, awac_mask, awac_lambda, awac_max, awac_alpha, critic_mask):
    # PPOロスを計算
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2)
        # awac maskでマスキングする
    else:
        ppo_loss = (action_neglog_probs * advantage)
    
    # critic用データとawac用データは取り除く
    #ppo_loss = ppo_loss * torch.logical_not(awac_mask) * torch.logical_not(critic_mask)

    
    # デバッグ用: 全てPPOで学習する。(1リーダーオンライン, 2フォロワーオンライン, 3リーダーオフライン, 4フォロワーAWACのうち、123をPPOで学習する。)
    # つまり、(critic_maskと,awac_maskがないもの) = AWAC_maskがないものをPPOで学習する。
    #w = awac_mask.count_nonzero().item()/awac_mask.shape[0]
    mask = torch.logical_or(critic_mask, off_policy_mask)
    assert mask.shape == ppo_loss.shape, "mask shape is {}, ppo_loss shape is {}".format(mask.shape, ppo_loss.shape)
    a_loss = ppo_loss * mask #/ w
    
    w = mask.count_nonzero().item()/mask.shape[0]
    a_loss = a_loss / w
    
    print("==========PPO loss is used for critic_mask or off_policy_mask data, for debug.===============")
    
    return a_loss


def decoupled_actor_loss(behavior_action_neglog_probs, action_neglog_probs, proxy_neglog_probs, advantage, curr_e_clip):
    logratio = proxy_neglog_probs - action_neglog_probs
    #neglogp_adj = -torch.max(-behavior_action_neglog_probs, -action_neglog_probs.detach() - math.log(100))
    pg_losses1 = -advantage * torch.exp(behavior_action_neglog_probs - action_neglog_probs)
    clipped_logratio = torch.clamp(logratio, math.log(1.0 - curr_e_clip), math.log(1.0 + curr_e_clip))
    pg_losses2 = -advantage * torch.exp(clipped_logratio - proxy_neglog_probs + behavior_action_neglog_probs)
    pg_losses = torch.max(pg_losses1,pg_losses2)

    return pg_losses
