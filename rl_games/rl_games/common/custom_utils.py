import numpy as np
import torch

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def remove_envs_from_info(infos, num_envs):
    for key in list(infos.keys()):
        if isinstance(infos[key], dict):
            infos[key] = remove_envs_from_info(infos[key], num_envs)
        elif isinstance(infos[key], list) or isinstance(infos[key], (np.ndarray, torch.Tensor)):
            if key in ['successes', 'closest_keypoint_max_dist']:
                block_size = len(infos[key]) - num_envs
                if len(infos[key]) % block_size == 0:
                    for i in range(len(infos[key]) // block_size):
                        infos[f"{key}_per_block/block_{i}"] = infos[key][i*block_size:(i+1)*block_size]
            infos[key] = infos[key][num_envs:]
    return infos

def shuffle_batch(batch_dict, horizon_length):
    indices = torch.randperm(len(batch_dict['returns']) // horizon_length).to(batch_dict['returns'].device).reshape(-1,1)*horizon_length + torch.arange(horizon_length).to(batch_dict['returns'].device).reshape(1,-1)
    flattened_indices = indices.reshape(-1)
    for key in batch_dict:
        if key == 'rnn_states':
            if batch_dict[key] is None:
                continue
            else:
                batch_dict[key] = [s[:, indices[:, 0] // horizon_length] for s in batch_dict[key]]
        elif key in ['played_frames', 'step_time']:
            continue
        else:
            batch_dict[key] = batch_dict[key][flattened_indices]
    return batch_dict

def create_sinusoidal_encoding(arr, dim, n=10):
    """
    Create dim dimensional sinusoidal encoding of values in arr
    arr is a 1-dimensional Torch tensor
    """
    assert dim % 2 == 0
    
    denom = n**(2*torch.arange(dim//2, dtype=torch.float32, device=arr.device)/dim)
    
    return torch.cat([torch.sin(arr.unsqueeze(-1) / denom), torch.cos(arr.unsqueeze(-1) / denom)], dim=-1)


def filter_leader(val, orig_len, repeat_idxs, num_blocks, required_mask):
    """
    Filters data corresponding to leader i.e. evaluation policy
    Used with mixed_expl
    """
    """
    repeat_indexは[0, 0, ?]
    obsの時は(envs300, horizon16),
        origi_len = 4800
        val: [4800*2, 100]
        new_val: [4800*2+800, 100]
        
    """
    
    if len(val) > 1: # tensorが来る[step, dim]や[step]のtensorが来る
        # 元の処理にrequired_maskを追加
        bsize = orig_len // num_blocks # 4800/6 = 800
        filtered_val = []
        
        for i, idx in enumerate(repeat_idxs):#(i,idx)=(0,1,2),(0,0,?))
            if idx == 0:
                filtered_val.append(val[i*orig_len:(i+1)*orig_len][required_mask[i*orig_len:(i+1)*orig_len]]) # [0:4800]を取り出す(オリジナル全て), required_maskはAWAC用に拡張してできたリーダーデータのリーダー埋め込みを除外するため
            else:
                filtered_val.append(val[i*orig_len + (idx-1)*bsize:i*orig_len + idx*bsize]) # [?x4800: ?x4800+800]を取り出す
        new_val = torch.cat(filtered_val, dim=0)
        
    else: # rnn_states[i]([1,num_envs,768])が来る
        # 元の処理にrequired_maskを追加
        bsize = orig_len // num_blocks
        horizon_len = required_mask.numel() // (orig_len * len(repeat_idxs))
        
        filtered_val = []
        for i, idx in enumerate(repeat_idxs):
            if idx == 0:
                #filtered_val.append(val[:, i*orig_len:(i+1)*orig_len][:,required_mask[i*orig_len:(i+1)*orig_len]]) # [0:4800]を取り出す(オリジナル全て), required_maskはAWAC用に拡張してできたリーダーデータのリーダー埋め込みを除外するため
                filtered_val.append(val[:, i*orig_len:(i+1)*orig_len][:,required_mask[0:-1:horizon_len][i*orig_len:(i+1)*orig_len]]) # [0:4800]を取り出す(オリジナル全て), required_maskはAWAC用に拡張してできたリーダーデータのリーダー埋め込みを除外するため
            else:
                filtered_val.append(val[:, i*orig_len + (idx-1)*bsize:i*orig_len + idx*bsize])
        new_val = torch.cat(filtered_val, dim=1)
    
    return new_val

