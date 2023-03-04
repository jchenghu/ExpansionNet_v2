
import os
import torch
from datetime import datetime

from torch.nn.parameter import Parameter

def load_most_recent_checkpoint(model,
                                optimizer=None,
                                scheduler=None,
                                data_loader=None,
                                rank=0,
                                save_model_path='./', datetime_format='%Y-%m-%d-%H:%M:%S',
                                verbose=True):
    ls_files = os.listdir(save_model_path)
    most_recent_checkpoint_datetime = None
    most_recent_checkpoint_filename = None
    most_recent_checkpoint_info = 'no_additional_info'
    for file_name in ls_files:
        if file_name.startswith('checkpoint_'):
            _, datetime_str, _, info, _ = file_name.split('_')
            file_datetime = datetime.strptime(datetime_str, datetime_format)
            if (most_recent_checkpoint_datetime is None) or \
                    (most_recent_checkpoint_datetime is not None and
                     file_datetime > most_recent_checkpoint_datetime):
                most_recent_checkpoint_datetime = file_datetime
                most_recent_checkpoint_filename = file_name
                most_recent_checkpoint_info = info

    if most_recent_checkpoint_filename is not None:
        if verbose:
            print("Loading: " + str(save_model_path + most_recent_checkpoint_filename))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(save_model_path + most_recent_checkpoint_filename,
                                map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if data_loader is not None:
            data_loader.load_state(checkpoint['data_loader_state_dict'])
        return True, most_recent_checkpoint_info
    else:
        if verbose:
            print("Loading: no checkpoint found in " + str(save_model_path))
        return False, most_recent_checkpoint_info


def save_last_checkpoint(model,
                         optimizer,
                         scheduler,
                         data_loader,
                         save_model_path='./',
                         num_max_checkpoints=3, datetime_format='%Y-%m-%d-%H:%M:%S',
                         additional_info='noinfo',
                         verbose=True):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'data_loader_state_dict': data_loader.save_state(),
    }

    ls_files = os.listdir(save_model_path)
    oldest_checkpoint_datetime = None
    oldest_checkpoint_filename = None
    num_check_points = 0
    for file_name in ls_files:
        if file_name.startswith('checkpoint_'):
            num_check_points += 1
            _, datetime_str, _, _, _ = file_name.split('_')
            file_datetime = datetime.strptime(datetime_str, datetime_format)
            if (oldest_checkpoint_datetime is None) or \
                    (oldest_checkpoint_datetime is not None and file_datetime < oldest_checkpoint_datetime):
                oldest_checkpoint_datetime = file_datetime
                oldest_checkpoint_filename = file_name

    if oldest_checkpoint_filename is not None and num_check_points == num_max_checkpoints:
        os.remove(save_model_path + oldest_checkpoint_filename)

    new_checkpoint_filename = 'checkpoint_' + datetime.now().strftime(datetime_format) + \
                              '_epoch' + str(data_loader.get_epoch_it()) + \
                              'it' + str(data_loader.get_batch_it()) + \
                              'bs' + str(data_loader.get_batch_size()) + \
                              '_' + str(additional_info) + '_.pth'
    if verbose:
        print("Saved to " + str(new_checkpoint_filename))
    torch.save(checkpoint, save_model_path + new_checkpoint_filename)


def partially_load_state_dict(model, state_dict, verbose=False, max_num_print=5):
    own_state = model.state_dict()
    max_num_print = max_num_print
    count_print = 0
    for name, param in state_dict.items():
        if name not in own_state:
            if verbose:
                print("Not found: " + str(name))
            continue
        if isinstance(param, Parameter):
            param = param.data
        own_state[name].copy_(param)
        if verbose:
            if count_print < max_num_print:
                print("Found: " + str(name))
                count_print += 1

