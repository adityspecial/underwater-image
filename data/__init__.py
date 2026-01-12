'''create dataset and dataloader'''
import logging
import torch.utils.data

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(f'Dataloader [{phase}] is not found.')

def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D

    # === SUPPORT dataroot_HR + dataroot_LR ===
    if 'dataroot_HR' in dataset_opt and 'dataroot_LR' in dataset_opt:
        dataset = D(
            dataroot_HR=dataset_opt['dataroot_HR'],
            dataroot_LR=dataset_opt['dataroot_LR'],
            l_resolution=dataset_opt['l_resolution'],
            r_resolution=dataset_opt['r_resolution'],  # ← PASS THIS
            split=phase,
            data_len=dataset_opt.get('data_len', -1),
            need_LR=(mode == 'LRHR')
        )
    else:
        dataset = D(
            dataroot=dataset_opt['dataroot'],
            l_resolution=dataset_opt['l_resolution'],
            r_resolution=dataset_opt['r_resolution'],  # ← PASS THIS
            split=phase,
            data_len=dataset_opt.get('data_len', -1),
            need_LR=(mode == 'LRHR')
        )

    logger = logging.getLogger('base')
    logger.info(f'Dataset [{dataset.__class__.__name__} - {dataset_opt["name"]}] is created.')
    return dataset