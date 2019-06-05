from flowNet.data_utils import MatlabBasedCosegDataset,PickleBasedCosegDataset
from torch.utils.data import dataloader
from flowNet import trainerClass
from tensorboardX import SummaryWriter
import torch
import os
import random
import pickle as pkl
log_dir = '/media/fastData/coSegTraining/8_batched_025_res'
writer = SummaryWriter(log_dir=log_dir)

#
# dataset = MatlabBasedCosegDataset(root_dir='/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses',
#                                   example_format='*/rectifiedPatchesSIFT_128X128_withField_*.mat')

batch_size = 8
dataset = PickleBasedCosegDataset(root_dir='/media/fastData/coSegDataPasses/SintelCleanPasses',
                                  example_format='affnet_*/*/*.pklz', max_batch_size=batch_size)
trainer = trainerClass.Trainer(train_dir=log_dir)

data_file = './small_batches_1000_less_crop.pkl'
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        batch_list = pkl.load(f)
else:
    # train_loader = dataloader.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    num_of_batches = 1000
    batch_list = []
    for k, sample in enumerate(dataset):
        batch_list.append(sample)
        if k == num_of_batches-1:
            break
    with open(data_file,'wb') as f:
        pkl.dump(file=f,obj=batch_list)
for kk in range(0,5000):
    random.shuffle(batch_list)
    total_loss = 0
    for k, batch in enumerate(batch_list):
        loss = trainer.train_sample(batch, writer)
        total_loss += loss
    writer.add_scalar('data/loss', total_loss/len(batch_list), kk)
    print(total_loss/len(batch_list))
    trainer.save_model(kk)
# while True:
#     for k,sample in enumerate(dataset):
#         # print(torch.mean(sample['gt_flow_025'][51, 0, :, :]))
#         # print(torch.mean(sample['gt_flow_0125'][51, 0, :, :]))
#         # torch.mean(sample['gt_flow_025'][51, 0, :, :]) / torch.mean(sample['gt_flow_0125'][51, 0, :, :]) - 2
#
#         loss = trainer.train_sample(sample)
#
#         ctr += 1
#         loss_sum += loss.item()
#         if not (ctr % mod):
#             writer.add_scalar('data/loss', loss_sum/mod, ctr)
#             print(loss_sum/mod)
#             loss_sum = 0
#         if not (ctr % (10*mod)):
#             trainer.save_model(ctr)
# writer.export_scalars_to_json(os.path.join(log_dir,"all_scalars.json"))
# writer.close()

# while True:
#     for k, sample in enumerate(train_iter):
#         loss = trainer.train_sample(sample)
#         ctr += 1
#         loss_sum += loss.item()
#         if not (ctr % mod):
#             writer.add_scalar('data/loss', loss_sum / mod, ctr)
#             print(loss_sum / mod)
#             loss_sum = 0
#         if not (ctr % (10 * mod)):
#             trainer.save_model(ctr)
# writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
# writer.close()