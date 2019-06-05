from flowNet.data_utils import MatlabBasedCosegDataset,PickleBasedCosegDataset,tfRecordBasedCosegDataset
from torch.utils.data import dataloader
from flowNet import trainerClass
from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
log_dir = '/media/fastData/coSegTraining/8_batched_025_res'
writer = SummaryWriter(log_dir=log_dir)
trainer = trainerClass.Trainer(train_dir=log_dir)

#
# dataset = MatlabBasedCosegDataset(root_dir='/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses',
#                                   example_format='*/rectifiedPatchesSIFT_128X128_withField_*.mat')

dataset = tfRecordBasedCosegDataset(record_path='/media/fastData/coSegDataPasses/SintelCleanPasses/all_data_tf.record')
# dataset = PickleBasedCosegDataset(root_dir='/media/fastData/coSegDataPasses/SintelCleanPasses',
#                                   example_format='affnet_*/*/*.pklz')
ctr = 0
mod = 100
loss_sum = 0
total_loss_sum = 0
while True:
    sample = dataset.get_batch(batch_size=64)
    if not (ctr % mod):
        ctr += 1
        loss = trainer.train_sample(sample, writer, ctr)
        loss_sum += loss.item()
        total_loss_sum += loss.item()
        writer.add_scalar('data/loss', loss_sum / mod, ctr)
        print('last loss: ', loss_sum / mod)
        loss_sum = 0
        print('total loss: ', total_loss_sum / ctr)
    else:
        ctr += 1
        loss = trainer.train_sample(sample)
        loss_sum += loss.item()
        total_loss_sum += loss.item()
    if not (ctr % (10 * mod)):
        trainer.save_model(ctr)
writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))


# train_loader = dataloader.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
# train_iter = iter(train_loader)
# ctr = 0
# mod = 100
# loss_sum = 0
# for k, sample in enumerate(train_iter):
#     loss = trainer.train_sample(sample)
#     ctr += 1
#     loss_sum += loss.item()
#     if not (ctr % mod):
#         writer.add_scalar('data/loss', loss_sum / mod, ctr)
#         print(loss_sum / mod)
#         loss_sum = 0
#     # if not (ctr % (10 * mod)):
#     #     trainer.save_model(ctr)
# writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
#
#
#
#
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
#
# # while True:
# #     for k, sample in enumerate(train_iter):
# #         loss = trainer.train_sample(sample)
# #         ctr += 1
# #         loss_sum += loss.item()
# #         if not (ctr % mod):
# #             writer.add_scalar('data/loss', loss_sum / mod, ctr)
# #             print(loss_sum / mod)
# #             loss_sum = 0
# #         if not (ctr % (10 * mod)):
# #             trainer.save_model(ctr)
# # writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
# writer.close()