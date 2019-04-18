from flowNet.data_utils import MatlabBasedCosegDataset,PickleBasedCosegDataset
from torch.utils.data import dataloader
from flowNet import trainerClass
from tensorboardX import SummaryWriter
import torch
import os
log_dir = '/media/fastData/coSegTraining/secondShot'
writer = SummaryWriter(log_dir=log_dir)

#
# dataset = MatlabBasedCosegDataset(root_dir='/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses',
#                                   example_format='*/rectifiedPatchesSIFT_128X128_withField_*.mat')

dataset = PickleBasedCosegDataset(root_dir='/media/fastData/coSegDataPasses/SintelCleanPasses',
                                  example_format='affnet_*/*/*.pklz')

train_loader = dataloader.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
train_iter = iter(train_loader)
trainer = trainerClass.Trainer(train_dir=log_dir)

ctr = 0
mod = 10
loss_sum = 0
while True:
    for k,sample in enumerate(dataset):
        # print(torch.mean(sample['gt_flow_025'][51, 0, :, :]))
        # print(torch.mean(sample['gt_flow_0125'][51, 0, :, :]))
        # torch.mean(sample['gt_flow_025'][51, 0, :, :]) / torch.mean(sample['gt_flow_0125'][51, 0, :, :]) - 2

        loss = trainer.train_sample(sample)

        ctr += 1
        loss_sum += loss.item()
        if not (ctr % mod):
            writer.add_scalar('data/loss', loss_sum/mod, ctr)
            print(loss_sum/mod)
            loss_sum = 0
        if not (ctr % (10*mod)):
            trainer.save_model(ctr)
writer.export_scalars_to_json(os.path.join(log_dir,"all_scalars.json"))
writer.close()

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