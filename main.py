import os

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking
from transforms import MinMaxNorm
from torchmetrics.classification import MultilabelF1Score

from dataset import AnuraSet
from models import mobilenetv3

from train import train, adjust_learning_rate
from val import validate

from transforms import TimeShift
from models import model_dim

from args import args

import warnings
warnings.filterwarnings("ignore")

NUM_CLASSES = 42

root_dir = args.rootdir

resamp = Resample(orig_freq=22050, new_freq=args.sr)

min_max_norm = MinMaxNorm()

mel_spectrogram = MelSpectrogram(n_fft=512, hop_length=128, n_mels=128)

time_mask = TimeMasking(time_mask_param=args.tmask)

freq_mask = FrequencyMasking(freq_mask_param=args.fmask)

tshift = TimeShift(Tshift=376) #376 = length of timesteps for 3s melspectrogram with the config above

train_transform = nn.Sequential(
    resamp,
    mel_spectrogram,
    AmplitudeToDB(),
    min_max_norm,              
    tshift,
    *[freq_mask for _ in range(args.fstripe)],
    *[time_mask for _ in range(args.tstripe)],
).to(args.device)

val_transform = nn.Sequential(
    resamp,
    mel_spectrogram,
    AmplitudeToDB(),
    min_max_norm,
).to(args.device)

ANNOTATIONS_FILE = os.path.join(root_dir, 'metadata.csv')

AUDIO_DIR = os.path.join(root_dir, 'audio')

training_data = AnuraSet(
    annotations_file=ANNOTATIONS_FILE, 
    audio_dir=AUDIO_DIR, 
    train=True,
)
print(f"There are {len(training_data)} samples in the training set.")

val_data = AnuraSet(
    annotations_file=ANNOTATIONS_FILE, 
    audio_dir=AUDIO_DIR, 
    train=False,
)
print(f"There are {len(val_data)} samples in the test set.")

train_dataloader = DataLoader(training_data, 
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=args.workers,
)

val_dataloader = DataLoader(val_data, 
    batch_size=args.bs,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=args.workers,
)

encoder = mobilenetv3()

projector = nn.Linear(model_dim[args.model], NUM_CLASSES)

encoder.to(args.device)
projector.to(args.device)    
trainable_params = list(encoder.parameters()) + list(projector.parameters())

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.AdamW(trainable_params, lr=args.lr)
metric_fn = MultilabelF1Score(num_labels=NUM_CLASSES).to(args.device)

scaler = torch.cuda.amp.GradScaler()

save_path = os.path.join(root_dir, 'ckpt')
os.makedirs(save_path, exist_ok=True)
pt_filepath = os.path.join(save_path, 'model.pt')    

best_score = None

print('Starting training')
for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}")

    adjust_learning_rate(optimiser, epoch, args)

    loss_train = train(encoder, projector, train_dataloader, train_transform, loss_fn, optimiser, scaler, args)
    metric_val, f1_freq, f1_common, f1_rare = validate(encoder, projector, val_dataloader, val_transform, metric_fn, args.device)

    if best_score is None:
        best_score = metric_val
        best_freq = f1_freq; best_common = f1_common; best_rare = f1_rare
        best_encoder_state = encoder.state_dict()
        best_projector_state = projector.state_dict()

        if args.save:
            torch.save(nn.Sequential(encoder, projector), pt_filepath)
        
    if metric_val > best_score:
        best_score = metric_val
        best_freq = f1_freq; best_common = f1_common; best_rare = f1_rare
        best_encoder_state = encoder.state_dict()
        best_projector_state = projector.state_dict()

        if args.save:
            torch.save(nn.Sequential(encoder, projector), pt_filepath)

    print(f"Loss train: {loss_train}\tMacro F1-score: {metric_val}\tFreq: {f1_freq}\tCommon: {f1_common}\tRare: {f1_rare}")
    
print("Finished training")