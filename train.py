from tqdm import tqdm
import numpy as np
import torch
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(encoder, projector, data_loader, transform, loss_fn, optimiser, scaler, args):
    
    num_batches = len(data_loader)
    encoder.train()
    projector.train()

    loss_total = 0.0

    for batch_idx, (input, target, index) in enumerate(tqdm(data_loader)):
        input, target = input.to(args.device), target.to(args.device)
        if args.mix == 'mixup':   
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                mixed_x, y_a, y_b, lam = mixup_data(transform(input), target)
                prediction = projector(encoder((mixed_x)))
                loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
        if args.mix == 'manmixup':   
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x = transform(input)
                z = encoder(x)
                mixed_z, y_a, y_b, lam = mixup_data(z, target)
                prediction = projector(mixed_z)
                loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
        elif args.mix == 'multimix':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x = transform(input)
                z = encoder(x)
                mixed_z, mixed_y, lam = multimix_data(z, target, args.mixnum, args.alpha1, args.alpha2)
                prediction = projector(mixed_z)
                loss = multimix_criterion(loss_fn, prediction, mixed_y, lam)
        elif args.mix == 'mix2':
            p = np.random.random(1)
            if p < 1/2: # 50% manmixup
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    x = transform(input)
                    z = encoder(x)
                    mixed_z, y_a, y_b, lam = mixup_data(z, target)
                    prediction = projector(mixed_z)
                    loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)           
            elif p < 3/4: # 25% multimix
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):    
                    x = transform(input)
                    z = encoder(x)
                    mixed_z, mixed_y, lam = multimix_data(z, target, args.mixnum, args.alpha1, args.alpha2)
                    prediction = projector(mixed_z)
                    loss = multimix_criterion(loss_fn, prediction, mixed_y, lam)
            else: # 25% mixup
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    mixed_x, y_a, y_b, lam = mixup_data(transform(input), target)
                    prediction = projector(encoder((mixed_x)))
                    loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prediction = projector(encoder(transform(input)))
                loss = loss_fn(prediction, target)
    
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        loss_total += loss.item()  

    loss_total /= num_batches   

    return loss_total

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample(((x.size(0)), 1))
        lam = lam.to(x.device)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    l = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return l.mean()

def multimix_data(z, y, num_mixed_examples, alpha1=1.0, alpha2=1.0, use_cuda=True):
    '''Returns mixed inputs, multiple pairs of targets, and multiple lambda values for each example in the batch'''
    batch_size = z.size()[0]

    alpha_values = np.random.uniform(alpha1, alpha2, size=batch_size)
    lambda_vectors = torch.from_numpy(np.random.dirichlet(alpha_values, num_mixed_examples)).float()

    if use_cuda:
        lambda_vectors = lambda_vectors.cuda()

    mixed_z = torch.matmul(lambda_vectors, z)
    mixed_y = torch.matmul(lambda_vectors, y)

    return mixed_z, mixed_y, lambda_vectors

def multimix_criterion(criterion, pred, mixed_y, lambda_vectors):
    '''Compute the loss for each mixed example and lambda vector'''
    mixed_loss = torch.sum(lambda_vectors.unsqueeze(2) * criterion(pred, mixed_y), dim=1)
    return torch.mean(mixed_loss)