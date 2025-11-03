import os
import torch


def save_model_ori(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def save_model(args, model, optimizer, current_epoch, prototype):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch, 'prototype': prototype}
    torch.save(state, out)


def save_best_model(args, model, optimizer, current_epoch, prototype):
    out = os.path.join(args.model_path, "checkpoint_best.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch, 'prototype': prototype}
    torch.save(state, out)


def save_last_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_last.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def save_model_open(args, model, optimizer, current_epoch, prototype):
    out = os.path.join(args.model_path, "checkpoint_{}_open.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch, 'prototype': prototype}
    torch.save(state, out)