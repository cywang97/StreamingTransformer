import logging
import torch


def set_deterministic_pytorch(args):
    """Ensures pytorch produces deterministic results depending on the program arguments

    :param Namespace args: The program arguments
    """
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # considering reproducibility
    # remove type check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # https://github.com/pytorch/pytorch/issues/6351
    )
    if args.debugmode < 2:
        logging.info("torch type check is disabled")
    # use deterministic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logging.info("torch cudnn deterministic is disabled")
