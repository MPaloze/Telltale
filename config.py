import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--checkpoints", type=str, default="./all_checkpoints/checkpoints_ssba_r(pr=0.01)/")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=200)
    parser.add_argument("--lr_C", type=float, default=1e-2)# 1e-3
    parser.add_argument("--schedulerC_milestones", type=list, default=[80, 120])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=120)
    parser.add_argument("--lambda_div", type=float, default=1)
    parser.add_argument("--lambda_norm", type=float, default=100)
    parser.add_argument("--num_workers", type=float, default=0)

    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--p_attack", type=float, default=0.01)
    parser.add_argument("--mask_density", type=float, default=0.032)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')

    return parser
