import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--source_data', type=str, help="dataset name",
                                 default='ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit')  # cora+citeseer+wikics+computer+photo+history+instagram
        self.parser.add_argument('--target_data', type=str, help="dataset name",
                                 default='cora+wikics+history')  # cora+citeseer+wikics+computer+photo+history+instagram
        self.parser.add_argument('--ckpt', type=str, help="the name of checkpoint",
                                 default='our_mask_aug_666')# pretrained_graphclip/our_mask2_10/our_mask_prototype/our_mask_prototype-cluster/our_backbone
        self.parser.add_argument('--resume', type=str, help="the name of checkpoint",
                                 default='./checkpoints/our_mask_prototype_zhengliu.pt')
        self.parser.add_argument('--is_resume', type=bool, help="the name of checkpoint",
                                 default=False)
        self.parser.add_argument('--text_ids', type=str, default='/media/date/fcl/all-MiniLM-L6-v2')
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=10)
        self.parser.add_argument('--batch_size', type=int, help="the batch size", default=800)

        # 不经常修改的参数
        self.parser.add_argument('--dataset', type=str, help="dataset name",
                                 default='children')
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.5)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu',
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        self.parser.add_argument('--last_activation', action='store_true',
                                 help="the last layer will use activation function or not")

        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam',
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-5)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=1e-5)


        # used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=256)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=3)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling",
                                 default='rw',
                                 choices=['rw', 'khop', 'shadow'])

    def parse_args(self):
        return self.parser.parse_args()
