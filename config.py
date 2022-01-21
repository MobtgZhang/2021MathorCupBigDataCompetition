import argparse
def get_parse_args():
    parser = argparse.ArgumentParser(description='The cars trade network.')
    parser.add_argument('--cuda', action='store_false',help='Wether select cuda as training device.')
    parser.add_argument('--normalize', action='store_false',help='Wether the continuious data is be normalized.')
    parser.add_argument('--data-dir',default='./data',type=str,help='The root path of the data directory.')
    parser.add_argument('--log-dir', default='./log', type=str, help='The log  path of the directory.')
    parser.add_argument('--result-dir', default='./result', type=str, help='The log  path of the directory.')
    parser.add_argument('--batch-size', default=4096, type=int, help='The batch size of the train dataset.')
    parser.add_argument('--graph-batch-size', default=512, type=int, help='The batch size of the test train dataset.')
    parser.add_argument('--neighbor-batch-size', default=32, type=int)
    parser.add_argument('--test-batch-size', default=8192, type=int)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--v-lambda", type=float, default=0.5)
    
    # args.graph_split_size, num_entities, num_relations, args.negative_sample
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--reg-ratio", type=float, default=1e-2)

    
    parser.add_argument('--learning-rate', default=1e-2, type=float, help='The learning rate of the model.')
    parser.add_argument('--epoch-times', default=50, type=int, help='The learning rate of the model.')
    parser.add_argument('--percentage', default=0.7, type=float, help='The percentage of the train dataset.')
    parser.add_argument('--model-name', default="TEIGANN", type=str, help='The percentage of the train dataset.')
    parser.add_argument('--embedding-dim', default=40, type=int, help='The embedding dimension of the model.')
    parser.add_argument('--triple-percentage', default=0.7, type=float, help='The percentage of the train dataset.')    
    parser.add_argument('--n-bases', default=4,type=int,help='The training RGCN model of n-bases.')
    parser.add_argument('--dropout', default=0.2,type=float,help='The dropout rate of model.')
    
    #parser.add_argument('--ent-dim', default=10, type=int, help='The percentage of the train dataset.')    
    parser.add_argument('--hidden-dim', default=6, type=int, help='The percentage of the train dataset.')    
    parser.add_argument('--linear-dim', default=8, type=int, help='The percentage of the train dataset.')    
    parser.add_argument('--time-dim-list', default=[10,10,10], type=list, help='The percentage of the train dataset.')    
    parser.add_argument('--year-span', default=101, type=int, help='The percentage of the train dataset.')
    args = parser.parse_args()
    return args


