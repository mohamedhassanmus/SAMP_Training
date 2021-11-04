import configargparse


def parse_config():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'SAMP'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SAMP')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--data_dir', type=str, default='$base_dir')
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--checkpoints_dir', type=str)
    parser.add_argument('--checkpoint_path', type=str, default=None, help='')
    parser.add_argument('--save_checkpoints', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--load_latest_checkpoint', type=lambda x: x.lower() in ['true', '1'], default=True)

    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--reduction', default='mean', type=str,
                        choices=['mean', 'sum'],
                        help='')
    parser.add_argument('--loss_type', default='mse', type=str,
                        choices=['l1', 'mse', 'bce'],
                        help='')
    parser.add_argument('--float_dtype', type=str, default='float32', help='The types of floats used')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reduce_lr', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--test', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--normalize', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--save_norm_data', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--keep_prob', type=float, default=1.0)

    # Architecture
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--h_dim_gate', type=int, default=256)
    parser.add_argument('--h_dim_encoder', type=int, default=256)
    parser.add_argument('--h_dim_I', type=int, default=256)
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--activation_fn', default='sigmoid', type=str, choices=['sigmoid', 'tanh', 'relu', 'None', ''],
                        help='')

    # Features
    parser.add_argument('--start_pose', type=int, default=None)
    parser.add_argument('--start_goal', type=int, default=None)
    parser.add_argument('--start_environment', type=int, default=None)
    parser.add_argument('--start_interaction', type=int, default=None)
    parser.add_argument('--start_gating', type=int, default=None)
    parser.add_argument('--dim_gating', type=int, default=None)
    parser.add_argument('--interaction_dim', type=int, default=2048)

    ## VAE
    parser.add_argument('--cond_dim', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=5)
    parser.add_argument('--kl_w', type=float, default=0.1)

    # scheduled sampling
    parser.add_argument('--state_dim', type=int, default=595)
    parser.add_argument('--I_dim', type=int, default=595)
    parser.add_argument('--num_actions', type=int, default=5)

    parser.add_argument('--scheduled_sampling', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--L', type=int, default=6, help="rollout_lenght")
    parser.add_argument('--C1', type=int, default=20, help="boundary 1")
    parser.add_argument('--C2', type=int, default=40, help="boundary 2")

    ## GOAL Net
    parser.add_argument('--input_dim_goalnet', type=int, default=6)
    parser.add_argument('--h_dim_goalnet', type=int, default=256)
    parser.add_argument('--z_dim_goalnet', type=int, default=3)

    args = parser.parse_args()
    args_dict = vars(args)

    return args, args_dict
