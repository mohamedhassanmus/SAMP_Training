import os
import os.path as osp
import torch
from src import misc_utils
from src.cmd_parser import parse_config
import numpy as np
import onnx
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api='blas'):
        args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    args_dict['data_dir'] = osp.expandvars(args_dict.get('data_dir'))
    args_dict['output_dir'] = osp.expandvars(args_dict.get('output_dir'))

    data_dir = args_dict.get('data_dir')
    output_dir = args_dict.get('output_dir')
    checkpoints_dir = osp.join(output_dir, 'checkpoints')
    args_dict['checkpoints_dir'] = checkpoints_dir

    model_dir = "~/SAMP_workspace/onnx_models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = osp.join(model_dir, "{}.onnx".format(args.model_name))

    rng = np.random.RandomState(23456)
    args_dict['rng'] = rng

    device = torch.device("cuda" if args.use_cuda else "cpu")
    if args_dict.get('float_dtype') == 'float64':
        dtype = torch.float64
    elif args_dict.get('float_dtype') == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(args_dict.get('float_dtype')))

    model = misc_utils.load_model_checkpoint(device=device, **args_dict).to(device)
    model.eval()

    if args.model_name == 'MotionNet':
        x1 = torch.randn(args.batch_size, args.state_dim, device=device, dtype=dtype)
        x2 = torch.randn(args.batch_size, 2048, device=device, dtype=dtype)

        z = torch.randn(args.batch_size, args.z_dim, device=device, dtype=dtype)

        input_names = ["z", "x1", "x2"]
        output_names = ["Y_hat"]
        torch.onnx.export(model.decoder, (z, x1, x2), model_path, verbose=False, opset_version=9, input_names=input_names,
                          output_names=output_names)
        print("**** ONNX model has been exported to {}".format(model_path))
        torch_out = model.decoder(z, x1, x2)
        print("output dim = {}".format(torch_out.shape))

        onnx_model = onnx.load(model_path)
        onnx.shape_inference.infer_shapes(onnx_model)
        ort_session = onnxruntime.InferenceSession(model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(z),
                      ort_session.get_inputs()[1].name: to_numpy(x1),
                      ort_session.get_inputs()[2].name: to_numpy(x2)}


        ort_outs = ort_session.run(None, ort_inputs)
        print("ONNX output shape {}".format(ort_outs[0].shape))
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    elif args.model_name == 'GoalNet':
        cond = torch.randn(args.batch_size, args.interaction_dim, device=device, dtype=dtype)
        z = torch.randn(args.batch_size, args.z_dim_goalnet, device=device, dtype=dtype)
        Y = torch.randn(args.batch_size, args.input_dim_goalnet, device=device, dtype=dtype)

        input_names = ["Z", "Cond"]
        output_names = ["Y_hat"]

        torch.onnx.export(model.decoder, (z, cond), model_path, verbose=False, opset_version=9, input_names=input_names,
                          output_names=output_names)
        print("**** ONNX model has been exported to {}".format(model_path))
        torch_out = model.decoder(z, cond)
        print("output dim = {}".format(torch_out.shape))

        onnx_model = onnx.load(model_path)
        onnx.shape_inference.infer_shapes(onnx_model)
        ort_session = onnxruntime.InferenceSession(model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(z),
                      ort_session.get_inputs()[1].name: to_numpy(cond)
                      }

        ort_outs = ort_session.run(None, ort_inputs)
        print("ONNX output shape {}".format(ort_outs[0].shape))
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")