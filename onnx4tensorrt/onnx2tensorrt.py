# Script for ONNX 2 Tensorrt conversion
# credits to: Shakhizat Nurgaliyev (https://github.com/shahizat)

import os
import torch
import argparse
import numpy as np
import pickle
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # this is important

from utils.args_utils import str2type
from utils.language_utils import tokens2description
from utils.image_utils import preprocess_image

from onnx4tensorrt.End_ExpansionNet_v2_onnx_tensorrt import create_pad_mask, create_no_peak_and_pad_mask
from onnx4tensorrt.End_ExpansionNet_v2_onnx_tensorrt import NUM_FEATURES, MAX_DECODE_STEPS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX 2 TensorRT')
    parser.add_argument('--onnx_path', type=str, default='./rf_model.onnx')
    parser.add_argument('--engine_path', type=str, default='./model_engine.trt')
    parser.add_argument('--data_type', type=str2type, default='fp32')
    args = parser.parse_args()

    with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    # Build TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    # The Onnx path is used for Onnx models.
    def build_engine_onnx(onnx_file_path, data_type):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        builder.max_batch_size = 1
        config.max_workspace_size = 1 << 32  # sono in bytes quindi 2^32 -> 4GB
        # Load the Onnx model and parse it in order to populate the TensorRT network.

        if data_type == 'fp32':
            pass  # do nothing, is the default
        elif data_type == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("fp16 is supported. Setting up fp16.")
            else:
                print("fp16 is not supported. Using the fp32 instead.")
        else:
            raise ValueError("Unsupported type. Only the following types are supported: " +
                             "fp32, fp16, int8.")

        with open(onnx_file_path, "rb") as onnx_file:
            if not parser.parse(onnx_file.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        return builder.build_engine(network, config)

    img_size = 384
    image = preprocess_image('./demo_material/napoleon.jpg', img_size)

    # generate optimized graph
    file_already_exist = os.path.isfile(args.engine_path)
    if file_already_exist:
        print("Engine File:" + str(args.engine_path) + " already exists, loading engine instead of ONNX file.")
        with open(args.engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
    else:
        print("Building TensorRT engine from ONNX")
        if args.data_type == 'fp32':
            engine = build_engine_onnx(args.onnx_path, args.data_type)
        elif args.data_type == 'fp16':
            engine = build_engine_onnx(args.onnx_path + '_fp16', args.data_type)
        with open(args.engine_path, "wb") as f:
            f.write(engine.serialize())
        print("Engine written in: " + str(args.engine_path))

    print("Finished Building.")
    # engine = build_engine('./trt_fp.engine')
    context = engine.create_execution_context()
    print("Created execution context.")

    print("Testing first image on TensorRT")
    batch_size = 1
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        print("Binding type: " + str(dtype) + " is input: " + str(engine.binding_is_input(binding)))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    # Masking creation - - - - -
    enc_exp_list = [32, 64, 128, 256, 512]
    dec_exp = 16
    num_heads = 8
    enc_mask = create_pad_mask(mask_size=(batch_size, sum(enc_exp_list), NUM_FEATURES),
                               pad_row=[0], pad_column=[0]).contiguous()
    no_peak_mask = create_no_peak_and_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS),
                                               num_pads=[0]).contiguous()
    cross_mask = create_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, NUM_FEATURES),
                                 pad_row=[0], pad_column=[0]).contiguous()
    cross_mask = 1 - cross_mask

    fw_dec_mask = no_peak_mask.unsqueeze(2).expand(batch_size, MAX_DECODE_STEPS,
                                                   dec_exp, MAX_DECODE_STEPS).contiguous(). \
        view(batch_size, MAX_DECODE_STEPS * dec_exp, MAX_DECODE_STEPS)

    bw_dec_mask = no_peak_mask.unsqueeze(-1).expand(batch_size,
                                                    MAX_DECODE_STEPS, MAX_DECODE_STEPS,
                                                    dec_exp).contiguous(). \
        view(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS * dec_exp)

    atten_mask = cross_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

    # - - - - - - - - - -

    # Set input values
    if args.data_type == 'fp32':
        inputs[0]['host'] = np.ravel(image).astype(np.float32)
        inputs[1]['host'] = np.array([sos_idx]).astype(np.int32)
        inputs[2]['host'] = np.array(enc_mask).astype(np.int32)
        inputs[3]['host'] = np.array(fw_dec_mask).astype(np.int32)
        inputs[4]['host'] = np.array(bw_dec_mask).astype(np.int32)
        inputs[5]['host'] = np.array(atten_mask).astype(np.int32)
    elif args.data_type == 'fp16':
        inputs[0]['host'] = np.ravel(image).astype(np.float16)
        inputs[1]['host'] = np.array([sos_idx]).astype(np.int32)
        inputs[2]['host'] = np.array(enc_mask).astype(np.int32)
        inputs[3]['host'] = np.array(fw_dec_mask).astype(np.float16)
        inputs[4]['host'] = np.array(bw_dec_mask).astype(np.float16)
        inputs[5]['host'] = np.array(atten_mask).astype(np.int32)

    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # Execute model
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # Synchronize the stream
    stream.synchronize()
    print(outputs[0]['host'].tolist())
    output_caption = tokens2description(outputs[0]['host'].tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
    output_probs = outputs[1]['host'].tolist()
    print(output_caption)
    print(output_probs)
