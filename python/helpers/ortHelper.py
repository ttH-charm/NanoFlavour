import numpy as np
import json
import onnxruntime


def _pad(a, min_length, max_length, value=0, dtype='float32'):
    if len(a) > max_length:
        return a[:max_length].astype(dtype)
    elif len(a) < min_length:
        x = (np.ones(min_length) * value).astype(dtype)
        x[:len(a)] = a.astype(dtype)
        return x
    else:
        return a.astype('float32')


class Preprocessor:

    def __init__(self, preprocess_file, debug_mode=False):
        with open(preprocess_file) as fp:
            self.prep_params = json.load(fp)
        self.debug = debug_mode

    def preprocess(self, inputs):
        data = {}
        for group_name in self.prep_params['input_names']:
            data[group_name] = []
            info = self.prep_params[group_name]
            for var in info['var_names']:
                a = np.array(inputs[var], dtype='float32')
                a = (a - info['var_infos'][var]['median']) * info['var_infos'][var]['norm_factor']
                a = np.clip(a, info['var_infos'][var].get('lower_bound', -5),
                            info['var_infos'][var].get('upper_bound', 5))
                try:
                    a = _pad(a, info['var_length'], info['var_length'])
                except KeyError:
                    a = _pad(a, info.get('min_length', 0), info.get('max_length', None))
                a = np.expand_dims(a, axis=0)
                if self.debug:
                    print(var, inputs[var], a)
                data[group_name].append(a)
            data[group_name] = np.nan_to_num(np.stack(data[group_name], axis=1))
        return data


class ONNXRuntimeHelper:

    def __init__(self, preprocess_file, model_files, output_prefix='score'):
        self.preprocessor = Preprocessor(preprocess_file)
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sessions = [onnxruntime.InferenceSession(model_path, sess_options=options,
                                                      providers=['CPUExecutionProvider']) for model_path in model_files]
        self.k_fold = len(self.sessions)
        self.output_names = [output_prefix + '_' + n for n in self.preprocessor.prep_params['output_names']]
        print('Loaded ONNX models:\n  %s\npreprocess file:\n  %s' % ('\n  '.join(model_files), str(preprocess_file)))

    def predict(self, inputs, model_idx=None):
        data = self.preprocessor.preprocess(inputs)
        if model_idx is not None:
            outputs = self.sessions[model_idx].run([], data)[0][0]
        else:
            outputs = [sess.run([], data)[0][0] for sess in self.sessions]
            outputs = np.stack(outputs, axis=0).mean(axis=0)
        outputs = {n: v for n, v in zip(self.output_names, outputs)}
        return outputs
