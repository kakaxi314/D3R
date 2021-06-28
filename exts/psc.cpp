//
// Created by jie on 09/02/19.
//
//#include <torch/torch.h>
#include <torch/extension.h>
//#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <tuple>


at::Tensor Conv_F(
        const at::Tensor &input, // BCiHW
        const at::Tensor &weight, // CoCiKK
        const at::Tensor& bias,  // Co
        at::IntList padding, //at::IntArrayRef
        at::IntList stride, //at::IntArrayRef
        at::IntList dilation, //at::IntArrayRef
        int64_t groups, bool benchmark, bool deterministic
) {
    return cudnn_convolution(
            input, weight, bias,
            padding, stride, dilation, groups, benchmark, deterministic);

}


std::tuple <at::Tensor, at::Tensor> Conv_B(
        const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &weight,
        at::IntList padding, at::IntList stride, at::IntList dilation, int64_t groups,
        bool benchmark, bool deterministic, bool allow_tf32, std::array<bool, 2> output_mask) {

    return cudnn_convolution_backward(
            input, grad_output, weight, padding, stride, dilation, groups,
            benchmark, deterministic, allow_tf32, output_mask);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("Conv_F", &Conv_F, "conv (CUDA)");
m.def("Conv_B", &Conv_B, "conv (CUDA)");
}

