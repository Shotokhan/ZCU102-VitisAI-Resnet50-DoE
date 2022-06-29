
/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"

#include <cassert>
#include <numeric>


std::unique_ptr<vart::SoftmaxRunner> getSoftmaxRunner(xir::Graph* graph){
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  xir::Subgraph* softmax_subgraph = nullptr;
  for (auto c: children) {
    if (c->get_attr<std::string>("device") == "SMFC" &&
          softmax_subgraph == nullptr) {
        softmax_subgraph = c;
      }
  }
  if (softmax_subgraph != nullptr) {
    auto attrs = xir::Attrs::create();
    auto sfm_runner(std::make_unique<vart::SoftmaxRunner>(softmax_subgraph, attrs.get()));
    return std::move(sfm_runner);
  } else {
    return nullptr;
  }
}


int DPUCalcSoftmax(vart::SoftmaxRunner* sfm_runner, const int8_t *data, size_t size, float* result){
  auto sfm_tensor_input = sfm_runner->get_input_tensors()[0];
  auto sfm_tensor_output = sfm_runner->get_output_tensors()[0];
  auto in_dims = sfm_tensor_input->get_shape();
  auto out_dims = sfm_tensor_output->get_shape();
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  in_dims[0] = size;
  out_dims[0] = size;

  auto data_to_tensor = xir::Tensor::create(sfm_tensor_input->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u});
  inputs.push_back(std::make_unique<CpuFlatTensorBuffer>((int8_t*)data, data_to_tensor.get())); 
  inputsPtr.push_back(inputs[0].get());

  auto result_to_tensor = xir::Tensor::create(sfm_tensor_output->get_name(), out_dims, xir::DataType{xir::DataType::FLOAT, 64u});
  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(result, result_to_tensor.get()));
  outputsPtr.push_back(outputs[0].get());

  auto job_id = sfm_runner->execute_async(inputsPtr, outputsPtr);
  return job_id.first;
}


int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
                   int cntout) {
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  if (shapes->output_mapping.empty()) {
    shapes->output_mapping.resize((unsigned)cntout);
    std::iota(shapes->output_mapping.begin(), shapes->output_mapping.end(), 0);
  }
  for (int i = 0; i < cntin; i++) {
    auto dim_num = inputTensors[i]->get_shape().size();
    if (dim_num == 4) {
      shapes->inTensorList[i].channel = inputTensors[i]->get_shape().at(3);
      shapes->inTensorList[i].width = inputTensors[i]->get_shape().at(2);
      shapes->inTensorList[i].height = inputTensors[i]->get_shape().at(1);
      shapes->inTensorList[i].size =
          inputTensors[i]->get_element_num() / inputTensors[0]->get_shape().at(0);
    } else if (dim_num == 2) {
      shapes->inTensorList[i].channel = inputTensors[i]->get_shape().at(1);
      shapes->inTensorList[i].width = 1;
      shapes->inTensorList[i].height = 1;
      shapes->inTensorList[i].size =
          inputTensors[i]->get_element_num() / inputTensors[0]->get_shape().at(0);
    }
  }
  for (int i = 0; i < cntout; i++) {
    auto dim_num = outputTensors[shapes->output_mapping[i]]->get_shape().size();
    if (dim_num == 4) {
      shapes->outTensorList[i].channel =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(3);
      shapes->outTensorList[i].width =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(2);
      shapes->outTensorList[i].height =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(1);
      shapes->outTensorList[i].size =
          outputTensors[shapes->output_mapping[i]]->get_element_num() /
          outputTensors[shapes->output_mapping[0]]->get_shape().at(0);
    } else if (dim_num == 2) {
      shapes->outTensorList[i].channel =
          outputTensors[shapes->output_mapping[i]]->get_shape().at(1);
      shapes->outTensorList[i].width = 1;
      shapes->outTensorList[i].height = 1;
      shapes->outTensorList[i].size =
          outputTensors[shapes->output_mapping[i]]->get_element_num() /
          outputTensors[shapes->output_mapping[0]]->get_shape().at(0);
    }
  }
  return 0;
}

static int find_tensor(std::vector<const xir::Tensor*> tensors,
                       const std::string& name) {
  int ret = -1;
  for (auto i = 0u; i < tensors.size(); ++i) {
    if (tensors[i]->get_name().find(name) != std::string::npos) {
      ret = (int)i;
      break;
    }
  }
  assert(ret != -1);
  return ret;
}
int getTensorShape(vart::Runner* runner, GraphInfo* shapes, int cntin,
                   std::vector<std::string> output_names) {
  for (auto i = 0u; i < output_names.size(); ++i) {
    auto idx = find_tensor(runner->get_output_tensors(), output_names[i]);
    shapes->output_mapping.push_back(idx);
  }
  getTensorShape(runner, shapes, cntin, (int)output_names.size());
  return 0;
}
