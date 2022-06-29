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

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
#include <chrono>
#include <utility>
#include <thread>
#include <unistd.h>
#include <stdlib.h>

#include "common.h"
#include "resnet_50_utils.h"

using namespace std;
using namespace cv;

GraphInfo shapes;

const string wordsPath = "./";


void dnn_stub(bool verbose, int t_id, vector<double>& exec_times) {
  std::vector<double> elapsed_times;

  for (int i=0; i<5; i++) {
    auto start = std::chrono::system_clock::now();

    sleep(1);
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed = elapsed_seconds.count();

    if (verbose)
      std::cout << "Elapsed time: " << elapsed << " s" << std::endl;

    elapsed_times.push_back(elapsed);
  }

  double total_exec_time = 0;
  for (auto t: elapsed_times) {
    total_exec_time += t;
  }
  if (verbose)
    cout << "Total exec time: " << total_exec_time << endl;
  exec_times.at(t_id) = total_exec_time;
  return;
}

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return vector of elapsed times, one for each batch
 */
void runResnet50(char* graph_name, bool post_process, bool hw_softmax, bool verbose, int t_id, vector<double>& exec_times,
string baseImagePath) {
  auto graph = xir::Graph::deserialize(graph_name);
  auto subgraph = get_dpu_subgraph(graph.get());
  std::unique_ptr<vart::SoftmaxRunner> sfm_runner = nullptr;
  if (hw_softmax) {
    sfm_runner = getSoftmaxRunner(graph.get());
    if (sfm_runner != nullptr) {
      if (verbose)
        cout << "Hardware softmax enabled, runner created" << endl;
    } else {
      if (verbose)
        cout << "Hardware softmax core not found in subgraph, falling back to software softmax" << endl;
      hw_softmax = false;
    }
  }

  if (verbose){
    CHECK_EQ(subgraph.size(), 1u)
        << "resnet50 should have one and only one dpu subgraph.";
    LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
    /*create runner*/
    std::cout << "Creating runner..." << std::endl;
  }
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  if (verbose)
    std::cout << "Create runner ok" << std::endl;

  /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);
  /* Mean value for ResNet50 specified in Caffe prototxt */
  std::vector<double> elapsed_times;
  vector<string> kinds, images;
  
  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* Load all kinds words.*/
  LoadWords(wordsPath + "words.txt", kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: No words exist in file words.txt." << endl;
    return;
  }
  
  float mean[3] = {104, 107, 123};

  /* get in/out tensors and dims*/
  outputTensors = runner->get_output_tensors();
  inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  /*get shape info*/
  int outSize = shapes.outTensorList[0].size;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;

  int batchSize = in_dims[0];
  if (verbose) 
    std::cout << "Batch max size: " << batchSize << std::endl;

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  vector<Mat> imageList;
  int8_t* imageInputs = new int8_t[inSize * batchSize];

  float* softmax = new float[outSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  if (verbose)
    std::cout << "Tensors allocation okay" << std::endl;

  /*run with batch*/
  for (unsigned int n = 0; n < images.size(); n += batchSize) {

    unsigned int runSize =
        (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;

    if (verbose)
      std::cout << "Run size: " << runSize << std::endl;

    for (unsigned int i = 0; i < runSize; i++) {
      Mat image = imread(baseImagePath + images[n + i]);

      /*image pre-process*/
      Mat image2;
      resize(image, image2, Size(inHeight, inWidth), 0, 0);
      for (int h = 0; h < inHeight; h++) {
        for (int w = 0; w < inWidth; w++) {
          for (int c = 0; c < 3; c++) {
            imageInputs[i * inSize + h * inWidth * 3 + w * 3 + c] =
                (int8_t)((image2.at<Vec3b>(h, w)[c] - mean[c]) * input_scale);
          }
        }
      }
      imageList.push_back(image);
    }
    
    if (verbose)
      std::cout << "Image read and resize okay" << std::endl;

    in_dims[0] = imageList.size();
    // TODO: print size
    out_dims[0] = batchSize;

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u})));

    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));

    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims, xir::DataType{xir::DataType::XINT, 8u})));

    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));

    if (verbose)
      std::cout << "Tensors refactoring okay" << std::endl;

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    /*run*/

    auto start = std::chrono::system_clock::now();

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    
    runner->wait(job_id.first, -1);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed = elapsed_seconds.count();

    if (verbose)
      std::cout << "Total elapsed time: " << elapsed << std::endl;

    elapsed_times.push_back(elapsed);
  

    if (post_process && verbose) {
      for (unsigned int i = 0; i < runSize; i++) {
        cout << "\nImage : " << images[n + i] << endl;
        /* Calculate softmax on CPU and display TOP-5 classification results */
        if (hw_softmax){
          int job_id = DPUCalcSoftmax(sfm_runner.get(), &FCResult[i * outSize], outSize, softmax);
          sfm_runner->wait(job_id, -1);
        } else {
          CPUCalcSoftmax(&FCResult[i * outSize], outSize, softmax, output_scale);
        }
        TopK(softmax, outSize, 5, kinds);
      }
    }

    imageList.clear();
    inputs.clear();
    outputs.clear();
  }
  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;

  double total_exec_time = 0;
  for (auto t: elapsed_times) {
    total_exec_time += t;
  }
  exec_times.at(t_id) = total_exec_time;
  return;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char* argv[]) {
  // Check args
  int number_of_threads = 1;
  std::string mode = "run";
  bool post_process = true;
  bool hw_softmax = false;
  bool verbose = false;
  string baseImagePath = "./";
  std::vector<std::string> paths;
  if (argc < 2 or cmdOptionExists(argv, argv+argc, "-h")) {
    cout << "Usage: " << argv[0] <<
     " [model_file] {--num-threads NUM} {--sim} {--no-postprocess} {--base-image-path NEW_PATH}"
     " {--hw-softmax} {--verbose} {--diff-path-for-thread}" << endl;
     cout << "Note: --base-image-path and --diff-path-for-thread are mutually exclusive" << endl;
     cout << "--diff-path-for-thread searches in ./images_0 , ./images_1 ecc" << endl;
    return -1;
  }
  if (cmdOptionExists(argv, argv+argc, "--verbose")) {
    verbose = true;
  }
  if (cmdOptionExists(argv, argv+argc, "--num-threads")) {
    number_of_threads = atoi(getCmdOption(argv, argv+argc, "--num-threads"));
  }
  if (cmdOptionExists(argv, argv+argc, "--sim")) {
    mode = "sim";
  }
  if (cmdOptionExists(argv, argv+argc, "--no-postprocess")) {
    post_process = false;
  }
  if (cmdOptionExists(argv, argv+argc, "--hw-softmax")) {
    hw_softmax = true;
  }
  if (cmdOptionExists(argv, argv+argc, "--diff-path-for-thread")) {
    string base("./images_");
    for (int i=0; i<number_of_threads; i++){
      string path = base + std::to_string(i) + string("/");
      paths.push_back(path);
    }
  } else {
    if (cmdOptionExists(argv, argv+argc, "--base-image-path")){
      char* path = getCmdOption(argv, argv+argc, "--base-image-path");
      if (path) {
        baseImagePath = std::string(path);
        if (verbose) {
          std::cout << "Base image path set to: " << baseImagePath << " instead of cwd" << std::endl;
          if (post_process) {
            std::cout << "This will also be the path for classified images" << std::endl;
          }
        }
      }
    }
  }

  if (verbose)
    std::cout << "Number of threads: " << number_of_threads << std::endl;
    
  std::vector<double> exec_times;
  for (int i=0; i<number_of_threads; i++){
    exec_times.push_back(0);
  }
  
  std::vector<std::thread> dnn_threads;

  auto start = std::chrono::system_clock::now();
  
  if (mode != "sim") {
    for (int i=0; i<number_of_threads; i++) {
      if (cmdOptionExists(argv, argv+argc, "--diff-path-for-thread") &&
       !cmdOptionExists(argv, argv+argc, "--base-image-path")){

        std::thread t(runResnet50, argv[1], post_process, hw_softmax, verbose, i, std::ref(exec_times), paths.at(i));
        dnn_threads.push_back(std::move(t));

       } else {

        std::thread t(runResnet50, argv[1], post_process, hw_softmax, verbose, i, std::ref(exec_times), baseImagePath);
        dnn_threads.push_back(std::move(t));

       }
      
    }
    // runResnet50(argv[1], post_process, hw_softmax, verbose, 0, exec_times, baseImagePath);
  } else {
    for (int i=0; i<number_of_threads; i++) {
      std::thread t(dnn_stub, verbose, i, std::ref(exec_times));
      dnn_threads.push_back(std::move(t));
    }
    // dnn_stub(verbose, 0, exec_times);
  }

  for (std::thread& t: dnn_threads) {
    // only threads that are not destructed can be joined; https://thispointer.com/c11-how-to-create-vector-of-thread-objects/
    if (t.joinable())
      t.join();
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  auto elapsed = elapsed_seconds.count();

  cout << "Total experiment time: " << elapsed << endl;

  for (int i=0; i<number_of_threads; i++) {
    std::cout << "Thread " << i << " execution time: " << exec_times[i] << " seconds" << std::endl;
  }
  return 0;
}
