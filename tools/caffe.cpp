#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_async_params_gpu.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu2.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu3.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu4.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu5.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu6.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu7.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu8.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu9.hpp"
#include "caffe/parallel/mpi_nocomm_gpu.hpp"
#include "caffe/parallel/mpi_nccl_async.hpp"
#include "caffe/parallel/mpi_nccl_sync.hpp"
#include "caffe/parallel/mpi_sync_gpu.hpp"
#include "caffe/parallel/mpi_sync_params_gpu.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/signal_handler.h"


using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");
DEFINE_int32(comm_threads, 1,
    "Optional; multinode mode,"
    " The number of threads used by communication code.");
DEFINE_string(par, "",
        "Optional; parallelization strategy, e.g., MPINCCLSync");
DEFINE_bool(cube, true, "for MPIGossipParamsGPU, use hypercube");
DEFINE_bool(avgdata, true, "for MPIGossipParamsGPU, average the params also");
DEFINE_bool(alldata, true, "for MPIGossipParamsGPU, average the params also");
DEFINE_bool(rotate, true, "for MPIGossipParamsGPU, rotate comm partner");
DEFINE_bool(batchwise, true, "for MPIGossipParamsGPU, update pair each batch (true) or layer (false)");
DEFINE_string(mpi, "MPI_THREAD_SINGLE", "MPI threading level");
DEFINE_bool(step_mpi, false, "divide stepsize by MPI comm size");
DEFINE_bool(max_iter_mpi, false, "divide max_iter by MPI comm size");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

// Hack to convert macro to string
#define STRINGIZE(m) #m
#define STRINGIZE2(m) STRINGIZE(m)

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
  if (FLAGS_par == "") {
#ifndef CPU_ONLY
    caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif
    // Set mode and device id[s]
    if (gpus.size() == 0) {
      LOG(INFO) << "Use CPU.";
      Caffe::set_mode(Caffe::CPU);
    } else {
      ostringstream s;
      for (int i = 0; i < gpus.size(); ++i) {
        s << (i ? ", " : "") << gpus[i];
      }
      LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
      cudaDeviceProp device_prop;
      for (int i = 0; i < gpus.size(); ++i) {
        cudaGetDeviceProperties(&device_prop, gpus[i]);
        LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
      }
#endif
      solver_param.set_device_id(gpus[0]);
      Caffe::SetDevice(gpus[0]);
      Caffe::set_mode(Caffe::GPU);
      Caffe::set_solver_count(gpus.size());
    }
  }
  else {
    int count = 0;
    int node_rank = caffe::mpi::node_rank();
    int node_size = caffe::mpi::node_size();
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (node_size <= count) {
      if (count != node_size) {
        LOG(INFO) << "MPI node size < cudaGetDeviceCount";
      }
    }
    else {
      throw std::runtime_error("too many MPI ranks per node");
    }
    caffe::GPUMemory::Scope gpu_memory_scope(vector<int>(1, node_rank));
    solver_param.set_device_id(node_rank);
    Caffe::SetDevice(node_rank);
    Caffe::set_mode(Caffe::GPU);
    if (FLAGS_step_mpi) {
      if (solver_param.has_stepsize()) {
        int old = solver_param.stepsize();
        int div = caffe::mpi::comm_size();
        solver_param.set_stepsize(old/div);
        CHECK_EQ(solver_param.stepsize(), old/div);
        LOG(INFO) << "stepsize changed: " << old << " / " << div
          << " = " << old/div;
      }
    }
    else {
      if (solver_param.has_stepsize()) {
        int old = solver_param.stepsize();
        LOG(INFO) << "stepsize remained: " << old;
      }
    }
    if (FLAGS_max_iter_mpi) {
      if (solver_param.has_max_iter()) {
        int old = solver_param.max_iter();
        int div = caffe::mpi::comm_size();
        solver_param.set_max_iter(old/div);
        CHECK_EQ(solver_param.max_iter(), old/div);
        LOG(INFO) << "max_iter changed: " << old << " / " << div
          << " = " << old/div;
      }
      if (solver_param.has_test_interval()) {
        int old = solver_param.test_interval();
        int div = caffe::mpi::comm_size();
        solver_param.set_test_interval(old/div);
        CHECK_EQ(solver_param.test_interval(), old/div);
        LOG(INFO) << "test_interval changed: " << old << " / " << div
          << " = " << old/div;
      }
    }
    else {
      if (solver_param.has_max_iter()) {
        int old = solver_param.max_iter();
        LOG(INFO) << "max_iter remained: " << old;
      }
      if (solver_param.has_test_interval()) {
        int old = solver_param.test_interval();
        LOG(INFO) << "test_interval remained: " << old;
      }
    }
  }

  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (FLAGS_par == "") {
    if (gpus.size() > 1) {
      caffe::P2PSync<float> sync(solver, 0, gpus.size(), solver->param());
      sync.Run(gpus);
    } else {
      LOG(INFO) << "Starting Optimization";
      solver->Solve();
    }
  }
  else {
    if (FLAGS_par == "MPISyncGPU") {
      caffe::MPISyncGPU<float> sync(solver, solver->param());
      sync.Run();
    }
    else if (FLAGS_par == "MPINoCommGPU") {
      caffe::MPINoCommGPU<float> sync(solver, solver->param());
      sync.Run();
    }
    else if (FLAGS_par == "MPIAsyncParamsGPU") {
      caffe::MPIAsyncParamsGPU<float> sync(solver, solver->param(),
          FLAGS_comm_threads);
      sync.Run();
    }
    else if (FLAGS_par == "MPISyncParamsGPU") {
      caffe::MPISyncParamsGPU<float> sync(solver, solver->param());
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU") {
      caffe::MPIGossipParamsGPU<float> sync(solver, solver->param(),
          FLAGS_comm_threads,
          FLAGS_cube,
          FLAGS_avgdata,
          FLAGS_alldata,
          FLAGS_rotate,
          FLAGS_batchwise);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU2") {
      caffe::MPIGossipParamsGPU2<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU3") {
      caffe::MPIGossipParamsGPU3<float> sync(solver, solver->param(),
          FLAGS_comm_threads,
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU4") {
      caffe::MPIGossipParamsGPU4<float> sync(solver, solver->param(),
          FLAGS_comm_threads,
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU5") {
      caffe::MPIGossipParamsGPU5<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU6") {
      caffe::MPIGossipParamsGPU6<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU7") {
      caffe::MPIGossipParamsGPU7<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU8") {
      caffe::MPIGossipParamsGPU8<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPIGossipParamsGPU9") {
      caffe::MPIGossipParamsGPU9<float> sync(solver, solver->param(),
          FLAGS_cube,
          FLAGS_rotate);
      sync.Run();
    }
    else if (FLAGS_par == "MPINCCLSync") {
      caffe::MPINCCLSync<float> sync(solver, solver->param());
      sync.Run();
    }
    else if (FLAGS_par == "MPINCCLAsync") {
      caffe::MPINCCLAsync<float> sync(solver, solver->param());
      sync.Run();
    }
    else {
      LOG(ERROR) << "unrecognized FLAGS_par";
    }
  }
  LOG(INFO) << "Optimization Done.";

  // solver.reset();
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Read flags for list of GPUs
  vector<int> gpus;
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
#ifndef CPU_ONLY
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif

  // Set mode and device id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size() + FLAGS_solver.size(), 0) << "Need a model definition to time.";
  vector<int> gpus;
#ifndef CPU_ONLY
  // Read flags for list of GPUs
  get_gpus(&gpus);
  while (gpus.size() > 1) {
    // Only use one GPU
    LOG(INFO) << "Not using GPU #" << gpus.back() << " for single-GPU function";
    gpus.pop_back();
  }
  caffe::GPUMemory::Scope gpu_memory_scope(gpus);
#endif

  caffe::SolverParameter solver_param;
  if (FLAGS_solver.size() > 0) {
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
  }

  // Set mode and device_id
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU " << gpus[0] << ": " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    solver_param.set_device_id(gpus[0]);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Instantiate the caffe net.
  shared_ptr<Net<float> > caffe_net;
  shared_ptr<caffe::Solver<float> > solver;

  if (FLAGS_solver.size() > 0) {
    solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe_net = solver->net();
  }
  else {
    caffe_net.reset(new Net<float>(FLAGS_model, caffe::TRAIN));
  }

  // Do a number of clean forward and backward pass,
  // so that memory allocation are done,
  // and future iterations will be more stable.
  Timer init_timer;
  Timer forward_timer;
  Timer backward_timer;
  Timer apply_timer;
  double forward_time = 0.0;
  double backward_time = 0.0;
  double apply_time = 0.0;
  const int kInitIterations = 5;
  LOG(INFO) << "Initialization for " << kInitIterations << " iterations.";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  LOG(INFO) << "Performing initial Forward/Backward";
  const vector<shared_ptr<Layer<float> > >& layers = caffe_net->layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net->bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net->top_vecs();
  const vector<Blob<float>*>& params = caffe_net->learnable_params();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net->bottom_need_backward();
  float initial_loss = 0.F;
  init_timer.Start();
  for (int j = 0; j < kInitIterations; ++j) {
    for (int i = 0; i < layers.size(); ++i) {
      initial_loss += layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
    for (int i = layers.size() - 1; i >= 0; --i) {
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
    }
  }
  double init_time = init_timer.MilliSeconds();
  LOG(INFO) << "Initial Forward/Backward complete, loss: " << initial_loss;
  LOG(INFO) << "Average Initialization Forward/Backward pass: " << init_time /
      kInitIterations << " ms.";

  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  std::vector<double> apply_time_per_param(params.size(), 0.0);
  forward_time = 0.0;
  backward_time = 0.0;
  apply_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    if (FLAGS_solver.size() > 0) {
      apply_timer.Start();
      float rate = solver->GetLearningRate();
      solver->ClipGradients();
      for (int i = params.size() - 1; i >= 0; --i) {
        timer.Start();
        solver->Normalize(i);
        solver->Regularize(i);
        solver->ComputeUpdateValue(i, rate);
        apply_time_per_param[i] += timer.MicroSeconds();
      }
      caffe_net->Update();
      apply_time += apply_timer.MicroSeconds();
    }
    if (FLAGS_solver.size() > 0) {
      LOG(INFO) << "Iteration: " << j + 1 << " forward-backward-apply time: "
        << iter_timer.MilliSeconds() << " ms.";
    }
    else {
      LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
        << iter_timer.MilliSeconds() << " ms.";
    }
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  if (FLAGS_solver.size() > 0) {
    for (int i = params.size() - 1; i >= 0; --i) {
      LOG(INFO) << std::setfill(' ') << std::setw(10) << i <<
        "\tapply: " << apply_time_per_param[i] / 1000 /
        FLAGS_iterations << " ms.";
    }
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  if (FLAGS_solver.size() > 0) {
    LOG(INFO) << "Average Apply pass: " << apply_time / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << "Average Forward-Backward-Apply: " << total_timer.MilliSeconds() /
      FLAGS_iterations << " ms.";
  }
  else {
    LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
      FLAGS_iterations << " ms.";
  }
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(STRINGIZE2(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  if (argc == 2) {
    if (FLAGS_par != "") {
      caffe::mpi::init(&argc, &argv, FLAGS_mpi);
      LOG(INFO) << "MPI rank " << caffe::mpi::comm_rank();
      // only log info from master
      if (caffe::mpi::comm_rank() > 0) {
        FLAGS_minloglevel = 2;
      }
      LOG(INFO) << "MPI is initialized, disabling logging from other ranks";
    }
    else {
      /* init mpi anyway so we can use pnetcdf reader */
      caffe::mpi::init(&argc, &argv, FLAGS_mpi);
    }
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
