#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_async_params_gpu.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

template<typename Dtype>
class MPIAsyncParamsGPU<Dtype>::Reducer : public InternalThread {
  public:
    MPIAsyncParamsGPU<Dtype> *sync_;
    int tid_;
    Timer timer_queue_;
    double time_in_queue_;
    Timer timer_comm_;
    double time_in_comm_;
    vector<double> time_per_param_;
    stats_t stats_queue_;
    stats_t stats_comm_;

    Reducer(MPIAsyncParamsGPU<Dtype> *sync, int tid)
        : sync_(sync), tid_(tid),
        timer_queue_(), time_in_queue_(0.0),
        timer_comm_(), time_in_comm_(0.0),
        time_per_param_(),
        stats_queue_(),
        stats_comm_()
  { 
    time_per_param_.resize(sync->params_.size());
    stats_clear(&stats_queue_);
    stats_clear(&stats_comm_);
  }

    void InternalThreadEntry() {
      try {
        while (!must_stop()) {
          timer_queue_.Start();
          int param_id = sync_->param_solo_.pop("solo param not yet ready");
          time_in_queue_ += timer_queue_.MilliSeconds();
          Blob<Dtype> *blob = sync_->params_[param_id];
          MPI_Comm comm = sync_->comms_[param_id];
          Dtype *sum = sync_->param_diffs_[param_id];
#ifdef USE_MPI
#if 0
          // sum gradients
          if (sync_->params_[param_id]->prv_diff()
              && (sync_->params_[param_id]->prv_diff_count()
                == sync_->params_[param_id]->count())) {
            timer_comm_.Start();
            caffe::mpi::allreduce_copy((const Dtype*)blob->prv_diff(),
                sum, blob->count(), MPI_SUM, comm);
            time_in_comm_ += timer_comm_.MilliSeconds();
          }
          else {
#endif
#if 1
            timer_comm_.Start();
            caffe::mpi::allreduce_copy((const Dtype*)blob->gpu_diff(),
                sum, blob->count(), MPI_SUM, comm);
            time_per_param_[param_id] += timer_comm_.MilliSeconds();
            time_in_comm_ += timer_comm_.MilliSeconds();
#else
            timer_comm_.Start();
            caffe::mpi::allreduce(blob->mutable_gpu_diff(),
                blob->count(), MPI_SUM, comm);
            time_in_comm_ += timer_comm_.MilliSeconds();
#endif
#if 0
          }
#endif
          //caffe_scal(blob->count(), Dtype(1.0 / sync_->comm_size_), sum);
#else       
          NO_MPI;
#endif        
          timer_queue_.Start();
          sync_->param_all_[param_id]->push(tid_);
          time_in_queue_ += timer_queue_.MilliSeconds();
        }
      } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
      }
    }
};

template<typename Dtype>
static void get_pointers(const vector<Blob<Dtype>*>& blobs,
    Dtype *ptr, vector<Dtype*>& ptrs)
{
  for (int i = 0; i < blobs.size(); ++i) {
    ptrs[i] = ptr;
    ptr += blobs[i]->count();
  }
}

template<typename Dtype>
MPIAsyncParamsGPU<Dtype>::MPIAsyncParamsGPU(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param,
    int comm_threads)
  : GPUParams<Dtype>(root_solver, param.device_id()),
    comm_size_(),
    solver_(),
    params_(root_solver->net()->learnable_params()),
    param_solo_(),
    param_all_(),
    comms_(),
    reducers(),
    diff_all_(),
    param_diffs_()
{
#ifdef USE_MPI
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  // one MPI_Comm per parameter
  comms_.resize(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    comms_[i] = caffe::mpi::comm_dup();
  }
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  node_rank = caffe::mpi::node_rank(comms_[0]);
  node_size = caffe::mpi::node_size(comms_[0]);

  if (node_size <= count) {
    if (count != node_size) {
      LOG(INFO) << "MPINCCLAsync MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("MPINCCLAsync too many MPI ranks per node");
  }

  if (0 == node_rank ) {
    cudaDeviceProp device_prop;
    for (int i = 0; i < count; ++i) {
      cudaGetDeviceProperties(&device_prop, i);
      LOG(INFO) << "GPU " << i << ": " << device_prop.name;
    }
  }

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
  solver_->set_use_mpi(true);

  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  //diff_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&diff_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), diff_all_);
  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);

  // create queue, one per param
  param_all_.resize(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    param_all_[i] = new BlockingQueue<int>;
  }
  
  // Start the gradient allreduce threads
  reducers.resize(comm_threads);
  for (int i = 0; i < comm_threads; ++i) {
    reducers[i] = new Reducer(this, i);
    reducers[i]->StartInternalThread();
  }

  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIAsyncParamsGPU<Dtype>::~MPIAsyncParamsGPU() {
  for (int i = 0; i < reducers.size(); ++i) {
    reducers[i]->StopInternalThread();
    delete reducers[i];
  }
  delete [] diff_all_;
  for (int i = 0; i < params_.size(); ++i) {
    delete param_all_[i];
  }
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::on_begin() {
  DLOG(INFO) << "on_begin()";
  for (int i=0; i<reducers.size(); ++i) {
    stats_sample_value(&reducers[i]->stats_queue_, reducers[i]->time_in_queue_);
    stats_sample_value(&reducers[i]->stats_comm_, reducers[i]->time_in_comm_);
    //LOG(INFO) << "reducer[" << i << "] time queue " << reducers[i]->time_in_queue_ << " time comm " << reducers[i]->time_in_comm_;
    LOG_EVERY_N(INFO, 20) << "reducer[" << i << "] time queue " << reducers[i]->stats_queue_._mean << " time comm " << reducers[i]->stats_comm_._mean;
#if 0
    if (solver_->iter() > 0) {
      for (int j=params_.size()-1; j >= 0; --j) {
        LOG(INFO) << j << ": " << reducers[i]->time_per_param_[j]/solver_->iter();
      }
    }
#endif
    reducers[i]->time_in_queue_ = 0.0;
    reducers[i]->time_in_comm_ = 0.0;
  }
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
  param_solo_.push(param_id);
}

template<typename Dtype>
int MPIAsyncParamsGPU<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  int who_did_the_work = param_all_[param_id]->pop("waiting in apply");
  Blob<Dtype> *blob = params_[param_id];
  Dtype *swap = blob->mutable_gpu_diff();
  blob->diff()->set_gpu_data(param_diffs_[param_id]);
  param_diffs_[param_id] = swap;
  return param_id;
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIAsyncParamsGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIAsyncParamsGPU);

}  // namespace caffe

