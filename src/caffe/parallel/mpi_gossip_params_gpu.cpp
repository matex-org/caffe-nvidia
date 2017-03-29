#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsGPU<Dtype>::Reducer : public InternalThread {
  public:
    MPIGossipParamsGPU<Dtype> *sync_;
    int tid_;
    Timer timer_queue_;
    double time_in_queue_;
    Timer timer_comm_;
    double time_in_comm_;
    vector<double> time_per_param_;
    stats_t stats_queue_;
    stats_t stats_comm_;

    Reducer(MPIGossipParamsGPU<Dtype> *sync, int tid)
        : sync_(sync), tid_(tid),
        timer_queue_(), time_in_queue_(0.0),
        timer_comm_(), time_in_comm_(0.0),
        time_per_param_(),
        stats_queue_(),
        stats_comm_()
    { 
      time_per_param_.resize(sync_->params_.size());
      stats_clear(&stats_queue_);
      stats_clear(&stats_comm_);
    }

    void InternalThreadEntry() {
      try {
        while (!must_stop()) {
          if (!sync_->batchwise_) {
            sync_->next();
          }
          timer_queue_.Start();
          int param_id = sync_->param_solo_.pop("solo param not yet ready");
          time_in_queue_ += timer_queue_.MilliSeconds();
          timer_comm_.Start();
          if (-1 == param_id) {
            MPI_Comm comm = sync_->comms_[0];
            vector<MPI_Request> requests(2);
            caffe::mpi::irecv(requests[0], sync_->data_all_,
                sync_->size_, sync_->recv_pair_, 3333, comm);
            caffe::mpi::isend(requests[1], sync_->data_,
                sync_->size_, sync_->send_pair_, 3333, comm);
            caffe::mpi::waitall(requests);
            time_in_comm_ += timer_comm_.MilliSeconds();
          }
          else {
            Blob<Dtype> *blob = sync_->params_[param_id];
            MPI_Comm comm = sync_->comms_[param_id];
            Dtype *recvdiff = sync_->param_diffs_[param_id];
            Dtype *recvdata = sync_->param_datas_[param_id];
#ifdef USE_MPI
            // exchange data
#if 0
            caffe::mpi::sendrecv(
                (const Dtype*)blob->gpu_diff(), blob->count(), sync_->send_pair_, 1234,
                recvdiff, blob->count(), sync_->recv_pair_, 1234, comm);
            if (sync_->avgdata_ && !sync_->alldata_) {
              caffe::mpi::sendrecv(
                  (const Dtype*)blob->gpu_data(), blob->count(), sync_->send_pair_, 1234,
                  recvdata, blob->count(), sync_->recv_pair_, 1234, comm);
            }
#endif
#if 1
            if (sync_->avgdata_ && !sync_->alldata_) {
              vector<MPI_Request> requests(4);
              caffe::mpi::irecv(requests[0], recvdiff,
                  blob->count(), sync_->recv_pair_, 2222, comm);
              caffe::mpi::irecv(requests[1], recvdata,
                  blob->count(), sync_->recv_pair_, 3333, comm);
              caffe::mpi::isend(requests[2], (const Dtype*)blob->gpu_diff(),
                  blob->count(), sync_->send_pair_, 2222, comm);
              caffe::mpi::isend(requests[3], (const Dtype*)blob->gpu_data(),
                  blob->count(), sync_->send_pair_, 3333, comm);
              caffe::mpi::waitall(requests);
            }
            else {
              vector<MPI_Request> requests(2);
              caffe::mpi::irecv(requests[0], recvdiff,
                  blob->count(), sync_->recv_pair_, 2222, comm);
              caffe::mpi::isend(requests[1], (const Dtype*)blob->gpu_diff(),
                  blob->count(), sync_->send_pair_, 2222, comm);
              caffe::mpi::waitall(requests);
            }
#endif
            time_per_param_[param_id] += timer_comm_.MilliSeconds();
            time_in_comm_ += timer_comm_.MilliSeconds();
            timer_queue_.Start();
            sync_->param_all_[param_id]->push(tid_);
            time_in_queue_ += timer_queue_.MilliSeconds();
          }
          // postpone average local data and diff into secondary buffers
#else
          NO_MPI;
#endif
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
void MPIGossipParamsGPU<Dtype>::next() {
  if (cube_) {
    if (rotate_) {
      next_cube_rotate();
    }
    else {
      next_cube();
    }
  }
  else {
    if (rotate_) {
      next_diffuse_rotate();
    }
    else {
      next_diffuse();
    }
  }
  //LOG(INFO) << "rank " << comm_rank_orig_ << " rot rank " << comm_rank_ << " send " << send_pair_ << " recv " << recv_pair_;
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::next_cube_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    comm_rank_ = (comm_rank_+1) % comm_size_;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = MPI_ANY_SOURCE;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::next_diffuse() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::next_diffuse_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    comm_rank_ = (comm_rank_+1) % comm_size_;
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  recv_pair_ = MPI_ANY_SOURCE;
  ++hci_;
}

template<typename Dtype>
MPIGossipParamsGPU<Dtype>::MPIGossipParamsGPU(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param,
    int comm_threads,
    bool cube,
    bool avgdata,
    bool alldata,
    bool rotate,
    bool batchwise)
  : GPUParams<Dtype>(root_solver, param.device_id()),
    comm_rank_(),
    comm_size_(),
    logp_(0),
    hci_(0),
    send_pair_(0),
    recv_pair_(0),
    solver_(),
    params_(root_solver->net()->learnable_params()),
    param_solo_(),
    param_all_(),
    comms_(),
    reducers(),
    diff_all_(),
    data_all_(),
    param_diffs_(),
    param_datas_(),
    cube_(cube),
    avgdata_(avgdata),
    alldata_(alldata),
    rotate_(rotate),
    batchwise_(batchwise)
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

  comm_rank_ = caffe::mpi::comm_rank(comms_[0]);
  comm_rank_orig_ = comm_rank_;
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  //diff_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&diff_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), diff_all_);
  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);

  //data_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&data_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), data_all_);
  param_datas_.resize(params_.size());
  get_pointers(params_, data_all_, param_datas_);

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
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIGossipParamsGPU<Dtype>::~MPIGossipParamsGPU() {
  for (int i = 0; i < reducers.size(); ++i) {
    reducers[i]->StopInternalThread();
    delete reducers[i];
  }
  delete [] diff_all_;
  delete [] data_all_;
  for (int i = 0; i < params_.size(); ++i) {
    delete param_all_[i];
  }
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::on_begin() {
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
  if (batchwise_) {
    next();
  }
  if (alldata_ && avgdata_) {
    // tell comm thread to send all data
    param_solo_.push(-1);
  }
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
  if (avgdata_ && alldata_) {
    // average pairwise exchange
    caffe_gpu_axpby(size_, Dtype(0.5), data_, Dtype(0.5), data_all_);
    // swap data pointer with reduction pointer
    Dtype *swap;
    swap = data_;
    data_ = data_all_;
    data_all_ = swap;
    apply_buffers(params_, data_, size_, replace_gpu);
  }
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
  param_solo_.push(param_id);
}

template<typename Dtype>
int MPIGossipParamsGPU<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  int who_did_the_work = param_all_[param_id]->pop("waiting in apply");
  Blob<Dtype> *blob = params_[param_id];

  // average pairwise exhange
  caffe_gpu_axpby(blob->count(), Dtype(0.5), blob->gpu_diff(), Dtype(0.5), param_diffs_[param_id]);
  if (avgdata_ && !alldata_) {
    caffe_gpu_axpby(blob->count(), Dtype(0.5), blob->gpu_data(), Dtype(0.5), param_datas_[param_id]);
  }

  // swap diff and data pointers with reduction pointers
  Dtype *swap;
  swap = blob->mutable_gpu_diff();
  blob->diff()->set_gpu_data(param_diffs_[param_id]);
  param_diffs_[param_id] = swap;
  if (avgdata_ && !alldata_) {
    swap = blob->mutable_gpu_data();
    blob->data()->set_gpu_data(param_datas_[param_id]);
    param_datas_[param_id] = swap;
  }
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsGPU);

}  // namespace caffe

