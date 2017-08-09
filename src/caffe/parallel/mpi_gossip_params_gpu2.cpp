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
#include "caffe/parallel/mpi_gossip_params_gpu2.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

#define EXCHANGE_HISTORY_ON_UPDATE 1

namespace caffe {

static CPUTimer timer_;
static stats_t stats_comm_;

template<typename Dtype>
static void apply_buffers(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next() {
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
void MPIGossipParamsGPU2<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_cube_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_diffuse() {
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
void MPIGossipParamsGPU2<Dtype>::next_diffuse_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
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
MPIGossipParamsGPU2<Dtype>::MPIGossipParamsGPU2(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param,
    bool cube,
    bool rotate)
  : GPUParams<Dtype>(root_solver, param.device_id()),
    comm_rank_(),
    comm_size_(),
    logp_(0),
    hci_(0),
    mci_(0),
    send_pair_(0),
    recv_pair_(0),
    solver_(),
    sgdsolver_(),
    adamsolver_(),
    params_(root_solver->net()->learnable_params()),
#ifdef USE_MPI
    comms_(),
#endif
    data_all_(),
    history_(),
    history_all_(),
    history_size_(),
    cube_(cube),
    rotate_(rotate)
{
#ifdef USE_MPI
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  stats_clear(&stats_comm_);

  CUDA_CHECK(cudaGetDeviceCount(&count));

  // one MPI_Comm per rank
  Timer timer_comm_create_;
  timer_comm_create_.Start();
  comm_size_ = caffe::mpi::comm_size();
  node_rank = caffe::mpi::node_rank();
  node_size = caffe::mpi::node_size();
  comms_.resize(comm_size_);
  comms_[0] = caffe::mpi::comm_dup();
  vector<int> ranks(comm_size_);
  for (int i = 0; i < comm_size_; ++i) {
    ranks[i] = i;
  }
  for (int i = 1; i < comm_size_; ++i) {
    for (int j = 0; j < comm_size_; ++j) {
      ranks[j] = (ranks[j]+1)%comm_size_;
    }
    comms_[i] = caffe::mpi::comm_create(ranks);
    LOG(INFO) << "my rank " << caffe::mpi::comm_rank(comms_[i]);
  }
  LOG(INFO) << "comm creation time " << timer_comm_create_.MilliSeconds();

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

  sgdsolver_ = boost::dynamic_pointer_cast<SGDSolver<Dtype> >(root_solver);
  if (NULL == sgdsolver_) {
      LOG(FATAL) << "dynamic cast of SGDSolver failed";
  }
  adamsolver_ = boost::dynamic_pointer_cast<AdamSolver<Dtype> >(root_solver);
  if (NULL == adamsolver_) {
      LOG(INFO) << "dynamic cast of AdamSolver failed";
  }

  comm_rank_ = caffe::mpi::comm_rank(comms_[0]);
  comm_rank_orig_ = comm_rank_;
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  //data_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&data_all_),
      size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(size_, Dtype(0), data_all_);

  /* AdamSolver has history that is twice the size */
  if (NULL != adamsolver_) {
    history_size_ = size_ * 2;
  }
  else {
    history_size_ = size_;
  }

  //history_ = new Dtype[history_size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&history_),
      history_size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(history_size_, Dtype(0), history_);

  //history_all_ = new Dtype[history_size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&history_all_),
      history_size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(history_size_, Dtype(0), history_all_);

  apply_buffers(sgdsolver_->history(), history_, history_size_, replace_gpu);

#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIGossipParamsGPU2<Dtype>::~MPIGossipParamsGPU2() {
  GPUMemory::deallocate(data_all_, buffer_device_, stream_);
  GPUMemory::deallocate(history_, buffer_device_, stream_);
  GPUMemory::deallocate(history_all_, buffer_device_, stream_);
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  timer_.Start();
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::on_begin() {
  DLOG(INFO) << "on_begin()";
  stats_sample_value(&stats_comm_, timer_.MilliSeconds());
  LOG_EVERY_N(INFO, 20) << "time comm " << stats_comm_._mean;
  solver_->DataShuffleBegin();
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::after_forward() {
  DLOG(INFO) << "after_forward()";
  solver_->DataShuffleEnd();
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
  timer_.Start();
#ifdef USE_MPI

  // select next exchange partners
  next();

  // exchange data
  {
      MPI_Comm comm = comms_[mci_];
      vector<MPI_Request> requests(2);
      caffe::mpi::irecv(requests[0], data_all_, size_, recv_pair_, 1234, comm);
      caffe::mpi::isend(requests[1], data_,     size_, send_pair_, 1234, comm);
      caffe::mpi::waitall(requests);
  }

#if EXCHANGE_HISTORY_ON_UPDATE
#else
  // exchange history
  {
      MPI_Comm comm = comms_[mci_];
      vector<MPI_Request> requests(2);
      caffe::mpi::irecv(requests[0], history_all_, history_size_, recv_pair_, 2345, comm);
      caffe::mpi::isend(requests[1], history_,     history_size_, send_pair_, 2345, comm);
      caffe::mpi::waitall(requests);
  }
#endif

  {
    // average pairwise exchange
    caffe_gpu_axpby(size_, Dtype(0.5), data_all_, Dtype(0.5), data_);
  }

#if EXCHANGE_HISTORY_ON_UPDATE
#else
  {
    // average pairwise exchange
    caffe_gpu_axpby(history_size_, Dtype(0.5), history_all_, Dtype(0.5), history_);
  }
#endif

  timer_.Stop();
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
}

template<typename Dtype>
int MPIGossipParamsGPU2<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::on_update() {
  DLOG(INFO) << "on_update()";
#if EXCHANGE_HISTORY_ON_UPDATE
  // exchange history
  {
      MPI_Comm comm = comms_[mci_];
      vector<MPI_Request> requests(2);
      caffe::mpi::irecv(requests[0], history_all_, history_size_, recv_pair_, 2345, comm);
      caffe::mpi::isend(requests[1], history_,     history_size_, send_pair_, 2345, comm);
      caffe::mpi::waitall(requests);
  }
  {
    // average pairwise exchange
    caffe_gpu_axpby(history_size_, Dtype(0.5), history_all_, Dtype(0.5), history_);
    // must copy history back into gradient diff also
    // in the case of adam, only the first portion is relevant
    caffe_copy(size_, history_, diff_);
  }
#endif
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsGPU2);

}  // namespace caffe

