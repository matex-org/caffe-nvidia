#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu9.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

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
void MPIGossipParamsGPU9<Dtype>::next() {
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
void MPIGossipParamsGPU9<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::next_cube_rotate() {
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
void MPIGossipParamsGPU9<Dtype>::next_diffuse() {
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
void MPIGossipParamsGPU9<Dtype>::next_diffuse_rotate() {
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
MPIGossipParamsGPU9<Dtype>::MPIGossipParamsGPU9(
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
    comms_(),
    data_requests_(),
    history_requests_(),
    time_comm_(),
    time_comp_(),
    stats_comm_(),
    stats_comp_(),
    cpu_send_data_(),
    cpu_recv_data_(),
    cpu_send_history_(),
    cpu_recv_history_(),
    data_all_(),
    history_(),
    history_all_(),
    history_size_(),
    data_send_copied_(),
    data_recv_copied_(),
    history_send_copied_(),
    history_recv_copied_(),
    cube_(cube),
    rotate_(rotate),
    first_time_(true),
    data_state_(UNINITIALIZED),
    history_state_(UNINITIALIZED)
{
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  stats_clear(&stats_comm_);
  stats_clear(&stats_comp_);

  CUDA_CHECK(cudaGetDeviceCount(&count));

  // one MPI_Comm per rank
  Timer timer_comm_create_;
  timer_comm_create_.Start();
  comm_size_ = caffe::mpi::comm_size();
  node_rank = caffe::mpi::node_rank();
  node_size = caffe::mpi::node_size();
  comms_.resize(comm_size_);
  comms_[0] = caffe::mpi::comm_dup();
  comm_rank_orig_ = caffe::mpi::comm_rank(comms_[0]);
  vector<int> ranks(comm_size_);
  for (int i = 0; i < comm_size_; ++i) {
    ranks[i] = i;
  }
  for (int i = 1; i < comm_size_; ++i) {
    if (0 == comm_rank_orig_) {
      std::random_shuffle(ranks.begin(), ranks.end());
    }
    caffe::mpi::bcast(ranks, 0, comms_[0]);
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
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  cpu_send_data_ = new Dtype[size_];
  cpu_recv_data_ = new Dtype[size_];

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

  cpu_send_history_ = new Dtype[history_size_];
  cpu_recv_history_ = new Dtype[history_size_];

  GPUMemory::allocate(reinterpret_cast<void **>(&history_),
      history_size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(history_size_, Dtype(0), history_);

  GPUMemory::allocate(reinterpret_cast<void **>(&history_all_),
      history_size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(history_size_, Dtype(0), history_all_);

  apply_buffers(sgdsolver_->history(), history_, history_size_, replace_gpu);

  CUDA_CHECK(cudaEventCreateWithFlags(&data_send_copied_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&data_recv_copied_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&history_send_copied_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&history_recv_copied_, cudaEventDisableTiming));

  LOG(INFO) << "buffer size_ " << size_*sizeof(Dtype);
  LOG(INFO) << "buffer history_size_ " << history_size_*sizeof(Dtype);
}

template<typename Dtype>
MPIGossipParamsGPU9<Dtype>::~MPIGossipParamsGPU9() {
  GPUMemory::deallocate(data_all_, buffer_device_, stream_);
  GPUMemory::deallocate(history_, buffer_device_, stream_);
  GPUMemory::deallocate(history_all_, buffer_device_, stream_);
  delete [] cpu_send_data_;
  delete [] cpu_recv_data_;
  delete [] cpu_send_history_;
  delete [] cpu_recv_history_;
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::on_begin() {
  DLOG(INFO) << "on_begin()";
  CPUTimer timer;

  // wait for comm to finish and update buffers 
  if (first_time_) {
    LOG(INFO) << "first iteration doesn't wait for comm";
    first_time_ = false;
  }
  else {
    timer.Start();
    while (FINISHED != data_state_ && FINISHED != history_state_) {
      make_progress();
    }
    timer.Stop();
    time_comm_ += timer.MilliSeconds();
    timer.Start();
    caffe_gpu_axpby(size_, Dtype(0.5), data_all_, Dtype(0.5), data_, stream_);
    caffe_gpu_axpby(history_size_, Dtype(0.5), history_all_, Dtype(0.5), history_, stream_);
    timer.Stop();
    time_comp_ += timer.MilliSeconds();
    stats_sample_value(&stats_comm_, time_comm_);
    stats_sample_value(&stats_comp_, time_comp_);
    LOG_EVERY_N(INFO, 20) << "time comm sample " << time_comm_;
    LOG_EVERY_N(INFO, 20) << "time comp sample " << time_comp_;
    LOG_EVERY_N(INFO, 20) << "time comm " << stats_comm_._mean
      << " += " << stats_stddev(&stats_comm_)
      << " min " << stats_comm_._min
      << " max " << stats_comm_._max;
    LOG_EVERY_N(INFO, 20) << "time comp " << stats_comp_._mean
      << " += " << stats_stddev(&stats_comp_)
      << " min " << stats_comp_._min
      << " max " << stats_comp_._max;
  }

  time_comm_ = 0;
  time_comp_ = 0;

  // select next exchange partners
  next();

  data_state_ = UNINITIALIZED;
  history_state_ = UNINITIALIZED;

  // begin exchange of samples, data, and history
  make_progress();
}

template<typename Dtype>
bool MPIGossipParamsGPU9<Dtype>::make_progress() {
  CPUTimer timer;
  bool state_changed = false;

  if (UNINITIALIZED == data_state_) {
    DLOG(INFO) << "UNINITIALIZED == data_state_";
    // copy device to host
    timer.Start();
    CUDA_CHECK(cudaMemcpyAsync(cpu_send_data_, data_, size_*sizeof(Dtype), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(data_send_copied_, stream_));
    timer.Stop();
    time_comp_ += timer.MilliSeconds();

    // update our state
    data_state_ = TEST_DEVICE_TO_HOST;
    state_changed = true;

    // prepost recv early
    timer.Start();
    MPI_Comm comm = comms_[mci_];
    data_requests_.assign(2, MPI_REQUEST_NULL);
    caffe::mpi::irecv(data_requests_[0], cpu_recv_data_, size_, recv_pair_, 1234, comm);
    timer.Stop();
    time_comp_ += timer.MilliSeconds();
  }
  else if (TEST_DEVICE_TO_HOST == data_state_) {
    DLOG(INFO) << "TEST_DEVICE_TO_HOST == data_state_";
    // check cuda event
    cudaError_t ret;
    ret = cudaEventQuery(data_send_copied_);
    if (cudaSuccess == ret) {
      // buffer ready, send it
      timer.Start();
      MPI_Comm comm = comms_[mci_];
      caffe::mpi::isend(data_requests_[1], cpu_send_data_, size_, send_pair_, 1234, comm);
      timer.Stop();
      time_comm_ += timer.MilliSeconds();
      // update our state
      data_state_ = TEST_MPI;
      state_changed = true;
    }
    else if (cudaErrorNotReady == ret) {
      // this is okay
    }
    else {
      CUDA_CHECK(ret);
    }
  }
  else if (TEST_MPI == data_state_) {
    DLOG(INFO) << "TEST_MPI == data_state_";
    // check MPI status
    bool mpi_done = false;
    timer.Start();
    mpi_done = caffe::mpi::testall(data_requests_);
    timer.Stop();
    time_comm_ += timer.MilliSeconds();
    if (mpi_done) {
      // recv finished, copy buffer back to GPU
      timer.Start();
      CUDA_CHECK(cudaMemcpyAsync(data_all_, cpu_recv_data_, size_*sizeof(Dtype), cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaEventRecord(data_recv_copied_, stream_));
      timer.Stop();
      time_comp_ += timer.MilliSeconds();
      // update our state
      data_state_ = TEST_HOST_TO_DEVICE;
      state_changed = true;
    }
  }
  else if (TEST_HOST_TO_DEVICE == data_state_) {
    DLOG(INFO) << "TEST_HOST_TO_DEVICE == data_state_";
    // check cuda event
    cudaError_t ret;
    ret = cudaEventQuery(data_recv_copied_);
    if (cudaSuccess == ret) {
      data_state_ = FINISHED;
      state_changed = true;
    }
    else if (cudaErrorNotReady == ret) {
      // this is okay
    }
    else {
      CUDA_CHECK(ret);
    }
  }
  else if (FINISHED == data_state_) {
    DLOG(INFO) << "FINISHED == data_state_";
    // do nothing
  }

  if (UNINITIALIZED == history_state_) {
    DLOG(INFO) << "UNINITIALIZED == history_state_";
    // copy device to host
    timer.Start();
    CUDA_CHECK(cudaMemcpyAsync(cpu_send_history_, history_, history_size_*sizeof(Dtype), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(history_send_copied_, stream_));
    timer.Stop();
    time_comp_ += timer.MilliSeconds();

    // update our state
    history_state_ = TEST_DEVICE_TO_HOST;
    state_changed = true;

    // prepost recv early
    timer.Start();
    MPI_Comm comm = comms_[mci_];
    history_requests_.assign(2, MPI_REQUEST_NULL);
    caffe::mpi::irecv(history_requests_[0], cpu_recv_history_, history_size_, recv_pair_, 1234, comm);
    timer.Stop();
    time_comp_ += timer.MilliSeconds();
  }
  else if (TEST_DEVICE_TO_HOST == history_state_) {
    DLOG(INFO) << "TEST_DEVICE_TO_HOST == history_state_";
    // check cuda event
    cudaError_t ret;
    ret = cudaEventQuery(history_send_copied_);
    if (cudaSuccess == ret) {
      // buffer ready, send it
      timer.Start();
      MPI_Comm comm = comms_[mci_];
      caffe::mpi::isend(history_requests_[1], cpu_send_history_, history_size_, send_pair_, 1234, comm);
      timer.Stop();
      time_comm_ += timer.MilliSeconds();
      // update our state
      history_state_ = TEST_MPI;
      state_changed = true;
    }
    else if (cudaErrorNotReady == ret) {
      // this is okay
    }
    else {
      CUDA_CHECK(ret);
    }
  }
  else if (TEST_MPI == history_state_) {
    DLOG(INFO) << "TEST_MPI == history_state_";
    // check MPI status
    bool mpi_done = false;
    timer.Start();
    mpi_done = caffe::mpi::testall(history_requests_);
    timer.Stop();
    time_comm_ += timer.MilliSeconds();
    if (mpi_done) {
      // recv finished, copy buffer back to GPU
      timer.Start();
      CUDA_CHECK(cudaMemcpyAsync(history_all_, cpu_recv_history_, history_size_*sizeof(Dtype), cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaEventRecord(history_recv_copied_, stream_));
      timer.Stop();
      time_comp_ += timer.MilliSeconds();
      // update our state
      history_state_ = TEST_HOST_TO_DEVICE;
      state_changed = true;
    }
  }
  else if (TEST_HOST_TO_DEVICE == history_state_) {
    DLOG(INFO) << "TEST_HOST_TO_DEVICE == history_state_";
    // check cuda event
    cudaError_t ret;
    ret = cudaEventQuery(history_recv_copied_);
    if (cudaSuccess == ret) {
      history_state_ = FINISHED;
      state_changed = true;
    }
    else if (cudaErrorNotReady == ret) {
      // this is okay
    }
    else {
      CUDA_CHECK(ret);
    }
  }
  else if (FINISHED == history_state_) {
    DLOG(INFO) << "FINISHED == history_state_";
    // do nothing
  }

  //solver_->DataShuffleTest();

  return false;
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::on_forward(int param_id) {
  DLOG(INFO) << "on_forward(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::after_forward() {
  DLOG(INFO) << "after_forward()";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
  make_progress();
}

template<typename Dtype>
int MPIGossipParamsGPU9<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  make_progress();
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::on_update() {
  DLOG(INFO) << "on_update()";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsGPU9<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsGPU9);

}  // namespace caffe

