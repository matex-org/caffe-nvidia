#ifndef CAFFE_PARALLEL_MPI_SYNC_GPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_GPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"

#ifdef CAFFE_FT
#include <tuple>
#endif

namespace caffe {

// Synchronous data parallelism using Allreduce between remote GPUs.
template<typename Dtype>
class MPISyncGPU : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPISyncGPU(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~MPISyncGPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
#ifdef CAFFE_FT
#ifdef SNAPSHOT_RESTART
  void Run(const string snapshot_file);
#endif
#endif
  void Step(int iters);

  void allreduce(int param_id) {}
  void syncCommStream() {}

 protected:
  void on_start() {}
#ifdef CAFFE_FT  
  std::tuple<int,bool> allreduce();
#else
  void allreduce();
#endif /*CAFFE_FT*/
  void soft_barrier() {}

#ifdef USE_MPI
  MPI_Comm comm_;
  #ifdef CAFFE_FT
  int error_code_;
  #endif
#endif
  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  Timer timer_;

  Dtype* cpu_ptr_; // when we cannot MPI_Allreduce the cuda memory directly
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
