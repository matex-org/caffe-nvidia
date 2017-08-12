#ifndef CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_GPU4_HPP_
#define CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_GPU4_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsGPU4 : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIGossipParamsGPU4(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param,
          int comm_threads, bool cube, bool rotate);
  virtual ~MPIGossipParamsGPU4();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  friend class Reducer;

 protected:
  class Reducer;

  void on_start();
  void on_begin();
  void after_forward();
  void allreduce();
  void allreduce(int param_id);
  int on_apply(int param_id);
  void on_update();

  void next();
  void next_cube();
  void next_cube_rotate();
  void next_diffuse();
  void next_diffuse_rotate();

  int comm_rank_orig_;
  int comm_rank_;
  int comm_size_;
  int logp_;
  int hci_;
  int mci_;
  int send_pair_;
  int recv_pair_;
  shared_ptr<Solver<Dtype> > solver_;
  shared_ptr<SGDSolver<Dtype> > sgdsolver_;
  shared_ptr<AdamSolver<Dtype> > adamsolver_;
  const vector<Blob<Dtype>*>& params_;
  BlockingQueue<int> param_solo_;
  BlockingQueue<int> param_all_;
#ifdef USE_MPI
  vector<MPI_Comm> comms_;
#endif
  vector<Reducer*> reducers;
  Dtype *data_all_;
  Dtype *history_;
  Dtype *history_all_;
  size_t history_size_;
  int comm_threads_;
  bool cube_;
  bool rotate_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  using GPUParams<Dtype>::buffer_device_;
  using GPUParams<Dtype>::stream_;
};

}  // namespace caffe

#endif

