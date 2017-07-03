#ifndef CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_GPU2_HPP_
#define CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_GPU2_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsGPU2 : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIGossipParamsGPU2(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param, bool cube, bool avgdata, bool rotate);
  virtual ~MPIGossipParamsGPU2();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:

  void on_start();
  void on_begin();
  void allreduce();
  void allreduce(int param_id);
  int on_apply(int param_id);

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
  const vector<Blob<Dtype>*>& params_;
#ifdef USE_MPI
  vector<MPI_Comm> comms_;
#endif
  Dtype *diff_all_;
  Dtype *data_all_;
  bool cube_;
  bool avgdata_;
  bool rotate_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

