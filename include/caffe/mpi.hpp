#ifndef CAFFE_MPI_HPP_
#define CAFFE_MPI_HPP_

#ifdef USE_MPI
#include <mpi.h>
#endif

#define NO_MPI LOG(FATAL) << "Cannot use MPI unless USE_MPI is enabled during make."

namespace caffe {
namespace mpi {

#ifdef USE_MPI

extern MPI_Comm default_comm_;
MPI_Comm get_comm_default();
void set_comm_default(MPI_Comm comm=MPI_COMM_WORLD);

void init(int *argc, char ***argv);
bool initialized();
int query_thread();
void finalize();

MPI_Comm comm_dup(MPI_Comm comm=MPI_COMM_NULL);
void comm_free(MPI_Comm comm);
int comm_rank(MPI_Comm comm=MPI_COMM_NULL);
int comm_size(MPI_Comm comm=MPI_COMM_NULL);
int node_rank(MPI_Comm comm=MPI_COMM_NULL);
int node_size(MPI_Comm comm=MPI_COMM_NULL);

template <typename Dtype>
MPI_Datatype datatype();

void allreduce_copy(const float& sendbuf, float& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double& sendbuf, double& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce(float& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce(double& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce(float* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce(double* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void bcast(float* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(double* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe

#endif // CAFFE_MPI_HPP_
