#ifndef CAFFE_MPI_HPP_
#define CAFFE_MPI_HPP_

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <string>
#include <vector>

#define NO_MPI LOG(FATAL) << "Cannot use MPI unless USE_MPI is enabled during make."

namespace caffe {
namespace mpi {

#ifdef USE_MPI

extern MPI_Comm default_comm_;
MPI_Comm get_comm_default();
void set_comm_default(MPI_Comm comm=MPI_COMM_WORLD);

void init(int *argc, char ***argv, const std::string &FLAGS_mpi);
bool initialized();
int query_thread();
void finalize();

MPI_Comm comm_dup(MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_split(int color, int key, MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_create(MPI_Group group, MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_create(const std::vector<int> &incl, MPI_Comm comm=MPI_COMM_NULL);
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

void iallreduce(MPI_Request &request, float* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void iallreduce(MPI_Request &request, double* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

bool testall(std::vector<MPI_Request> &requests);
void waitall(std::vector<MPI_Request> &requests);
bool test(MPI_Request &request);

void bcast(std::vector<int> &buffer, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(float* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(double* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);

void send(const float *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void send(const double *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void recv(float *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void recv(double *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void sendrecv(const int *sendbuf, int sendcount, int dest, int sendtag,
    int *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const signed char *sendbuf, int sendcount, int dest, int sendtag,
    signed char *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const float *sendbuf, int sendcount, int dest, int sendtag,
    float *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const double *sendbuf, int sendcount, int dest, int sendtag,
    double *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);

void isend(MPI_Request &request, const signed char *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const int *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const float *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const double *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void irecv(MPI_Request &request, signed char *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, int *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, float *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, double *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe

#endif // CAFFE_MPI_HPP_
