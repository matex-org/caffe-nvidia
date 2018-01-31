#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
#ifndef cpu_only
  if (gpu_ptr_ && own_gpu_data_) {
#ifdef debug
    cudapointerattributes attr;
    cudaerror_t status = cudapointergetattributes(&attr, gpu_ptr_);
    if (status == cudasuccess) {
      check_eq(attr.memorytype, cudamemorytypedevice);
      check_eq(attr.device, gpu_device_);
    }
#endif
    // gpumemory::deallocate(gpu_ptr_, gpu_device_, stream_);
    GPUMemory::deallocate(gpu_ptr_, gpu_device_, stream_);
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    // DLOG(INFO) << "HEAD IN SYNC!!!!!!!!!1";
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    stream_ = GPUMemory::device_stream(gpu_device_);
    GPUMemory::allocate(&gpu_ptr_, size_, gpu_device_, stream_);
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    DLOG(INFO) << "HEAD IN HEAD_AT_GPU!!!!!!!!!1";
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      stream_ = GPUMemory::device_stream(gpu_device_);
      GPUMemory::allocate(&gpu_ptr_, size_, gpu_device_, stream_);
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    // DLOG(INFO) << "HEAD IN SYNC!!!!!!!!!2";
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (gpu_ptr_ && own_gpu_data_) {
    GPUMemory::deallocate(gpu_ptr_, gpu_device_, stream_);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  DLOG(INFO) << "HEAD IN HEAD_AT_GPU!!!!!!!!!2";
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  // DLOG(INFO) << "HEAD IN HEAD_AT_GPU!!!!!!!!!3";
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push() {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    stream_ = GPUMemory::device_stream(gpu_device_);
    GPUMemory::allocate(&gpu_ptr_, size_, gpu_device_, stream_);
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream_));
  // Assume caller will synchronize on the stream before use
  // DLOG(INFO) << "HEAD IN SYNC!!!!!!!!!3";
  head_ = SYNCED;
}

void SyncedMemory::async_gpu_recopy() {
  CHECK(head_ == SYNCED);
  if (gpu_ptr_ != NULL) {
// #ifndef CPU_ONLY
    if (gpu_ptr_ && own_gpu_data_) {
#ifdef debug
      cudapointerattributes attr;
      cudaerror_t status = cudapointergetattributes(&attr, gpu_ptr_);
      if (status == cudasuccess) {
        check_eq(attr.memorytype, cudamemorytypedevice);
        check_eq(attr.device, gpu_device_);
      }
#endif
     GPUMemory::deallocate(gpu_ptr_, gpu_device_, stream_);
    }
  }
  CUDA_CHECK(cudaGetDevice(&gpu_device_));
  stream_ = GPUMemory::device_stream(gpu_device_);
  GPUMemory::allocate(&gpu_ptr_, size_, gpu_device_, stream_);
  own_gpu_data_ = true;
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream_));
  // Assume caller will synchronize on the stream before use
}
#endif

}  // namespace caffe
