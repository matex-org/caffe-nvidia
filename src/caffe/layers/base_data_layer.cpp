#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#ifdef KNL
#include <hbwmalloc.h>
#endif

#ifdef USE_MLSL
using namespace MLSL;
#endif /* USE_MLSL */

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
#ifdef USE_DEEPMEM
  //LOG(INFO) << "Cache size" << param.data_param().cache_size(0);
  cache_size_ = param.data_param().cache_size();
  LOG(INFO) << "Caches " << cache_size_;
  prefetch=false;
  if(cache_size_)
  {
    //Allocate array to hold caches on heap or in hbm
    #ifdef KNL
      void * ptr = hbw_malloc(sizeof(Cache<Dtype> *) * cache_size_);
      caches_ = new (ptr) Cache<Dtype> * [cache_size_];
    #else  
      caches_ = new Cache<Dtype> * [cache_size_];
    #endif

  }
  for(int i = cache_size_, j=0; i > 0; i--, j++)
  {

    bool thread_safe = param.data_param().cache(j).thread_safe();
    
    //If one cache is thread_save then set this global to turn the prefetcher
    //Thread
    if(thread_safe)
      prefetch = true;

    if(param.data_param().cache(j).type() == CacheParameter::HEAP)
    {
      //Create a new cache, set size, a dirty structure
      caches_[i-1] = new MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      caches_[i-1]->create( new Batch<Dtype>[caches_[i-1]->size], new bool[caches_[i-1]->size], thread_safe );
    }
  #ifdef KNL
    else if(param.data_param().cache(j).type() == CacheParameter::HBM)
    {
      //Use hbm to create a new cache, set size, and dirty structure
      void * ptr = hbw_malloc(sizeof(MemoryCache<Dtype>));
      caches_[i-1] = new (ptr) MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      ptr = hbw_malloc(sizeof(Batch<Dtype>)*caches_[i-1]->size);
      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
      caches_[i-1]->create( new (ptr) Batch<Dtype>[caches_[i-1]->size], ptr2, thread_safe );
    }
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      //Use hbm to create a new cache, read/write bufs, set size, and dirty structure
      void * ptr = hbw_malloc(sizeof(DiskCache<Dtype>));
      caches_[i-1] = new (ptr) DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      //Read/write buffer
      ptr = hbw_malloc(sizeof(Batch<Dtype>)*2);
      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
      caches_[i-1]->create( new (ptr) Batch<Dtype>[2], ptr2, thread_safe );
    }
  #else
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      caches_[i-1] = new DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      caches_[i-1]->create( new Batch<Dtype>[2], new bool[caches_[i-1]->size], thread_safe );
    }
  #endif
    else
    {
      LOG(INFO) << "Cache Type not supported";
      exit(1);
    }
   
    //Setup cache to point one level above 
    if(i-1==cache_size_-1)
      caches_[i-1]->next = NULL; 
    else  
      caches_[i-1]->next = caches_[i]; 
   
    // Pass data_layer (used for filling)
    caches_[i-1]->data_layer = this;
    //Initially needs to be filled
    caches_[i-1]->used = caches_[i-1]->size;
    caches_[i-1]->refill_start = 0;
    caches_[i-1]->current_shuffle_count = 0;
    caches_[i-1]->eviction_rate = param.data_param().cache(j).eviction_rate();
    caches_[i-1]->refill_policy = &Cache<Dtype>::rate_replace_policy;
    caches_[i-1]->local_refill_policy = &Cache<Dtype>::local_rate_replace_policy;
    caches_[i-1]->disk_location = param.data_param().cache(j).disk_location();
    LOG(INFO) << "Cacher " <<  param.data_param().cache(j).disk_location() << " " << caches_[i-1]->disk_location;
  }
  
  //Setup cache to point one level below
  for(int j=0; j < cache_size_; j++)
  {
    if(j==0)
      caches_[j]->prev = NULL; 
    else
      caches_[j]->prev = caches_[j-1]; 
  }
#endif 

  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
#ifdef USE_DEEPMEM
  randomGen.Init();
#endif 
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->mutate_data(this->output_labels_);
  }
#endif
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
      CUDA_CHECK(cudaEventCreate(&prefetch_[i].copied_));
    }
#ifdef USE_DEEPMEM
    //Setup the caches data
    for (int i = 0; i < cache_size_; ++i) {
      for (int j = 0; j < caches_[i]->size; ++j) {
        caches_[i]->cache[j].data_.mutable_gpu_data();
        if (this->output_labels_) {
          caches_[i]->cache[j].label_.mutable_gpu_data();
        }
      }
    }
#endif
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();

#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->fill(false);
  }
#endif

  // Only if GPU mode on then we use background threads
#ifdef USE_DEEPMEM
//If the global prefetch is set create a prefetch thread which is just below
  if (prefetch) {
#else
  if (Caffe::mode() == Caffe::GPU) {
#endif
   StartInternalThread();
  }
  // StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {

#ifdef USE_DEEPMEM
 while (!must_stop()) {
    if(cache_size_)
    {
      for(int i=cache_size_-1; i>= 0; i--)
      {
        //If we handle the refilling apply the member pointer to the current
        //Cache class
        if(caches_[i]->prefetch)
          (caches_[i]->*(caches_[i]->refill_policy))(1);
      }
    }
  }
#else
  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data()->async_gpu_push();
        if (this->output_labels_) {
            batch->label_.data()->async_gpu_push();
        }
        cudaStream_t stream = batch->data_.data()->stream();
        CUDA_CHECK(cudaEventRecord(batch->copied_, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_DEEPMEM
  Batch<Dtype> * batch;
  PopBatch<Dtype> pop_batch;
  //If there are any caches
  if(cache_size_)
  {
    //Do we handle the refill on l1 cache?
    if(!caches_[0]->prefetch && caches_[0]->empty()) //empty cache
    {
      //LOG(INFO) << "Local Refill ";
      //Refill before poping using the policy we have
      (caches_[0]->*(caches_[0]->local_refill_policy))(1);
    }
    pop_batch = caches_[0]->pop();
    batch = pop_batch.batch;
  }
  else //Use the original unmofified code to get a batch
  {
    //int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    //if (Caffe::mode() == Caffe::CPU) {
    if (!prefetch) {
      this->GetBatch();
    }
    batch = prefetch_full_.pop("Prefetch cache queue empty");
  }
#else
  // Here for CPU we do transformation
  if (Caffe::mode() == Caffe::CPU) {
    this->GetBatch();
  }
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
#endif

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
#ifdef USE_DEEPMEM
  if(cache_size_) // We finished copy the batch so mark it for replacement
    *pop_batch.dirty = true;
  //Use the orginal code if caches are turned off
  if(cache_size_ == 0 || caches_[0]->size == 0)
    prefetch_free_.push(batch);
#endif

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
