#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_deque.hpp"

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
#ifndef USE_DEEPMEM
      prefetch_free_(), prefetch_full_() {
#else
      //prefetch_free_(), prefetch_full_(), prefetch_reuse_(), prefetch_shuffle_() {
      prefetch_free_(), prefetch_full_(), prefetch_shuffle_() {
// #ifdef USE_DEEPMEM
  //LOG(INFO) << "Cache size" << param.data_param().cache_size(0);
  cache_size_ = param.data_param().cache_size();
  LOG(INFO) << "Caches " << cache_size_;
  DLOG(INFO) << "BPDL Initialization";
  prefetch=false;
  shuffle_batches = false; // Default value. TODO: change this!
  // DLOG(INFO) << "CacheSize: " << cache_size_;
  if(cache_size_)
  {
    // Allocate No of Caches (Count = Num Level of Caches ? )
    caches_ = new Cache<Dtype> * [cache_size_];
    DLOG(INFO) << "Cache Created ,num: " << cache_size_;
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
      // caches_[i-1]->create( new Batch<Dtype>[caches_[i-1]->size], new bool[caches_[i-1]->size], thread_safe );
      caches_[i-1]->create( new Batch<Dtype>[caches_[i-1]->size]
          , new bool[caches_[i-1]->size]
          // , new bool[caches_[i-1]->size]
          , new boost::atomic<volatile bool>[caches_[i-1]->size]
          , thread_safe );
    }
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      caches_[i-1] = new DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      // caches_[i-1]->create( new Batch<Dtype>[2], new bool[caches_[i-1]->size], thread_safe );
      caches_[i-1]->create( new Batch<Dtype>[2]
          , new bool[caches_[i-1]->size]
          // , new bool[caches_[i-1]->size]
          , new boost::atomic<volatile bool>[caches_[i-1]->size]
          , thread_safe );
    }
//  #endif
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
    LOG(INFO) << "Caches " <<  param.data_param().cache(j).disk_location() << " " << caches_[i-1]->disk_location;
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
  DLOG(INFO) << "BasePrefetchingData LayerSetup";
  randomGen.Init();
#endif
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
      CUDA_CHECK(cudaEventCreate(&prefetch_[i].copied_));
    }
#endif
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    // prefetch_[i].count = 10;
    prefetch_[i].count = 3;
    prefetch_[i].shuffle_count = 10;
    prefetch_[i].dirty = true;
    prefetch_[i].full_reused = true;
  }

#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->mutate_data(this->output_labels_);
  }
#endif
  }
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();

#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->fill(false);
  }

  if(cache_size_) {
  // typedef MemoryCache<Dtype> MemCacheType;
  // MemCacheType * memcache;
  // if(( memcache = dynamic_cast<MemCacheType*>(caches_[0]))) {
    // for (int i = 0; i < memcache->size; i++) {
    for (int i = 0; i < caches_[0]->size; i++) {
      l0cache_free_.push(caches_[0]->pop());
    }
  // }
  }
#endif
/*
  // Only if GPU mode on then we use background threads
#ifdef USE_DEEPMEM
//If the global prefetch is set create a prefetch thread which is just below
  if (prefetch) {
#else
  if (Caffe::mode() == Caffe::GPU) {
#endif
*/

  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::shuffle() {
  bool done = false;
  std::size_t shuffle_remaining = prefetch_shuffle_.size();
  // std::vector<ListType::iterator> itrVec;
  while (!done) {

    if(shuffle_remaining < 2) {
      done = true;
      break;
    }
    Batch<Dtype>* b1 = prefetch_shuffle_.front();
    prefetch_shuffle_.pop();
    Batch<Dtype>* b2 = prefetch_shuffle_.front();
    prefetch_shuffle_.pop();
    if(!b1->shuffled && !b2->shuffled) {
      int rand;
      for(int i = 0; i < b1->data_.shape(0); i++) {
        rand = randomGen(b2->data_.shape(0));
        MemoryCache<Dtype>::shuffle_cache(b1, i, b2, rand);
      }
      b1->shuffled = true;
      b2->shuffled = true;
      b1->shuffle_count -= 1;
      b2->shuffle_count -= 1;
      shuffle_remaining -= 2;
      prefetch_shuffle_.push(b1);
      prefetch_shuffle_.push(b2);
    }
  };
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
DLOG(INFO) << "InternalThrdEnt";
#ifdef USE_DEEPMEM
  PopBatch<Dtype> *pbatch;
  try {
    while (!must_stop()) {
      if(cache_size_)
      {
        for(int i=cache_size_-1; i>= 0; i--)
        {
          //If we handle the refilling apply the member pointer to the current Cache class
          if(caches_[i]->prefetch)
            (caches_[i]->*(caches_[i]->refill_policy))(1);
        }
        DLOG(INFO) << "l0CACHE_FREE_SIZE:" << l0cache_free_.size();

        if(!l0cache_free_.peek()->batch->dirty) {
        PopBatch<Dtype> *p_batch = l0cache_free_.pop("Cache pop");
        // *p_batch->dirty = false;
        // p_batch = caches_[0]->pop();
        Batch<Dtype>* batch = p_batch->batch;
#ifndef CPU_ONLY
        // bool dty = (bool)*p_batch->dirty;
        // bool pgpu = (bool)p_batch->pushed_to_gpu->load(boost::memory_order_release);
        // if(!pgpu && !dty) {
          if (Caffe::mode() == Caffe::GPU) {
            batch->data_.data()->async_gpu_push();
            if (this->output_labels_) {
                batch->label_.data()->async_gpu_push();
            }
            cudaStream_t stream = batch->data_.data()->stream();
            // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            DLOG(INFO) << "Before CudaEvenRecord ";
            CUDA_CHECK(cudaEventRecord(batch->copied_, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          // *p_batch->pushed_to_gpu = true;
          // batch->count_ += 1;
          l0cache_full2_.push(p_batch);
          DLOG(INFO) << "l0CACHE_FULL2_SIZE:" << l0cache_full2_.size();
        // }
#endif
        }
      } else {
        // Use Default approach:
        Batch<Dtype>* batch; std::size_t reuse_count; bool f_reuse;
        if(prefetch_shuffle_.size() >= 2) {
          // shuffle data in shuffle_queue (non blocking queue)
          shuffle();
          Batch<Dtype>* b = prefetch_shuffle_.front();
          prefetch_shuffle_.pop();
#ifndef CPU_ONLY
          if(b->shuffled = true) {
            b->data_.data()->async_gpu_recopy();
            if(this->output_labels_)
              b->label_.data()->async_gpu_recopy();
            b->shuffled = false;
#endif
            prefetch_full_.push(b);
          } else {
            prefetch_shuffle_.push(b);
          }
        } else {
            batch = prefetch_free_.pop("DEEPMEMCACHE DataLayer Free Queue Empty");
            load_batch(batch);
#ifndef CPU_ONLY
            if (Caffe::mode() == Caffe::GPU) {
              batch->data_.data()->async_gpu_push();
              if (this->output_labels_) {
                  batch->label_.data()->async_gpu_push();
              }
              cudaStream_t stream = batch->data_.data()->stream();
              // CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
              CUDA_CHECK(cudaEventRecord(batch->copied_, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              prefetch_full_.push(batch);
#endif
            }
        }
      }
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
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
  DLOG(INFO) << "FCPU Call";
#ifdef USE_DEEPMEM
  Batch<Dtype> * batch;
  PopBatch<Dtype>* p_batch;
  //If there are any caches
  DLOG(INFO) << "FCPU Call DEEPMEM";
  if(cache_size_)
  {
    //Do we handle the refill on l1 cache?
    if(!caches_[0]->prefetch && caches_[0]->empty()) //empty cache
    {
      //LOG(INFO) << "Local Refill ";
      //Refill before poping using the policy we have
      (caches_[0]->*(caches_[0]->local_refill_policy))(1);
    }
    p_batch = l0cache_full2_.pop("Cache not ready yet");
    batch = p_batch->batch;
  }
  else //Use the original unmodified code to get a batch
  {
    batch = prefetch_full_.pop("Prefetch cache queue empty");
  }
#else
  DLOG(INFO) << "FCPU Original Call, no DEEPMEM";
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
  {
    batch->dirty = true;
    // *p_batch->dirty = true;
    // *p_batch->pushed_to_gpu = false;
    l0cache_free_.push(p_batch);
  }
  //Use the orginal code if caches are turned off
  if(cache_size_ == 0 || caches_[0]->size == 0)
    prefetch_free_.push(batch);
#else
  prefetch_free_.push(batch);
#endif
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe

/*
        // SyncedMemory::SyncedHead dataHead = l0cache_free_.peek()->batch->data_.data()->head();
        // if(dataHead == SyncedMemory::SYNCED)
        //   DLOG(INFO) << "GPU SYNCED ALREADY!";

        // if(dataHead =! SyncedMemory::SYNCED) {
#ifdef USE_DEEPMEM
    if(cache_size_) {
      typedef MemoryCache<Dtype> MemCacheType;
      //Setup the caches data
      for (int i = 0; i < cache_size_; ++i) {
        for (int j = 0; j < caches_[i]->size; ++j) {
          MemCacheType * memcache;
          // DiskCacheType * diskcache;
          if((memcache
              = dynamic_cast<MemCacheType *>(caches_[i]))) {
            memcache->cache[j].data_.mutable_cpu_data();
            if (this->output_labels_) {
              memcache->cache[j].label_.mutable_cpu_data();
            }
#ifndef CPU_ONLY
            if (Caffe::mode() == Caffe::GPU) {
              memcache->cache[j].data_.mutable_gpu_data();
              if (this->output_labels_) {
                memcache->cache[j].label_.mutable_gpu_data();
              }
              CUDA_CHECK(cudaEventCreate(&memcache->cache[j].copied_));
            }
#endif
          } else {
          LOG(INFO) << "Unsupported Cache Type. DiskCache?";
        }
      }
    }
#ifndef CPU_ONLY
    MemCacheType * memcache;
    if(( memcache = dynamic_cast<MemCacheType *>(caches_[0]))) {
      l0cache_free_.push(&(memcache->cache[memcache->last_i]));
    }
#endif
    // load/fill batches to the caches
    // for (int i = 0; i < cache_size_; ++i) {
    //   caches_[i]->fill(false);
    // }
    }
#endif
*/
// #ifndef CPU_ONLY
//   cudaStream_t stream;
//   if (Caffe::mode() == Caffe::GPU) {
//     CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//   }
// #endif
//  #ifdef KNL
//    else if(param.data_param().cache(j).type() == CacheParameter::HBM)
//    {
//      //Use hbm to create a new cache, set size, and dirty structure
//      void * ptr = hbw_malloc(sizeof(MemoryCache<Dtype>));
//      caches_[i-1] = new (ptr) MemoryCache<Dtype>;
//      caches_[i-1]->size = param.data_param().cache(j).size();
//      ptr = hbw_malloc(sizeof(Batch<Dtype>)*caches_[i-1]->size);
//      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
//      caches_[i-1]->create( new (ptr) Batch<Dtype>[caches_[i-1]->size], ptr2, thread_safe );
//    }
//    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
//    {
//      //Use hbm to create a new cache, read/write bufs, set size, and dirty structure
//      void * ptr = hbw_malloc(sizeof(DiskCache<Dtype>));
//      caches_[i-1] = new (ptr) DiskCache<Dtype>;
//      caches_[i-1]->size = param.data_param().cache(j).size();
//      //Read/write buffer
//      ptr = hbw_malloc(sizeof(Batch<Dtype>)*2);
//      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
//      caches_[i-1]->create( new (ptr) Batch<Dtype>[2], ptr2, thread_safe );
//    }
//  #else

//int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    //if (Caffe::mode() == Caffe::CPU) {
    // if (!prefetch) {
    //   this->GetBatch();
    // }

// typedef MemoryCache<Dtype> MemCacheType;
  // if(cache_size_) {
  //   for (int i = 0; i < caches_[0]->size; i++) {
  //     PopBatch<Dtype> *pbatch;
  //     MemCacheType * memcache;
  //     if((memcache
  //       = dynamic_cast<MemCacheType *>(caches_[0]))) {
  //       // pbatch->batch = &memcache->cache[i];
  //       // pbatch->dirty = &memcache->dirty[i];
  //       pbatch->batch = &memcache->cache[i];
  //       pbatch->dirty = &memcache->dirty[i];
  //       l0cache_free_.push(pbatch);
  //     }
  //   }
  // }
