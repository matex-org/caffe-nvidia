#include <vector>

#include "caffe/layers/base_data_layer.hpp"
// #include "caffe/util/blocking_queue.hpp"

namespace caffe {

extern int get_num_caches();

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "FGPU Call";
#ifdef USE_DEEPMEM
  typedef MemoryCache<Dtype> MemCacheType;
  Batch<Dtype> * batch;
  volatile bool * dirty;
  // volatile bool *dirtybit;
  PopBatch<Dtype> *p_batch;
  // DLOG(INFO) << "FGPU Call DEEPMEM";
  if(this->cache_size_)
  {
    //Do we handle the refill on l1 cache?
    if(!caches_[0]->prefetch && caches_[0]->empty()) //empty cache
    {
      //LOG(INFO) << "Local Refill ";
      //Refill before poping using the policy we have
      (caches_[0]->*(caches_[0]->local_refill_policy))(1);
    }
    DLOG(INFO) << "l0CACHE_FULL2_SIZE(beforepop_cu):" << l0cache_full2_.size();
    p_batch = l0cache_full2_.pop("Cache not ready yet");
    batch = p_batch->batch;
    // dirty = p_batch->dirty;
  }
  else //Use the original unmofified code to get a batch
  {
    batch = prefetch_full_.pop("DEEPMEMCACHE DataLayer Full Queue Empty (gpu)");
  }
#else
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
#endif

  // check batch has finished copying to the device
  CUDA_CHECK(cudaStreamWaitEvent(cudaStreamDefault, batch->copied_, 0));

  // Reshape to loaded data.
  if (this->transform_param_.use_gpu_transform()) {
    // instead of copy, perform out-of-place transform(!)
    this->data_transformer_->TransformGPU(top[0]->num(),
                                       top[0]->channels(),
                                       batch->data_.height(),
                                       batch->data_.width(),
                                       batch->data_.gpu_data(),
                                       top[0]->mutable_gpu_data(),
                                       batch->random_vec_.mutable_gpu_data());
  }  else {
    // Copy the data
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
               top[0]->mutable_gpu_data());
  }

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#ifdef USE_DEEPMEM
  if(this->cache_size_) {
  // We finished copy the batch so mark it for replacement
    // volatile bool tdirty = true;
    // *((volatile bool*)p_batch->dirty) =  &tdirty
    // *dirty = true;
    batch->dirty = true;
    // *(p_batch->dirty) = true;
    // p_batch->pushed_to_gpu->store(false, boost::memory_order_relaxed); //= false;
    // p_batch->pushed_to_gpu->store(false, boost::memory_order_consume); //= false;
    l0cache_free_.push(p_batch);
    DLOG(INFO) << "l0CACHE_free_SIZE(afterpush_cu):" << l0cache_free_.size();

  }
  //Use the orginal code if caches are turned off
  batch->count -= 1;
  if(this->cache_size_ == 0 || this->caches_[0]->size == 0){
    if(batch->count > 0) {
      DLOG(INFO) << "Batch Reuse Count: " << batch->count;
      prefetch_full_.push(batch);
    } else {
      batch->full_reused = true;
      prefetch_free_.push(batch);
    }
  }
#else
  prefetch_free_.push(batch);
#endif
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe

    // } else if(batch->shuffle_count > 0 && shuffle_batches == true) {
      // non-blocking queue
    //  prefetch_shuffle_.push(batch);
// #ifndef CPU_ONLY
//     cudaStream_t stream;
//     if (Caffe::mode() == Caffe::GPU) {
//       CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//     }
// #endif

    //int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    //if (Caffe::mode() == Caffe::CPU) {
    // if (!prefetch) {
    //   this->GetBatch();
    // }
