#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>
#ifdef USE_DEEPMEM
#include <iostream>
#include <fstream>
#include "caffe/util/cache.hpp"
#endif

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

#ifndef USE_DEEPMEM
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
#ifndef CPU_ONLY
  cudaEvent_t copied_;
#endif
  // stored random numbers for this batch
  Blob<int> random_vec_;
};
#endif

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
#ifdef USE_DEEPMEM
  virtual void Pass_Value_To_Layer(Dtype value, unsigned int position) {
    //LOG(INFO) << "Base Pass";
    //ignoreAccuracy_=false;
    historical_accuracy_.push_back(value);
  }
  int cache_size_;
#endif 

 protected:
#ifdef USE_DEEPMEM
  bool prefetch;
  void refill_cache(int current_cache);
#endif
  virtual void InternalThreadEntry();
#ifdef USE_DEEPMEM
  virtual void load_batch(Batch<Dtype>* batch, bool in_thread) = 0;
#else
  virtual void load_batch(Batch<Dtype>* batch) = 0;
#endif 

#ifdef USE_DEEPMEM
  void rate_replace_policy(int next_cache);
  void thread_rate_replace_policy(int next_cache);
  

  GenRandNumbers randomGen;
#endif
  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

#ifdef USE_DEEPMEM  
  Cache<Dtype> ** caches_;
  vector<Dtype> historical_accuracy_;
#endif

  Blob<Dtype> transformed_data_;

#ifdef USE_DEEPMEM  
  friend class Cache<Dtype>;
  friend class MemoryCache<Dtype>;
  friend class DiskCache<Dtype>;
#endif 
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
