#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>
#ifdef USE_DEEPMEM
#include <iostream>
#include <fstream>
#include "caffe/util/cache.hpp"
#include <boost/atomic.hpp>
#include <boost/memory_order.hpp>
#include <cstdlib>
#endif

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_deque.hpp"

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

  virtual inline void PassParameterToLayer(const int value) {}

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

  virtual void PassParameterToLayer(const int value) {
    LOG(INFO) << "Data Layer New Reuse Count: " << value;
    this->reuse_count = value;
  }

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
#ifdef USE_DEEPMEM
  virtual void Pass_Value_To_Layer(Dtype value, unsigned int position) {
    //LOG(INFO) << "Base Pass";
    //ignoreAccuracy_=false;
    historical_accuracy_.push_back(value);
  }
  int cache_size_;
  virtual void shuffle();
#endif

 protected:
#ifdef USE_DEEPMEM
  bool prefetch;
  void refill_cache(int current_cache);
#endif
  virtual void InternalThreadEntry();
// #ifdef USE_DEEPMEM
//  virtual void load_batch(Batch<Dtype>* batch, bool in_thread) = 0;
// #else
  virtual void load_batch(Batch<Dtype>* batch) = 0;
// #endif

#ifdef USE_DEEPMEM
  virtual size_t reader_full_queue_size() = 0; //{
    // DataLayer<Dtype> *d_layer =
    //    dynamic_cast<DataLayer<Dtype> *>(this);
    // return d_layer->reader_full_queue_size();
  //}
  void rate_replace_policy(int next_cache);
  void thread_rate_replace_policy(int next_cache);

  void copy_batch(Batch<Dtype> *cbatch) { cbatch = disk_copy_.front(); disk_copy_.pop();}
  std::size_t get_copy_qsize() { return disk_copy_.size(); }

  GenRandNumbers randomGen;

  int reuse_count;
  int prefetch_count;
#endif

  // Batch<Dtype> prefetch_[PREFETCH_COUNT];
  // Batch<Dtype> * prefetch_;
  std::vector<Batch<Dtype>* > prefetch_;
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  std::vector<PopBatch<Dtype>* > pop_prefetch_;
  BlockingQueue<PopBatch<Dtype> > pop_prefetch_free_;
  BlockingQueue<PopBatch<Dtype> > pop_prefetch_full_;

  // copy batch into disk as it is fed to full queue
  // BlockingQueue<PopBatch<Dtype> > disk_copy_;
  std::queue<Batch<Dtype>*> disk_copy_;

  Blob<Dtype> transformed_data_;

#ifdef USE_DEEPMEM
  Cache<Dtype> ** caches_;
  vector<Dtype> historical_accuracy_;

  friend class Cache<Dtype>;
  friend class MemoryCache<Dtype>;
  friend class DiskCache<Dtype>;
#endif
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
