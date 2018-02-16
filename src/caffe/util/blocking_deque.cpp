#include <boost/thread.hpp>
#include <string>

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_deque.hpp"

namespace caffe {

template<typename T>
class BlockingDeque<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
BlockingDeque<T>::BlockingDeque()
    : sync_(new sync()), count_(), local_count_() {
}

template<typename T>
void BlockingDeque<T>::push(const T& t) {
  // Always push at the back
  boost::mutex::scoped_lock lock(sync_->mutex_);
  // queue_.push(t);
  deque_.push_back(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
void BlockingDeque<T>::insert(const T& t, int pos) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  typedef typename std::deque<T>::iterator ItrType;
  ItrType itr = deque_.begin();
  itr += pos;
  deque_.insert(itr, t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingDeque<T>::try_pop(T* t) {
  // Always pop from front
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (deque_.empty()) {
    return false;
  }

  // *t = queue_.front();
  *t = deque_.front();
  // queue_.pop();
  deque_.pop_front();
  return true;
}

template<typename T>
T BlockingDeque<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  // while (queue_.empty()) {
  while (deque_.empty()) {
    if (!log_on_wait.empty()) {
      ++local_count_;
      if(local_count_ == 1000) {
        ++count_; local_count_ = 0;
      }
      LOG_EVERY_N(INFO, 1000)<< log_on_wait << " , Wait Count: " << count_;
    }
    sync_->condition_.wait(lock);
  }

  // T t = queue_.front();
  T t = deque_.front();
  // queue_.pop();
  deque_.pop_front();
  return t;
}

template<typename T>
bool BlockingDeque<T>::try_peek(T* t) {
  // Peek the first element
  boost::mutex::scoped_lock lock(sync_->mutex_);

  // if (queue_.empty()) {
  if (deque_.empty()) {
    return false;
  }

  // *t = queue_.front();
  *t = deque_.front();
  return true;
}

template<typename T>
T BlockingDeque<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  // while (queue_.empty()) {
  while (deque_.empty()) {
    sync_->condition_.wait(lock);
  }

  // return queue_.front();
  return deque_.front();
}

template<typename T>
size_t BlockingDeque<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  // return queue_.size();
  return deque_.size();
}

template class BlockingDeque<Batch<float>*>;
template class BlockingDeque<Batch<double>*>;
template class BlockingDeque<PopBatch<float>*>;
template class BlockingDeque<PopBatch<double>*>;
template class BlockingDeque<shared_ptr<Batch<float> > >;
template class BlockingDeque<shared_ptr<Batch<double> > >;
// template class BlockingQueue<Datum*>;
template class BlockingDeque<string*>;
template class BlockingDeque<shared_ptr<DataReader::QueuePair> >;
template class BlockingDeque<P2PSync<float>*>;
template class BlockingDeque<P2PSync<double>*>;

}  // namespace caffe
