
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/blocking_deque.hpp"
#include "caffe/util/cache.hpp"

namespace caffe {

//These are replacement policies based to the individual caches:
//This is for the thread
template <typename Dtype>
void Cache<Dtype>::rate_replace_policy(int next_cache)
{
  // typedef DiskCache<Dtype> DiskCacheType;
  // DiskCacheType *diskcache;
  if(next == NULL) //Last level -> refill
  {
    //Fill in the cache -> we pass true to indicate to not use openmp in the data
    //Transformer since it will spawn more oepnmp threads
    typedef DiskCache<Dtype> DiskCacheType;
    DiskCacheType *diskcache;
    if(!(diskcache = dynamic_cast<DiskCacheType *>(this)))
      fill(true);
  }
  else //Refill level of the cache
  {
    //Refill higher levels-> not really required because we handle this issue
    //inside also during poping
    if(next->prefetch && next->empty() ) //empty cache
      (next->*(next->refill_policy))(next_cache+1);
    boost::random::uniform_int_distribution<> dist(5, 10);
    int random_val = dist(gen);

    typedef DiskCache<Dtype> DiskCacheType;
    DiskCacheType *diskcache;
    diskcache = dynamic_cast<DiskCacheType *>(next);
    DLOG(INFO) << "Reader FULL Queue Size: " <<
        Cache<Dtype>::data_layer->reader_full_queue_size()
        << "...... ";
    if((diskcache->size >= diskcache->disk_cache_min_size) &&
        // (Cache<Dtype>::data_layer->reader_full_queue_size() < random_val))
        (Cache<Dtype>::data_layer->reader_full_queue_size() < this->batch_size))
      refill(next);
    else
      fill(true);
  }
}
//Same as above but for within a node
template <typename Dtype>
void Cache<Dtype>::local_rate_replace_policy(int next_cache)
{
  // else if(next == NULL) //Last level -> refill
  if(next == NULL) //Last level -> refill
  {
    typedef DiskCache<Dtype> DiskCacheType;
    DiskCacheType *diskcache;
    if(!(diskcache = dynamic_cast<DiskCacheType *>(this)))
      fill(false); //We can use openmp here so false is passed since we are within
  }
  else
  {
    //LOG(INFO) << "Refilling Level " << next_cache-1 << " " << size;
    //Refill higher levels
    if(!next->prefetch && next->empty() ) //empty cache
      (next->*(next->local_refill_policy))(next_cache+1);
    typedef DiskCache<Dtype> DiskCacheType;
    DiskCacheType *diskcache;
    diskcache = dynamic_cast<DiskCacheType *>(next);
    if ((diskcache->size >= diskcache->disk_cache_min_size) &&
        (Cache<Dtype>::data_layer->reader_full_queue_size() < this->batch_size)
        )
      refill(next);
    else
      fill(true);
  }
}

//Shuffles image at pos 1 in batch 1 with image at pos 2 in batch 2
template <typename Dtype>
void MemoryCache<Dtype>::shuffle_cache(Batch<Dtype>* batch1, int batchPos1, Batch<Dtype>*  batch2, int batchPos2) {
  const int datum_channels = batch1->data_.shape(1);
  const int datum_height = batch1->data_.shape(2);
  const int datum_width = batch1->data_.shape(3);

  Dtype * data1 = batch1->data_.mutable_cpu_data();
  Dtype * data2 = batch2->data_.mutable_cpu_data();
  Dtype * label1 = batch1->label_.mutable_cpu_data();
  Dtype * label2 = batch2->label_.mutable_cpu_data();
  int offset1 = batch1->data_.offset(batchPos1);
  int offset2 = batch2->data_.offset(batchPos2);
  int top_index;
  data1+=offset1;
  data2+=offset2;

  int height = datum_height;
  int width = datum_width;

  // int h_off = 0;
  // int w_off = 0;
  //Fairly self explanatory -> swap labelsi, swap each channel for height and width
  std::swap(label1[batchPos1], label2[batchPos2]);
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        top_index = (c * height + h) * width + w;
        std::swap(data1[top_index], data2[top_index]);
      }
    }
  }
}
//Ptr is the cache buffer and pt2 is for the dirty buffer
template <typename Dtype>
// void MemoryCache<Dtype>::create( void * ptr, bool * ptr2, bool * ptr3, bool thread_safe )
void MemoryCache<Dtype>::create( void * ptr
    , boost::shared_ptr<std::vector<boost::shared_ptr<bool> > >ptr2
    , bool thread_safe )
{
  cache = static_cast<Batch<Dtype> *> (ptr);
  Cache<Dtype>::prefetch = thread_safe;
  Cache<Dtype>::full_replace = false;
  Cache<Dtype>::dirty = ptr2;
  // Cache<Dtype>::pushed_to_gpu = static_cast<boost::atomic<volatile bool>*>(ptr3);
  Cache<Dtype>::last_i = 0;
  Cache<Dtype>::slot = 0;
  //Initially the cache is dirty and needs to be filled out
  for(int i=0; i< Cache<Dtype>::size; i++) {
    // cache[i].dirty = true;
    // Cache<Dtype>::dirty[i] = true;
   *(*Cache<Dtype>::dirty)[i] = true;
    // Cache<Dtype>::pushed_to_gpu[i].store(false,boost::memory_order_relaxed); //= false;
  }
}

template <typename Dtype>
bool MemoryCache<Dtype>::empty()
{
  //int bounds = Cache<Dtype>::used.fetch_add(0, boost::memory_order_relaxed);
  return Cache<Dtype>::used == Cache<Dtype>::size;
}

template <typename Dtype>
PopBatch<Dtype> MemoryCache<Dtype>::pop()
{
  //Empty function above uses this variable
  //  Cache<Dtype>::used.fetch_add(1, boost::memory_order_relaxed);
  //Only 1 thread doing this
  int my_slot = Cache<Dtype>::slot++;

  if(Cache<Dtype>::slot == Cache<Dtype>::size)
    Cache<Dtype>::slot = 0;

  //LOG(INFO) << "Waiting " << this << " " << my_slot;
  //Wait til someone fills your slot or if somehow you get in a bad state and
  //This thread is suppose to refill the slot, but managed not to -> refill the
  //State above you
  // DLOG(INFO) << "Cache Dirty: " << *(*Cache<Dtype>::dirty)[my_slot];

  while(*(*Cache<Dtype>::dirty)[my_slot])
  {
    if(Cache<Dtype>::prev && this->prev->prefetch == Cache<Dtype>::prefetch)
    {
      if(this->prefetch)
      {
        (this->prev->*(this->prev->refill_policy))(1);
        (this->*(Cache<Dtype>::refill_policy))(1);
      }
      else
      {
        (this->prev->*(this->prev->local_refill_policy))(1);
        (this->*(Cache<Dtype>::refill_policy))(1);
      }
    }
  };
  //LOG(INFO) << "Waiting done" << this << " " << my_slot;

  PopBatch<Dtype> pbatch;
  //Data we are sending out
  pbatch.batch = &cache[my_slot];
  // pbatch.batch = boost::shared_ptr<Batch<Dtype> >(&cache[my_slot]);
            // boost::make_shared<Batch<Dtype> >(cache[my_slot]);
  //Structure to indicate that we are dirty once the popper copies the data
  // pbatch.dirty = &((Cache<Dtype>::dirty.get())[my_slot]);
  // std::vector<bool>& dirty_ref = *Cache<Dtype>::dirty;
  // pbatch.dirty = &(dirty_ref[my_slot]);
  pbatch.dirty = (*Cache<Dtype>::dirty)[my_slot];

  return pbatch;
}
template <typename Dtype>
void MemoryCache<Dtype>::shuffle()
{
  // int rand;
  //Basically we need to loop through each cache position and shuffle when it
  //is dirty, because this shuffle can happen on a different thread than the
  //thread using the data and we want to go through all the data without favoring
  //the first indexes we store i between calles to this function
  /* for (int i = Cache<Dtype>::last_i; i < Cache<Dtype>::size; ++i) {

    Cache<Dtype>::last_i=i; //Store i to use during the next function call
    if((*(*Cache<Dtype>::dirty)[i]) == true) //Cache pos needs to be replaced
    // if(cache[i].dirty == true) //Cache pos needs to be replaced
    {
      //For each image in the cache pos
      for(int j=0; j< cache[i].data_.shape(0); j++)
      {
        //Select a random cache pos to swap with
        rand = Cache<Dtype>::data_layer->randomGen(Cache<Dtype>::size);
        //Make sure it is dirty otherwise it hasn't been used yet
        while(!Cache<Dtype>::dirty[rand])
        // while(!cache[rand].dirty)
          rand = Cache<Dtype>::data_layer->randomGen(Cache<Dtype>::size);

        //Swap the current image j in cache[i] with the random cache at pos this is random
        shuffle_cache(&cache[i], j, &cache[rand], Cache<Dtype>::data_layer->randomGen(cache[i].data_.shape(0)));
      }
      //Decrement used to indicate emptiness of cache
      Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      //cache_full.push(&cache[i]);
      //LOG(INFO)  << "Shuffle used "  << Cache<Dtype>::used << " clear " << i;
      Cache<Dtype>::dirty[i] = false;
      // cache[i].dirty = false;
      Cache<Dtype>::last_i++;
    }
    else // Break if we can't shuffle in order
      break;
  }
  //We have shuffeled the whole cache once
  if(Cache<Dtype>::last_i == Cache<Dtype>::size)
  {
    //bounds=0;
    Cache<Dtype>::full_replace = true; //Indicate the cache has shuffled once
    Cache<Dtype>::last_i=0; //Start over
  }
  */
}
template <typename Dtype>
void MemoryCache<Dtype>::fill(bool in_thread)
{
  //Same logic as shuffle but it is reading data from the data_layer
  //in_thread indicates it is in a thread other than main
  //this is passed to load_batch which avoids calling openmp if it is true
  // DLOG(INFO) << "Memory Cache fill called.... ";
  // for (int j = Cache<Dtype>::last_i; j < Cache<Dtype>::size; ++j) {
  for (int j = Cache<Dtype>::slot; j < Cache<Dtype>::size; ++j) {
    // DLOG(INFO) << "MemCache Last_i: " << Cache<Dtype>::last_i;
    // DLOG(INFO) << "MemCache Slot_curr_value: " << Cache<Dtype>::slot;
    // Cache<Dtype>::last_i=j;
    // Cache<Dtype>::slot=j;
    if(*(*Cache<Dtype>::dirty)[j] == true)
    {
      // Cache<Dtype>::data_layer->load_batch(&cache[j], in_thread);
      Cache<Dtype>::data_layer->load_batch(&cache[j]);
      // Dirty bit set:
      // cache[j].dirty = false;  // Need this?
      // Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      *(*Cache<Dtype>::dirty)[j] = false;
      // Cache<Dtype>::last_i++;
      // Cache<Dtype>::last_i++;
      cache[j].count = this->reuse_count;
    }
    else
      break;
  }
  // if(this->last_i == Cache<Dtype>::size)
  if(this->slot == Cache<Dtype>::size)
  {
    Cache<Dtype>::full_replace = true;
    // Cache<Dtype>::last_i=0;
    Cache<Dtype>::slot=0;
  }
  // DLOG(INFO) << "Memory Cache fill called(end).... ";
}
template <typename Dtype>
void MemoryCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
  PopBatch<Dtype> pbatch;
  //Batch<Dtype> * temp_cache;

  // for (int j = Cache<Dtype>::last_i; j < Cache<Dtype>::size; ++j) {
  for (int j = Cache<Dtype>::slot; j < Cache<Dtype>::size; ++j) {
    // Cache<Dtype>::last_i=j;
    Cache<Dtype>::slot=j;
    if(*(*Cache<Dtype>::dirty)[j] == true)
    {
      if(Cache<Dtype>::data_layer->reader_full_queue_size() > this->batch_size) {
        Cache<Dtype>::data_layer->load_batch(&cache[j]);
        DLOG(INFO) << " Cache Load Batch(refill) CALLED.....";
      }
      else {
        DLOG(INFO) << " Cache Copy Batch(refill) CALLED.....";
        pbatch = next_cache->pop();
        DLOG(INFO) << "MemCACHE DATA SIZE: " << cache[j].data_.data()->size();
        DLOG(INFO) << "PBATCH DATA SIZE: " << pbatch.batch->data_.data()->size();
        // cache[j].data_.CopyFrom( pbatch.batch->data_ , false, true);
        // cache[j].label_.CopyFrom( pbatch.batch->label_, false, true );
        cache[j].data_.CopyFromCPU( pbatch.batch->data_ , false, true);
        cache[j].label_.CopyFromCPU( pbatch.batch->label_, false, true );
        *pbatch.dirty = true;
        pbatch.batch->dirty = true;
        // delete pbatch.batch;
      }

      // Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      *(*Cache<Dtype>::dirty)[j] = false;
      cache[j].dirty = false;
      // Cache<Dtype>::last_i++;
      cache[j].count = this->reuse_count;
    }
    else
      break;
  }
  // if(Cache<Dtype>::last_i == Cache<Dtype>::size)
  if(Cache<Dtype>::slot == Cache<Dtype>::size)
  {
    Cache<Dtype>::full_replace = true;
    // Cache<Dtype>::last_i=0;
    Cache<Dtype>::slot=0;
  }
}

template <typename Dtype>
void MemoryCache<Dtype>::reshape(vector<int> * top_shape, vector<int> * label_shape)
{
  for(int i=0; i< Cache<Dtype>::size; i++) {
      cache[i].data_.Reshape(*top_shape);
  }
  if (label_shape) {
    for(int i=0; i< Cache<Dtype>::size; i++) {
      cache[i].label_.Reshape(*label_shape);
    }
  }
}
template <typename Dtype>
void MemoryCache<Dtype>::mutate_data(bool labels, const int level)
{
  for(int i=0; i< Cache<Dtype>::size; i++) {
      cache[i].data_.mutable_cpu_data();
  }
  if (labels) {
    for(int i=0; i< Cache<Dtype>::size; i++) {
      cache[i].label_.mutable_cpu_data();
    }
  }
// /*
#ifndef CPU_ONLY
 if(level == 0) {
   if (Caffe::mode() == Caffe::GPU) {
      for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].data_.mutable_gpu_data();
	  }
      if (labels) {
		for(int i=0; i< Cache<Dtype>::size; i++) {
          cache[i].label_.mutable_gpu_data();
        }
	  }
	  for(int i=0; i< Cache<Dtype>::size; i++) {
        CUDA_CHECK(cudaEventCreate(&cache[i].copied_));
      }
   }
 }
#endif
// */
}
template <typename Dtype>
void DiskCache<Dtype>::shuffle_cache(int batch1, int batchPos1, int  batch2, int batchPos2, int image_count, int data_count, int label_count) {

  LOG(INFO) << "Error: Disk Caching Shuffle Disabled";
  /*unsigned int image1_loc = (batch1*(image_count*(data_count+1))*sizeof(Dtype))+(batchPos1*(data_count+1)*sizeof(Dtype));
  unsigned int image2_loc = (batch2*(image_count*(data_count+1))*sizeof(Dtype))+(batchPos2*(data_count+1)*sizeof(Dtype));
  int offset1 = cache_buffer->data_.offset(batchPos1);
  int offset2 = cache_buffer->data_.offset(batchPos2);
  //unsigned int last_loc = (image_count*(image_count*(data_count))*sizeof(Dtype));
  Dtype * data = cache_buffer->data_.mutable_cpu_data();
  Dtype * label = cache_buffer->label_.mutable_cpu_data();

  cache.seekg (image1_loc);
  for (int i = 0; i < data_count; ++i)
    cache >> data[i];

  cache.seekg (image2_loc);
  for (int i = 0; i < data_count; ++i)
    cache >> data[i+data_count];

  cache.seekg (last_loc+sizeof(Dtype)*batch1*image_count + sizeof(Dtype)*batchPos1);
  cache >> label[0];

  cache.seekg (last_loc+sizeof(Dtype)*batch2*image_count + sizeof(Dtype)*batchPos2);
  cache >> label[1];

  //Write stuff
  cache.seekg (image1_loc);
  for (int i = 0; i < data_count; ++i)
    cache << data[i+data_count];

  cache.seekg (image2_loc);
  for (int i = 0; i < data_count; ++i)
    cache << data[i];

  cache.seekg (last_loc+sizeof(Dtype)*batch1);
  cache << label[1];

  cache.seekg (last_loc+sizeof(Dtype)*batch2);
  cache << label[0];*/
}
template <typename Dtype>
void DiskCache<Dtype>::create( void * ptr
    , boost::shared_ptr<std::vector<boost::shared_ptr<bool> > >ptr2
    , bool thread_safe )
{
  Cache<Dtype>::prefetch = thread_safe;
  Cache<Dtype>::full_replace = false;
  Cache<Dtype>::dirty = ptr2;
  Cache<Dtype>::last_i = 0;
  Cache<Dtype>::slot = 0;
  for(int i=0; i< Cache<Dtype>::size; i++) {
    *(*Cache<Dtype>::dirty)[i] = true;
  }

  //if(thread_safe)
  //  Cache<Dtype>::sync_var = boost::make_shared<Cache<Dtype>::sync>();
  //else
  //  Cache<Dtype>::sync_var = NULL;
  open = false;
  cache_buffer = static_cast<Batch<Dtype> *> (ptr);
  cache_read_buffer = static_cast<Batch<Dtype> *> (ptr)+1;
  current_offset = 0;
  if( Cache<Dtype>::eviction_rate!=0)
  {
    LOG(INFO) << "Error: Disk Caching shuffle is not supported / auto turning off shuffling";
    Cache<Dtype>::eviction_rate=0;
  }
}

template <typename Dtype>
bool DiskCache<Dtype>::empty()
{
  return Cache<Dtype>::used == Cache<Dtype>::size;
}
template <typename Dtype>
PopBatch<Dtype> DiskCache<Dtype>::pop() {  //Cache<Dtype>::lock();
  DLOG(INFO) << "Attempting DiskCache Size(pop): +++++++++++++" << this->size;
  PopBatch<Dtype> pbatch;
  boost::lock_guard<boost::mutex> lck(this->mtx_);
  if(!cache_read) {
    DLOG(INFO) << "STREAM OBJECT INVALID@@@@@@@@@";
    cache_read.clear();
    cache.clear();
    open = false;
    if(!open) {
      char * disk_loc_char = new char [Cache<Dtype>::disk_location.length()+1];
      strcpy(disk_loc_char, Cache<Dtype>::disk_location.c_str());
      cache.open (disk_loc_char, ios::trunc| ios::in | ios::out | ios::binary );
      cache_read.open (disk_loc_char, ios::in | ios::out | ios::binary );
      open = true;
    }
  }

  // Cache<Dtype>::used.fetch_add(1, boost::memory_order_relaxed);
  int my_slot = Cache<Dtype>::slot++;
  if(Cache<Dtype>::slot == Cache<Dtype>::size)
  {
    // cache_read.seekg (0, ios::beg);
    Cache<Dtype>::slot = 0;
  }
  while(*(*Cache<Dtype>::dirty)[my_slot])
  {
    /*if(Cache<Dtype>::prev && this->prev->prefetch == Cache<Dtype>::prefetch)
    {
      if(Cache<Dtype>::prefetch)
      {
        (this->prev->*(this->prev->refill_policy))(1);
        (this->*(Cache<Dtype>::refill_policy))(1);
      }
      else
      {
        (this->prev->*(this->prev->local_refill_policy))(1);
        (this->*(Cache<Dtype>::refill_policy))(1);
      }
    }*/
    fill_pos(my_slot);

  }
  int image_count;
  long datum_size;
  char * bytes;
  char *img_count, *d_size;
  // long file_buffer_size = 0;
  // cache_read.seekg(0, ios::beg);
  // cache_read.read( reinterpret_cast<char*>(&image_count), sizeof(int));
  // cache_read.read( reinterpret_cast<char*>(&datum_size), sizeof(long));
  // file_buffer_size = sizeof(int)+sizeof(long) +
  //                     image_count*(datum_size+sizeof(Dtype));
  // cache_read.seekg(0, ios::beg);
  // cache_read.write( reinterpret_cast<char*>(&image_count), sizeof(int));
  // cache_read.write( reinterpret_cast<char*>(&datum_size), sizeof(long));
  cache_read.seekg(my_slot * this->file_buffer_size_, ios::beg);
  // cache_read.read( reinterpret_cast<char*>(&image_count), sizeof(int));
  // cache_read.read((char*)(img_count), sizeof(int));
  // image_count = *(reinterpret_cast<int*>(img_count));
  // cache_read.read( reinterpret_cast<char*>(&image_count), sizeof(int));
  // if(image_count > 5000 || image_count < 0){
  //   cache_read.clear();
  //   cache_read.seekg(my_slot * this->file_buffer_size_, ios::beg);
  //   cache_read.read( reinterpret_cast<char*>(&image_count), sizeof(int));
  // }
  // cache_read.read((char*)(img_count), sizeof(int));
  // image_count = reinterpret_cast<int*>(img_count);
  // cache_read.read( (char*)(&img_count), sizeof(int));
  // cache_read.read( reinterpret_cast<char*>(&datum_size), sizeof(long));
  // cache_read.read((char*)(d_size), sizeof(long));
  // datum_size = reinterpret_cast<long*>(d_size);
  DLOG(INFO) << "========================!";
  DLOG(INFO) << "DISK BATCH IMAGE Count: (saved)" << this->image_count;
  // DLOG(INFO) << "DISK BATCH DATUM SIZE: " << datum_size;
  // DLOG(INFO) << "DISK BATCH filebuf SIZE: "
  //    << sizeof(int)+sizeof(long) + (this->image_count)*(datum_size+sizeof(Dtype));
  DLOG(INFO) << "DISK BATCH filebuf SIZE(saved): " << this->file_buffer_size_;
  DLOG(INFO) << "DISK BATCH Current Slot No.: " << my_slot;
  // this->ref_data_shape_[0] = image_count;
  cache_read_buffer->data_.Reshape(this->ref_data_shape_);
  Dtype * data = cache_read_buffer->data_.mutable_cpu_data();
  cache_read_buffer->label_.Reshape(this->ref_label_shape_);
  Dtype * label = cache_read_buffer->label_.mutable_cpu_data();
  for (int i = 0; i < this->image_count; ++i)
  {
    int offset = cache_read_buffer->data_.offset(i);
    bytes = (char*) (data+offset);
    // bytes = reinterpret_cast<char*>(data+offset);
    cache_read.read( bytes, datum_size);
    bytes = (char*) (label+i);
    cache_read.read( bytes, sizeof(Dtype));
  }
  //current_offset++;
  //Cache<Dtype>::used++;

  pbatch.batch = cache_read_buffer;
  // pbatch.dirty = &Cache<Dtype>::dirty[my_slot];
  pbatch.dirty = (*Cache<Dtype>::dirty)[my_slot];

  return pbatch;
}

template <typename Dtype>
void DiskCache<Dtype>::fill(bool in_thread)
{
  boost::lock_guard<boost::mutex> lck(this->mtx_);
  DLOG(INFO) << "DiskCache Size(fill): ---------------" << this->size;
  if(!open)
  {
    LOG(INFO) << "Cache Location" << Cache<Dtype>::disk_location;
    char * disk_loc_char = new char [Cache<Dtype>::disk_location.length()+1];
    strcpy(disk_loc_char, Cache<Dtype>::disk_location.c_str());
    // cache.open (Cache<Dtype>::disk_location, ios::trunc| ios::in | ios::out | ios::binary );
    cache.open (disk_loc_char, ios::trunc| ios::in | ios::out | ios::binary );
    //cache.open (disk_loc_char, ios::trunc| ios::in | ios::out | ios::binary );
    // cache_read.open (Cache<Dtype>::disk_location, ios::in | ios::binary );
    cache_read.open (disk_loc_char, ios::in | ios::out | ios::binary );
    // cache_read.open (disk_loc_char, ios::in | ios::binary );
    open = true;
      if(!cache.is_open() || !cache_read.is_open())
      {
        LOG(INFO) << "Couldn't open disk cache location: " << Cache<Dtype>::disk_location;
        exit(1);
      }
    }

    int copy_qsize = Cache<Dtype>::data_layer->get_copy_qsize();
    int size_tocopy = 0;

    // if(copy_qsize > this->disk_cache_min_size) {
    if(copy_qsize > 50 || copy_qsize < 20) {
      size_tocopy = copy_qsize;// 2 * this->disk_cache_min_size;
    }
    else // if(copy_qsize > 20)
      size_tocopy = 20;

    // LOG(INFO) << "Size to Copy: +++++++++ " << size_tocopy;

    // for(int qcount = 0; (qcount < 20) && (qcount < copy_qsize); ++qcount ) {
    for(int qcount = 0; qcount < size_tocopy; ++qcount ) {

      if (Cache<Dtype>::size < this->disk_cache_max_size) {
        fill_pos(Cache<Dtype>::size);

        (*Cache<Dtype>::dirty).push_back(boost::make_shared<bool>(false));
        Cache<Dtype>::size.fetch_add(1, boost::memory_order_release);
      }
      else {
        // for (int j = Cache<Dtype>::last_i; j < Cache<Dtype>::size; ++j) {
        for (int j = Cache<Dtype>::slot; j < Cache<Dtype>::size; ++j) {

          if(*(*Cache<Dtype>::dirty)[j] == true)
          {
          // Cache<Dtype>::data_layer->load_batch(cache_buffer);
            fill_pos(j);
            // Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
            *(*Cache<Dtype>::dirty)[j] = false;
          }
          else
            break;
        }

        for (int j = 0; j < Cache<Dtype>::size; ++j) {
          if(*(*Cache<Dtype>::dirty)[j] == true)
          {
            fill_pos(j);
            *(*Cache<Dtype>::dirty)[j] = false;
            break;
          }
        }
    }
    //delete cache_buffer; // free the copied memory;
  }
  // if (Cache<Dtype>::last_i == Cache<Dtype>::size) {
  if (Cache<Dtype>::slot == Cache<Dtype>::size) {
    Cache<Dtype>::full_replace = true;
    // Cache<Dtype>::last_i = 0;
    Cache<Dtype>::slot = 0;
  }
}

template <typename Dtype>
void DiskCache<Dtype>::fill_pos(int pos) {
  char * bytes;

  // boost::lock_guard<boost::mutex> lck(this->mtx_);
  shared_ptr<Batch<Dtype> > cache_buffer_tmp;
  Cache<Dtype>::data_layer->copy_batch(cache_buffer_tmp);
  Dtype * data = cache_buffer_tmp->data_.mutable_cpu_data();
  Dtype * label = cache_buffer_tmp->label_.mutable_cpu_data();

  if(ref_data_shape_.size() == 0) {
    ref_data_shape_.push_back(cache_buffer_tmp->data_.shape(0));
    ref_data_shape_.push_back(cache_buffer_tmp->data_.shape(1));
    ref_data_shape_.push_back(cache_buffer_tmp->data_.shape(2));
    ref_data_shape_.push_back(cache_buffer_tmp->data_.shape(3));
  }
  // = cache_buffer_tmp->data_.shape(); }
  int image_count = cache_buffer_tmp->data_.shape(0);
  long datum_size = cache_buffer_tmp->data_.shape(1);
  datum_size *= cache_buffer_tmp->data_.shape(2);
  datum_size *= cache_buffer_tmp->data_.shape(3);
  datum_size *= sizeof(Dtype);
  DLOG(INFO) << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&";
  DLOG(INFO) << "DISK CACHE FILL:Image count: " << image_count;
  DLOG(INFO) << "DISK CACHE FILL:Datum size: " << datum_size;
  if(!this->file_buffer_size_)
    this->file_buffer_size_ = sizeof(int) +sizeof(long)+ image_count * (datum_size + sizeof(Dtype));
  if(!this->image_count)
    this->image_count = image_count;
  DLOG(INFO) << "DISK CACHE FILL:filebuf size: "
             << sizeof(int) +sizeof(long)+ image_count * (datum_size + sizeof(Dtype));
  DLOG(INFO) << "DISK CACHE FILL:filebuf size(saved): " << this->file_buffer_size_;
  DLOG(INFO) << "DISK CACHE FILL:slot pos : " << pos;
             // for label also.
             // << sizeof(int) +sizeof(long)+ image_count * (datum_size);

  // Seek last element
  // int last_element = pos * (sizeof(int) +sizeof(long) + image_count * (
  //                     datum_size + sizeof(Dtype)));
  int last_element = pos * (image_count * ( datum_size + sizeof(Dtype)));
  // cache.seekg(
  //    pos *
  //    (2 * sizeof(int) + image_count * (datum_size + sizeof(Dtype)), ios::beg));
  cache.seekg(last_element, ios::beg);
  // cache.write(reinterpret_cast<char *>(&image_count), sizeof(int));
  // cache.write(reinterpret_cast<char *>(&image_count), sizeof(int));
  // cache.write(reinterpret_cast<char *>(&datum_size), sizeof(long));
  // cache.write(reinterpret_cast<char *>(&datum_size), sizeof(long));
  // writing each image and its label
  for (int i = 0; i < image_count; ++i) {
    int offset = cache_buffer_tmp->data_.offset(i);
    bytes = (char *)(data + offset);
    cache.write(bytes, datum_size);
    bytes = (char *)(label + i);
    cache.write(bytes, sizeof(Dtype));
  }
  cache.flush();
  /*char * disk_loc_char = new char [Cache<Dtype>::disk_location.length()+1];
  strcpy(disk_loc_char, Cache<Dtype>::disk_location.c_str());
  fstream tmp;
  tmp.open(disk_loc_char, ios::out | ios::in | ios::binary);

  tmp.seekg(last_element, ios::beg);
  tmp.read(reinterpret_cast<char*>(&image_count), sizeof(int));
  DLOG(INFO) << "<<<<<<<<<<<<<<<<<<<<<<<<<<";
  DLOG(INFO) << "Written Image Count Size: " << image_count;
  DLOG(INFO) << "DISK CACHE FILL:slot pos : " << pos;
  tmp.seekg(last_element, ios::beg);
  tmp.write(reinterpret_cast<char*>(&image_count), sizeof(int));
  tmp.flush();
  */
  // delete cache_buffer;
}

template <typename Dtype>
void DiskCache<Dtype>::shuffle() {
  //Cache<Dtype>::lock();
  cache.seekg (0, ios::beg);
  int image_count;
  int datum_size;
  cache.read( (char *)&image_count, sizeof(int));
  cache.read( (char *)&datum_size, sizeof(int));
  for(int i=0; i< Cache<Dtype>::size; i++)
  {
    for(int j=0; j< cache_buffer->data_.shape(0); j++)
    {
      shuffle_cache(i, j, Cache<Dtype>::data_layer->randomGen(this->size), Cache<Dtype>::data_layer->randomGen(image_count), image_count, datum_size, 1);
    }
  }
  cache.seekg (0, ios::beg);
  current_offset = 0;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void DiskCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
  //Cache<Dtype>::lock();
  PopBatch<Dtype> pbatch;
  Dtype * data = cache_buffer->data_.mutable_cpu_data();
  Dtype * label = cache_buffer->label_.mutable_cpu_data();
  current_offset=0;
  //cache.seekg (0, ios::beg);
  char * bytes;
  for (int j = Cache<Dtype>::last_i; j < Cache<Dtype>::size; ++j) {
    Cache<Dtype>::last_i=j;
    if(*(*Cache<Dtype>::dirty)[j] == true)
    {
      pbatch = next_cache->pop();
      data = pbatch.batch->data_.mutable_cpu_data();
      label = pbatch.batch->label_.mutable_cpu_data();
      //cache_buffer->data_.CopyFrom( batch->data_ );
      //cache_buffer->label_.CopyFrom( batch->label_ );

      int image_count = pbatch.batch->data_.shape(0);
      int datum_size = pbatch.batch->data_.shape(1);
      datum_size *= pbatch.batch->data_.shape(2);
      datum_size *= pbatch.batch->data_.shape(3);
      datum_size *= sizeof(Dtype);

      cache.write( (char *)&image_count, sizeof(int));
      cache.write( (char *)&datum_size, sizeof(int));
      for (int i = 0; i < image_count; ++i)
      {
        int offset = pbatch.batch->data_.offset(i);
        bytes = (char*) (data+offset);
        cache.write( bytes, datum_size);
        bytes = (char*) (label+i);
        cache.write( bytes, sizeof(Dtype));
      }
      *pbatch.dirty = boost::make_shared<bool>(true);
      pbatch.batch->dirty = true;
      // Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      *(*Cache<Dtype>::dirty)[j] = false;
      Cache<Dtype>::last_i++;
    }
    else
      break;
  }
  if(Cache<Dtype>::last_i == Cache<Dtype>::size)
  {
    //bounds=0;
    Cache<Dtype>::full_replace = true;
    Cache<Dtype>::last_i=0;
  }
  //cache.seekg (0, ios::beg);
  //Cache<Dtype>::used=0;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void DiskCache<Dtype>::reshape(vector<int> * top_shape, vector<int> * label_shape)
{
  //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->data_.Reshape(*top_shape);
      cache_read_buffer->data_.Reshape(*top_shape);
  //}
  if (label_shape) {
    //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->label_.Reshape(*label_shape);
      cache_read_buffer->label_.Reshape(*label_shape);
    //}
  }
}
template <typename Dtype>
void DiskCache<Dtype>::mutate_data(bool labels, const int level)
{
  //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->data_.mutable_cpu_data();
      cache_read_buffer->data_.mutable_cpu_data();
  //}
  if (labels) {
    //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->label_.mutable_cpu_data();
      cache_read_buffer->label_.mutable_cpu_data();
    //}
  }
  /*
#ifndef CPU_ONLY
 if (Caffe::mode() == Caffe::GPU) {
    cache_buffer->data_.mutable_gpu_data();
    cache_read_buffer->data_.mutable_gpu_data();
    if (labels) {
      cache_buffer->label_.mutable_gpu_data();
      cache_read_buffer->label_.mutable_gpu_data();
    }
    CUDA_CHECK(cudaEventCreate(&cache_buffer->copied_));
    CUDA_CHECK(cudaEventCreate(&cache_read_buffer->copied_));
  }
#endif
  */
}
INSTANTIATE_CLASS(Cache);
INSTANTIATE_CLASS(MemoryCache);
INSTANTIATE_CLASS(DiskCache);
}
