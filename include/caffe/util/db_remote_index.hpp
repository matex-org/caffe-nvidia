/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_REMOTE_INDEX
#ifndef CAFFE_UTIL_DB_REMOTE_INDEX_HPP
#define CAFFE_UTIL_DB_REMOTE_INDEX_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

class RemoteIndexEnv
{
  public: 
  
  RemoteIndexEnv()
  {
      buffer_size = 1024;
      buffer = new char[buffer_size];
      current_index = 0;
      index_position = 0;
      block_position = 0;
      valid_=false;
  }

  void get_next(string& key, string& value)
  {
    if(current_index >= image_index.size())
    {
      valid_=false;
      return;
    }
    //value.resize(image_index[current_index]);
    if(image_index[current_index]>buffer_size)
    {
      delete[] buffer;
      buffer_size = image_index[current_index];
      buffer = new char[buffer_size];
    }

    block_stream_.read( buffer, image_index[current_index]);
    value = string(buffer, image_index[current_index]);
    key = key_index[current_index++];//std::to_string(current_index++);
  }

  void put(string& key, string& value)
  {
    index_stream_ << key << ":" << value.size() << ",";
    //block_stream_.write( value.c_str(), value.size());
    block_stream_ << value; //.write( value.data(), value.size());
  }

  int open(const string &source,  std::ios_base::openmode ios_mode, int mode)
  {
    string index = source + "/index";
    string block = source + "/db";
    string key;
    index_stream_.open(index, ios_mode);
    block_stream_.open(block, ios_mode);
    uint64_t value;
    valid_=true;
    if(ios_mode & ios::in)
    {
      //while(!index_stream_.eof())
      while(1)
      {
        //index_stream.ignore(numeric_limits<streamsize>::max(),':');
        std::getline(index_stream_,key,':');
        if(index_stream_.eof())
          return 0;
        //index_stream_ >> key;
        key_index.push_back(key);
        //index_stream_.get();
        index_stream_ >> value;
        image_index.push_back(value);
        index_stream_.get();
      }
      //f
      
    }
    return 0;
  }
  void close()
  {
     index_stream_.close();
     block_stream_.close();
  }
  /*void seek(string& key, string& value)
  {
    current_index = std::stoull(key);
    value.resize(image_index[current_index]);
    block_stream_.seek();
    block_stream_.read( &value, image_index[current_index]);
    key = std::to_string(current_index++);
    if(current_index > image_index.size())
    {
      current_index = 0;
    }
    
  }*/
  
  bool valid(){ return valid_; };

  void reset() { valid_=true; index_stream_.seekg(0,index_stream_.beg); block_stream_.seekg(0,block_stream_.beg); current_index =0;  }

  private:
  // Master List of image sizes
  vector<uint64_t> image_index;
  vector<string> key_index;
  bool valid_;
  char * buffer;
  unsigned int buffer_size;
  fstream index_stream_;
  fstream block_stream_;
  unsigned int current_index;
  uint64_t index_position;
  uint64_t block_position;
};

class RemoteIndexCursor : public Cursor {
 public:
  explicit RemoteIndexCursor(RemoteIndexEnv * env)
    : env_(env), valid_(false) {
    SeekToFirst();
  }
  virtual ~RemoteIndexCursor() {
    env_->close();
  }
  virtual void SeekToFirst() { env_->reset(); env_->get_next(key_, value_); }
  virtual void Next() { env_->get_next(key_, value_); }
  //virtual void SeekToImage() { env->seek(key_, value_); }
  //virtual void EntryCount() { return env->EntryCount(); }
  virtual string key() {
    return key_;
  }
  virtual string value() {
    return value_;
  }
  virtual std::pair<void*, size_t> valuePointer() {
    return std::make_pair((void*)value_.c_str(), value_.size());
  }

  virtual bool valid() { return env_->valid(); }

 private:
  RemoteIndexEnv * env_;
  string key_, value_;
  bool valid_;
};

class RemoteIndexTransaction : public Transaction {
 public:
  explicit RemoteIndexTransaction(RemoteIndexEnv * env):
  env_(env)
  { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  vector<string> keys, values;
  RemoteIndexEnv * env_;

  DISABLE_COPY_AND_ASSIGN(RemoteIndexTransaction);
};

class RemoteIndex : public DB {
 public:
  RemoteIndex() { }
  virtual ~RemoteIndex() { Close(); }
  virtual void Open(const string& source, Mode mode, const LayerParameter * param);
  virtual void Close() { env_->close();
  }
  virtual RemoteIndexCursor* NewCursor();
  virtual RemoteIndexTransaction* NewTransaction();

 private:
  RemoteIndexEnv * env_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
