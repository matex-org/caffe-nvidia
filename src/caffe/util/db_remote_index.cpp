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
#include "caffe/util/db_remote_index.hpp"

#include <sys/stat.h>
#include <fstream>

#include <string>

namespace caffe { namespace db {

void RemoteIndex::Open(const string& source, Mode mode, const LayerParameter * param) {
  env_ = new RemoteIndexEnv();
  if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << " failed";
    
  }

  std::ios_base::openmode ios_mode; //ios::binary;
  if (mode == READ) {
    ios_mode = ios::in;
  }
  else
    ios_mode = ios::out | ios::trunc;
  int rc = env_->open(source, ios_mode, 0664);
  /*if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << " failed";
  }
  ios_base::openmode ios_mode = ios::binary;
  if (mode == READ) {
    ios_mode |= ios::in;
  }
  else
    ios_mode |= ios::out | ios::trunc;
  
  ios_stream.open(source, ios_mode);*/

  LOG(INFO) << "RemoteIndex " << source;
}

RemoteIndexCursor* RemoteIndex::NewCursor() {
  return new RemoteIndexCursor(env_);
}

RemoteIndexTransaction* RemoteIndex::NewTransaction() {
  return new RemoteIndexTransaction(env_);
}

void RemoteIndexTransaction::Put(const string& key, const string& value) {
  keys.push_back(key);
  values.push_back(value);
}

void RemoteIndexTransaction::Commit() {

  for (int i = 0; i < keys.size(); i++) {
    //mdb_key.mv_size = keys[i].size();
    //mdb_key.mv_data = const_cast<char*>(keys[i].data());
    //mdb_data.mv_size = values[i].size();
    //mdb_data.mv_data = const_cast<char*>(values[i].data());

    // Add data to the transaction
    env_->put(keys[i], values[i]);
    /*if (put_rc == MDB_MAP_FULL) {
      // Out of memory - double the map size and retry
      mdb_txn_abort(mdb_txn);
      mdb_dbi_close(mdb_env_, mdb_dbi);
      DoubleMapSize();
      Commit();
      return;
    }*/
    // May have failed for some other reason
    //MDB_CHECK(put_rc);
  }

  // Commit the transaction
  /*int commit_rc = mdb_txn_commit(mdb_txn);
  if (commit_rc == MDB_MAP_FULL) {
    // Out of memory - double the map size and retry
    mdb_dbi_close(mdb_env_, mdb_dbi);
    DoubleMapSize();
    Commit();
    return;
  }*/
  // May have failed for some other reason
  //MDB_CHECK(commit_rc);

  // Cleanup after successful commit
  //mdb_dbi_close(mdb_env_, mdb_dbi);
  keys.clear();
  values.clear();
}

}  // namespace db
}  // namespace caffe
#endif  // USE_LMDB
