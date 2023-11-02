import os
import pickle
import json
import hashlib

class DiskCache:
    """
    A convenient disk cache that stores key-value pairs on disk.
    Useful for querying LLM API.
    """
    def __init__(self, cache_dir='cache', load_cache=True):
        self.cache_dir = cache_dir
        self.data = {}
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            if load_cache:
                self._load_cache()

    def _generate_filename(self, key):
        key_str = json.dumps(key)
        key_hash = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        return f"{key_hash}.pkl"

    def _load_cache(self):
        for filename in os.listdir(self.cache_dir):
            with open(os.path.join(self.cache_dir, filename), 'rb') as file:
                key, value = pickle.load(file)
                self.data[json.dumps(key)] = value

    def _save_to_disk(self, key, value):
        filename = self._generate_filename(key)
        with open(os.path.join(self.cache_dir, filename), 'wb') as file:
            pickle.dump((key, value), file)

    def __setitem__(self, key, value):
        str_key = json.dumps(key)
        self.data[str_key] = value
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        str_key = json.dumps(key)
        return self.data[str_key]
    
    def __contains__(self, key):
        str_key = json.dumps(key)
        return str_key in self.data

    def __repr__(self):
        return repr(self.data)


if __name__ == '__main__':
    cache = DiskCache()
    cache_key = {'id': 1, 'name': 'John'}
    cache[cache_key] = 'value1'

    print(cache)  # {"{'id': 1, 'name': 'John'}": 'value1'}

    # When you restart your program, the cache will load the previous key-value pairs
    cache2 = DiskCache()
    print(cache2)  # {"{'id': 1, 'name': 'John'}": 'value1'}


