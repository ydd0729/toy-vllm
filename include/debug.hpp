#pragma once
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#ifdef DEBUG

template <typename T>
inline void dumpVector(const std::vector<T>& v, const char* name, size_t max_count = SIZE_MAX)
{
    std::cerr << "== " << name << " (size=" << v.size() << ") == [ ";
    size_t n = std::min(v.size(), max_count);
    for (size_t i = 0; i < n; ++i)
    {
        std::cerr << v[i] << ' ';
    }
    if (n < v.size())
    {
        std::cerr << "... ";
    }
    std::cerr << "]" << std::endl;
}


#define DUMP_VEC(v) dumpVector((v), #v)
#define DUMP_VEC_N(v, n) dumpVector((v), #v, (n))
#else
#define DUMP_VEC(v) ((void) 0)
#define DUMP_VEC_N(v, n) ((void) 0)
#endif
