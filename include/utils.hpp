#pragma once

#define UNREACHABLE(msg)                                                                           \
    do                                                                                             \
    {                                                                                              \
        std::cerr << "Unreachable: " << (msg) << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
        std::abort();                                                                              \
    } while (0)
