#include <chrono>

struct Timer
{
    void start()
    {
        startTime = std::chrono::steady_clock::now();
    }

    long long elapsed()
    {
        auto endTime = std::chrono::steady_clock::now();
        long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        return elapsed;
    }

    std::chrono::steady_clock::time_point startTime;
};
