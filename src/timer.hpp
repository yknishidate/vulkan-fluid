#include <chrono>

struct Timer
{
    void start()
    {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double elapsed()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        return elapsed;
    }

    std::chrono::steady_clock::time_point startTime;
};
