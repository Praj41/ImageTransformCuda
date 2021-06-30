#pragma once

#include <queue>
#include <string>

namespace praj {
    class fileQueue {
        std::queue<std::string> _queue;
    public:
        bool empty();
        void push(std::string &&str);
        std::string pop();
    };
}
