#include "SafeQ.cuh"

void praj::fileQueue::push(std::string &&str) {
    _queue.emplace(str);
}

std::string praj::fileQueue::pop() {
    //std::mutex k;
    //std::lock_guard<std::mutex> p(k);
    std::string str(_queue.front());
    _queue.pop();
    return str;
}

bool praj::fileQueue::empty() {
    return _queue.empty();
}
