#pragma once

#include <cstddef>

// Allocates from CUDA's memory pool, which reuses freed buffers
void* gpu_alloc(size_t bytes);
void gpu_free(void* ptr) noexcept;

// Returns all unused buffers in the pool to CUDA
void gpu_free_unused();

// GPU buffer of n elements, freed at the end of the scope
template <typename T>
struct Buffer {
    T* ptr;

    explicit Buffer(size_t n)
        : ptr(static_cast<T*>(gpu_alloc(n * sizeof(T)))) {}
    // Takes ownership of a buffer from gpu_alloc
    explicit Buffer(T* ptr) : ptr(ptr) {}
    ~Buffer() { gpu_free(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // The caller must gpu_free the returned pointer
    T* release() {
        T* released = ptr;
        ptr = nullptr;
        return released;
    }
};
