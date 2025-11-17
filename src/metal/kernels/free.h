#pragma once

#include "metal_impl.h"

static void releaseBuffer(void* ptr) noexcept {
    auto b = static_cast<MTL::Buffer*>(const_cast<void*>(ptr));
    if (b) {
        b->release();  // release the Metal buffer
    }
}
