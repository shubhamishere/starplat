#include<atomic>
#include<algorithm>

template<typename T>
inline void atomicMin(T* targetVar, T update_val)
{ 
    T oldVal, newVal;
  do {
       oldVal = *targetVar;
       newVal = std::min(oldVal,update_val);
       if (oldVal == newVal) 
            break;
     } while ( __sync_val_compare_and_swap(targetVar, oldVal, newVal) == false);
   

}
template<typename T>
inline void atomicAdd(T* targetVar, T update_val) {

    if (update_val == 0) return;

    T  oldValue, newValue;
    do {
        oldValue = *targetVar;
        newValue = oldValue + update_val;
    } while (!__sync_bool_compare_and_swap(targetVar,oldValue,newValue));
}

// atomicAdd for double types is not supported on all CUDA architectures. 
// Specifically, atomicAdd for double was introduced in CUDA 8.0 for devices with compute capability 6.0 (Pascal) and later. 
// If you're compiling for an older architecture (e.g., sm_60), you might run into issue and get compile time errors.
// hence implemented our own custom atomicAdd for double types using atomicCAS (compare and swap).

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}