#include <iostream>
using namespace std;

#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

void initialize(thrust::device_vector<int>& v)
{
    thrust::default_random_engine rng(123456);
    thrust::uniform_int_distribution<int> dist(10, 99);
    for(size_t i = 0; i < v.size(); i++)
      v[i] = dist(rng);
}

void print(const thrust::device_vector<int>& v)
{
    for(size_t i = 0; i < v.size(); i++)
        cout << " " << v[i];
    cout << "\n";
}

int main()
{
    size_t N = 16;
    cout << "sorting integers\n";
    thrust::device_vector<int> keys(N);
    initialize(keys);
    print(keys);
    thrust::sort(keys.begin(), keys.end());
    print(keys);
    
 }
