#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

namespace Test {

namespace p = boost::python;
namespace np = boost::python::numpy;

class EnvTest {

 public:
    
    EnvTest() :
            gain_(1.f) { EnvInit(); }

    EnvTest(float gain) :
            gain_(gain) { EnvInit(); }

    static void EnvInit();

    static int TestPrint();

    static void TestSum(float *buffer1,
            float *buffer2,
            float *buffer_out,
            unsigned int n_channels,
            unsigned int n_samples);

    static void TestSum_py(np::ndarray buffer1,
            np::ndarray buffer2,
            np::ndarray buffer_out);

    void TestGain(float *buffer1,
            float *buffer_out,
            unsigned int n_channels,
            unsigned int n_samples);

    void TestGain_py(np::ndarray buffer1,
            np::ndarray buffer_out);
 protected:

    struct BufferSize {
        unsigned int n_channels;
        unsigned int n_samples;
    };
    static BufferSize CheckAndGetSize_py(np::ndarray &buffer_main);
    static BufferSize CheckAndGetSize_py(np::ndarray &buffer_main, np::ndarray &buffer_other);
    template<typename ... Args>
    static BufferSize CheckAndGetSize_py(np::ndarray &buffer_main, np::ndarray &buffer_other,
            Args ... args);
    float gain_;
};

}
