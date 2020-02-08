#include "EnvTest.hpp"
// TODO replace with raising Python AssertionError
#include <cassert>


using namespace Test;

void EnvTest::EnvInit() {
    Py_Initialize();
    np::initialize();
}

EnvTest::BufferSize EnvTest::CheckAndGetSize_py(np::ndarray &buffer_main) {
    // Infer samples and channels from array shape.
    unsigned int n_channels;
    unsigned int n_samples;

    switch(buffer_main.get_nd()) {
        case 0: {
            // TODO raise exception
            return { 0, 0 };
        }
        case 1: {
            n_samples = buffer_main.shape(0);
            n_channels = 1;
            break;
        }
        case 2: {
            // Presume interleaved format (innermost dimension is frame size)
            n_samples = buffer_main.shape(0);
            n_channels = buffer_main.shape(1);
            break;
        }
        default: {
            // TODO raise exception
            return { 0, 0 };
        }
    }

    return BufferSize{ n_channels, n_samples };
}

EnvTest::BufferSize EnvTest::CheckAndGetSize_py(np::ndarray &buffer_main,
        np::ndarray &buffer_other) {
    assert(buffer_main.shape(0) == buffer_other.shape(0));
    if (buffer_main.get_nd() > 1) {
        assert(buffer_main.shape(1) == buffer_other.shape(1));
    }

    return CheckAndGetSize_py(buffer_main);
}

template<typename ... Args>
EnvTest::BufferSize EnvTest::CheckAndGetSize_py(np::ndarray &buffer_main,
        np::ndarray &buffer_other, Args ... args) {
    assert(buffer_main.shape(0) == buffer_other.shape(0));
    if (buffer_main.get_nd() > 1) {
        assert(buffer_main.shape(1) == buffer_other.shape(1));
    }

    return CheckAndGetSize_py(buffer_main, args...);
}

int EnvTest::TestPrint() {
    std::cout << "Test succeeded!" << std::endl;
    return 0;
}

void EnvTest::TestSum(float *buffer1,
            float *buffer2,
            float *buffer_out,
            unsigned int n_channels,
            unsigned int n_samples) {

    // 1-D sum over channels * samples
    int n = n_channels * n_samples;
    while (--n >= 0) {
        buffer_out[n] = buffer1[n] + buffer2[n];
    }
}

void EnvTest::TestSum_py(np::ndarray buffer1,
        np::ndarray buffer2,
        np::ndarray buffer_out) {

    BufferSize size = CheckAndGetSize_py(buffer1, buffer2, buffer_out);

    TestSum(reinterpret_cast<float *>(buffer1.get_data()),
            reinterpret_cast<float *>(buffer2.get_data()),
            reinterpret_cast<float *>(buffer_out.get_data()),
            size.n_channels,
            size.n_samples);
}

void EnvTest::TestGain(float *buffer1,
            float *buffer_out,
            unsigned int n_channels,
            unsigned int n_samples) {

    // 1-D scaling over channels * samples
    int n = n_channels * n_samples;
    while (--n >= 0) {
        buffer_out[n] = buffer1[n] * gain_;
    }
}

void EnvTest::TestGain_py(np::ndarray buffer1,
        np::ndarray buffer_out) {

    BufferSize size = CheckAndGetSize_py(buffer1, buffer_out);

    TestGain(reinterpret_cast<float *>(buffer1.get_data()),
            reinterpret_cast<float *>(buffer_out.get_data()),
            size.n_channels,
            size.n_samples);
}

BOOST_PYTHON_MODULE(EnvTest)
{
    using namespace boost::python;

    class_<EnvTest>("EnvTest", init<float>())
        .def("TestPrint", &EnvTest::TestPrint)
        .def("TestSum", &EnvTest::TestSum_py)
        .def("TestGain", &EnvTest::TestGain_py);
}
