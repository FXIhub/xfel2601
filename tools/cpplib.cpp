#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace py = pybind11;

void correct_array(
        py::array_t<float> &outData, // dimensions: (y, x, cell id)
        py::array_t<bool> &badpixMask, // dimensions: (y, x, cell id)
        const py::array_t<float> &gainData, // dimensions: (y, x, cell id)
        const py::array_t<int> &gainLevelData, // dimensions: (gain level, y, x, cell id) 
        const py::array_t<int> &darkOffset, // dimensions: (gain level, y, x, cell id) 
        const py::array_t<float> &relativeGain,  // dimensions: (gain level, y, x, cell id) 
        const py::array_t<std::uint8_t> &badpixData  // dimensions: (gain level, y, x, cell id) 
        ){
    unsigned int Y = gainData.shape(0);
    unsigned int X = gainData.shape(1);
    unsigned int cellNum = gainData.shape(2);

    //py::array_t<double> result = py::array_t<double>({Y, X});
    auto outData_ptr = outData.mutable_unchecked<3>();
    auto badpixMask_ptr = badpixMask.mutable_unchecked<3>();
    auto gainData_ptr = gainData.unchecked<3>();
    auto gainLevelData_ptr = gainLevelData.unchecked<4>();
    auto darkOffset_ptr = darkOffset.unchecked<4>();
    auto relativeGain_ptr = relativeGain.unchecked<4>();
    auto badpixData_ptr = badpixData.unchecked<4>();

    for(unsigned int cellId = 0; cellId < cellNum; cellId++) {
        for (unsigned int x = 0; x < X; ++x) {
            for (unsigned int y = 0; y < Y; ++y) {
                // find the gain level `g` for pixel (x, y) at cell cellId
                unsigned int g=0;
                auto gd = gainData_ptr(y, x, cellId);
                if (gd > gainLevelData_ptr(2, y, x, cellId)) {
                    g = 2;
                } else {
                    if (gd > gainLevelData_ptr(1, y, x, cellId)) 
                        g = 1;
                }  
                //outData_ptr(y, x, cellId) = 0;
                outData_ptr(y, x, cellId) = 
                    (outData_ptr(y, x, cellId) - darkOffset_ptr(g, y, x, cellId)) * relativeGain_ptr(g, y, x, cellId); 
                badpixMask_ptr(y, x, cellId) = badpixData_ptr(g, y, x, cellId) == 0;
            }
        }
    }
}

template<typename T>
inline T norm2D(T x1, T y1, T x2, T y2)
{
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> radial_avg(
        float cenX, float cenY, // central: (x,y)
        const py::array_t<bool> mask, // dimensions: (y, x, hit id)
        const py::array_t<float> hits, // dimensions: (y, x, hit id)
        float bin_size
        )
{
    unsigned int Y = hits.shape(0);
    unsigned int X = hits.shape(1);
    unsigned int hitNum = hits.shape(2);
    auto maskPtr = mask.unchecked<3>();
    auto hitsPtr = hits.unchecked<3>();
    std::vector<std::vector<unsigned int>> radialMap(Y, std::vector<unsigned int>(X));
    unsigned int minRmap = 0xffffffff, maxRmap=0;
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            unsigned int rmap = norm2D<float>(x, y, cenX, cenY) / bin_size;
            radialMap[y][x] = rmap;
            maxRmap = std::max(maxRmap, rmap);
            minRmap = std::min(minRmap, rmap);
            //if (rmap > max_rmap) max_rmap = rmap;
        }
    }
    // std::cerr<<"max: " << maxRmap << " min: " << minRmap << std::endl;

    unsigned int len = maxRmap - minRmap + 1;
    std::vector<unsigned int> countsRaw(len, 0);
    std::vector<unsigned int> countsDiff(len);
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            radialMap[y][x] -= minRmap;
            countsRaw[radialMap[y][x]]++;
        }
    }

    py::array_t<float> outRad({len});
    py::array_t<float> outData({hitNum, len});
    py::array_t<unsigned int> outCount({hitNum, len});
    auto outDataPtr = outData.mutable_unchecked<2>();
    auto outCountPtr = outCount.mutable_unchecked<2>();
    auto outRadPtr = outRad.mutable_unchecked<1>();
    for(unsigned int i = 0; i < len; i++) {
        outRadPtr(i) = (i+minRmap) * bin_size;
    }
    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for (unsigned int i = 0; i < len; ++i) {
            outDataPtr(hitId, i) = 0;
        }
    }

    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for(unsigned int i=0; i<len; i++) {
            outCountPtr(hitId, i) = countsRaw[i];
        }
        for (unsigned int y = 0; y < Y; ++y) {
            for (unsigned int x = 0; x < X; ++x) {
                unsigned int rmap = radialMap[y][x];
                if (maskPtr(y, x, hitId)) {
                    outDataPtr(hitId, rmap) += hitsPtr(y, x, hitId);
                } else {
                    outCountPtr(hitId, rmap)--;
                }
            }
        }
    }

    return std::make_tuple(outData, outCount, outRad);
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> angular_avg(
        float cenX, float cenY, // central: (x,y)
        const py::array_t<bool> mask, // dimensions: (y, x, hit id)
        const py::array_t<float> hits, // dimensions: (y, x, hit id)
        float bin_size
        )
{
    unsigned int Y = hits.shape(0);
    unsigned int X = hits.shape(1);
    unsigned int hitNum = hits.shape(2);
    auto maskPtr = mask.unchecked<3>();
    auto hitsPtr = hits.unchecked<3>();
    std::vector<std::vector<int>> angularMap(Y, std::vector<int>(X));
    int minRmap = 100000, maxRmap= -100000;
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            int rmap = std::floor(std::atan((double)(y - cenY) / (double)(x - cenX)) / bin_size);
            angularMap[y][x] = rmap;
            maxRmap = std::max(maxRmap, rmap);
            minRmap = std::min(minRmap, rmap);
            //if (rmap > max_rmap) max_rmap = rmap;
        }
    }
    std::cerr<<"max: " << maxRmap << " min: " << minRmap << std::endl;

    unsigned int len = maxRmap - minRmap + 1;
    std::vector<unsigned int> countsRaw(len, 0);
    std::vector<unsigned int> countsDiff(len);
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            angularMap[y][x] -= minRmap;
            countsRaw[angularMap[y][x]]++;
        }
    }

    py::array_t<float> outRad({len});
    py::array_t<float> outData({hitNum, len});
    py::array_t<unsigned int> outCount({hitNum, len});
    auto outDataPtr = outData.mutable_unchecked<2>();
    auto outCountPtr = outCount.mutable_unchecked<2>();
    auto outRadPtr = outRad.mutable_unchecked<1>();
    for(unsigned int i = 0; i < len; i++) {
        outRadPtr(i) = (i+minRmap) * bin_size;
    }
    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for (unsigned int i = 0; i < len; ++i) {
            outDataPtr(hitId, i) = 0;
        }
    }

    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for(unsigned int i=0; i<len; i++) {
            outCountPtr(hitId, i) = countsRaw[i];
        }
        for (unsigned int y = 0; y < Y; ++y) {
            for (unsigned int x = 0; x < X; ++x) {
                unsigned int rmap = angularMap[y][x];
                if (maskPtr(y, x, hitId)) {
                    outDataPtr(hitId, rmap) += hitsPtr(y, x, hitId);
                } else {
                    outCountPtr(hitId, rmap)--;
                }
            }
        }
    }

    return std::make_tuple(outData, outCount, outRad);
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> radial_m2(
        float cenX, float cenY, // central: (x,y)
        const py::array_t<bool> mask, // dimensions: (y, x, hit id)
        const py::array_t<float> hits, // dimensions: (y, x, hit id)
        float bin_size
        )
{
    unsigned int Y = hits.shape(0);
    unsigned int X = hits.shape(1);
    unsigned int hitNum = hits.shape(2);
    auto maskPtr = mask.unchecked<3>();
    auto hitsPtr = hits.unchecked<3>();
    std::vector<std::vector<unsigned int>> radialMap(Y, std::vector<unsigned int>(X));
    unsigned int minRmap = 0xffffffff, maxRmap=0;
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            unsigned int rmap = norm2D<float>(x, y, cenX, cenY) / bin_size;
            radialMap[y][x] = rmap;
            maxRmap = std::max(maxRmap, rmap);
            minRmap = std::min(minRmap, rmap);
            //if (rmap > max_rmap) max_rmap = rmap;
        }
    }
    // std::cerr<<"max: " << maxRmap << " min: " << minRmap << std::endl;

    unsigned int len = maxRmap - minRmap + 1;
    std::vector<unsigned int> countsRaw(len, 0);
    std::vector<unsigned int> countsDiff(len);
    for (unsigned int y = 0; y < Y; ++y) {
        for (unsigned int x = 0; x < X; ++x) {
            radialMap[y][x] -= minRmap;
            countsRaw[radialMap[y][x]]++;
        }
    }

    py::array_t<float> outRad({len});
    py::array_t<float> outData({hitNum, len});
    py::array_t<unsigned int> outCount({hitNum, len});
    auto outDataPtr = outData.mutable_unchecked<2>();
    auto outCountPtr = outCount.mutable_unchecked<2>();
    auto outRadPtr = outRad.mutable_unchecked<1>();
    for(unsigned int i = 0; i < len; i++) {
        outRadPtr(i) = (i+minRmap) * bin_size;
    }
    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for (unsigned int i = 0; i < len; ++i) {
            outDataPtr(hitId, i) = 0;
        }
    }

    for(unsigned int hitId = 0; hitId < hitNum; hitId++) {
        for(unsigned int i=0; i<len; i++) {
            outCountPtr(hitId, i) = countsRaw[i];
        }
        for (unsigned int y = 0; y < Y; ++y) {
            for (unsigned int x = 0; x < X; ++x) {
                unsigned int rmap = radialMap[y][x];
                if (maskPtr(y, x, hitId)) {
                    auto hv = hitsPtr(y, x, hitId);
                    outDataPtr(hitId, rmap) +=  hv * hv;
                } else {
                    outCountPtr(hitId, rmap)--;
                }
            }
        }
    }

    return std::make_tuple(outData, outCount, outRad);
}

PYBIND11_MODULE(cpplib, m) {
    m.doc() = "Module docstring";
    m.def("correctAGIPD", &correct_array, "Convert raw AGIPD data to calibrated.");
    m.def("radialAverage", &radial_avg, "Input: center x, center y, mask:(Y, X, cell Id), hits:(Y, X, cell Id), bin_size\nOutput: pixel sum of different radi, pixel count of different radi, radi");
    m.def("angularAverage", &angular_avg);
    m.def("radialM2", &radial_m2, "Input: center x, center y, mask:(Y, X, cell Id), hits:(Y, X, cell Id), bin_size\nOutput: pixel squared sum of different radi, pixel count of different radi, radi");
}
