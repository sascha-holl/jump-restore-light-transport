// begin [sholl]
#include <algorithm>
#include <execution>
#include <sstream>

#include <boost/iterator/counting_iterator.hpp>
// end [sholl]

#include "image.h"

#include <OpenImageIO/imageio.h>
namespace OpenImageIO = OIIO;
Image3::Image3(const std::string &filename) {
    std::unique_ptr<OpenImageIO::ImageInput> in = OpenImageIO::ImageInput::open(filename);
    if (in == nullptr)
    {
        // begin [sholl]
        std::stringstream ss;
        ss << "File not found: " << filename << '\n';
        // end [sholl]
        Error(ss.str());
        //Error("File not found");
    }
    const OpenImageIO::ImageSpec &spec = in->spec();
    pixelWidth = spec.width;
    pixelHeight = spec.height;
    OpenImageIO::TypeDesc typeDesc = sizeof(Float) == sizeof(double) ? OpenImageIO::TypeDesc::DOUBLE
                                                                     : OpenImageIO::TypeDesc::FLOAT;
    
    if (spec.nchannels == 1) {
        std::vector<Float> pixels(pixelWidth * pixelHeight);
        in->read_image(typeDesc, &pixels[0]);
        data.resize(pixelWidth * pixelHeight);
        int p = 0;
        for (int y = 0; y < pixelHeight; y++) {
            for (int x = 0; x < pixelWidth; x++) {
                Float val = pixels[p++];
                At(x, y) = Vector3(val, val, val);
            }
        }
    } else if (spec.nchannels == 3) {
        std::vector<Float> pixels(3 * pixelWidth * pixelHeight);
        in->read_image(typeDesc, &pixels[0]);
        data.resize(pixelWidth * pixelHeight);
        int p = 0;
        for (int y = 0; y < pixelHeight; y++) {
            for (int x = 0; x < pixelWidth; x++) {
                Float val0 = pixels[p++];
                Float val1 = pixels[p++];
                Float val2 = pixels[p++];
                At(x, y) = Vector3(val0, val1, val2);
            }
        }
    } else {
        printf("image filename: %s, channels: %d\n", filename.c_str(), spec.nchannels);
        Error("Unsupported number of channels");
    }
    in->close();
}

// [sholl]
Vector3 Image3::MSE(Image3 const& reference, Image3* mse_image) const
{
    if (mse_image)
    {
        mse_image->data.resize(pixelWidth * pixelHeight);
        std::fill(std::execution::par_unseq, mse_image->data.begin(), mse_image->data.end(), Vector3::Zero());
    }

    //Eigen::Vector3<std::atomic<Float>> mse;
    //std::for_each_n(std::execution::par_unseq, boost::counting_iterator<std::size_t>{ 0 }, data.size(), [&](std::size_t const i)
    //{
    //    for (std::size_t j = 0; j < 3; ++j)
    //    {
    //        Float const difference = data[i][j] - reference.data[i][j];
    //        if (!std::isinf(difference))
    //        {
    //            Float const squared_difference = difference * difference;
    //            mse[j] += squared_difference;
    //            if (mse_image)
    //                mse_image->data[i][j] = squared_difference;
    //        }
    //    }
    //});
    //return Vector3{ mse[0], mse[1], mse[2] } / (pixelWidth * pixelHeight);

    Eigen::Vector3<double> mse{};
    for (int v = 0; v < pixelHeight; ++v)
    {
        for (int u = 0; u < pixelWidth; ++u)
        {
            int const i = v * pixelWidth + u;
            for (std::size_t j = 0; j < 3; ++j)
            {
                double const difference = data[i][j] - reference.data[i][j];
                if (!std::isinf(difference))
                {
                    Float const squared_difference = difference * difference;
                    if (std::isinf(difference))
                        std::cerr << "std::isinf(difference)\n" << std::flush;
                    mse[j] += squared_difference;
                    if (mse_image)
                    {
                        //if (u == 1 && v == 1)
                        //    mse_image->data[i] = Vector3{ 1.1, 1.2, 1.3 };
                        //else if (u == 1 && v == 2)
                        //    mse_image->data[i] = Vector3{ 2.1, 2.2, 2.3 };
                        //else
                        //    mse_image->data[i][j] = reference.data[i][j];
                        mse_image->data[i][j] = squared_difference;
                    }
                }
            }
        }
    }
    return {
        static_cast<Float>(mse[0] / (pixelWidth * pixelHeight)),
        static_cast<Float>(mse[1] / (pixelWidth * pixelHeight)),
        static_cast<Float>(mse[2] / (pixelWidth * pixelHeight))
    };
}

void WriteImage(const std::string &filename, const Image3 *image) {
    std::unique_ptr<OpenImageIO::ImageOutput> out = OpenImageIO::ImageOutput::create(filename);
    if (out == nullptr) {
        Error("Fail to create file");
        return;
    }
    OpenImageIO::ImageSpec spec(
        image->pixelWidth, image->pixelHeight, 3, OpenImageIO::TypeDesc::HALF);
    out->open(filename, spec);
    out->write_image(sizeof(Float) == sizeof(double) ? OpenImageIO::TypeDesc::DOUBLE
                                                     : OpenImageIO::TypeDesc::FLOAT,
                     &image->data[0]);
    out->close();
    // OpenImageIO::ImageOutput::destroy(out.get());
}