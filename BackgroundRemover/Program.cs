// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using SixLabors.ImageSharp;
using NumSharp.Generic;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime.Tensors;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
        var img = "img/input/"+
            "3.png";
        string outputFilePath = "";
        FileInfo inputFileInfo = new FileInfo(img);
        Console.WriteLine("Input: " + inputFileInfo.FullName);
        DirectoryInfo outputDirectory = new DirectoryInfo(inputFileInfo.Directory.Parent.Name + "/output/");        
        if(!outputDirectory.Exists )
        {
            outputDirectory.Create();
        }
        outputFilePath = outputDirectory.FullName + inputFileInfo.Name;
        Console.WriteLine("Output: " + outputFilePath);
        string modelFilePath = "model/isnet_fp16";
        var session = initBase(modelFilePath);
        var inputImageTensor = imageSourceToImageData(img);
        Console.WriteLine("compute:inference");
        var (alphamask, imageTensor) = runInference(inputImageTensor, session);
        Console.WriteLine("compute:mask");
        var outImageTensor = imageTensor;
        var width = outImageTensor.shape[0];
        var height = outImageTensor.shape[1];
        var stride = width * height;
        var bytes = outImageTensor.Data<byte>().ToArray();
        var maskData = alphamask.GetData<byte>();
        for (int i = 0; i < stride; i += 1)
        {
            //bytes[4 * i + 3] = maskData[i];
            var x = bytes[4 * i + 3];
            var y = maskData[i];
            bytes[4 * i + 3] = y;
        }

        ToImage(outImageTensor, bytes, outputFilePath);

    }
   
    static void SetRgba8PixelArray(Image<Rgba32> image, byte[] rgbaBytes)
    {
        int width = image.Width;
        int height = image.Height;

        // 将 RGBA8 像素数据设置到图像中
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int offset = (y * width + x) * 4;
                Rgba32 pixel = new Rgba32(rgbaBytes[offset], rgbaBytes[offset + 1], rgbaBytes[offset + 2], rgbaBytes[offset + 3]);
                image[x, y] = pixel;
            }
        }
    }
    static void ToImage(NDArray<byte> outImageTensor, byte[] data, string outputFilePath)
    {
        var srcHeight = outImageTensor.shape[0];
        var srcWidth = outImageTensor.shape[1];
        var srcChannels = outImageTensor.shape[2];
        //var fn = "output/"+;
        using (Image<Rgba32> image = new Image<Rgba32>(srcWidth, srcHeight))
        {
            // 将 RGBA8 像素数据设置到图像中
            SetRgba8PixelArray(image, data);

            // 保存为 JPG 图片
            image.SaveAsPng(outputFilePath);
        }
        //System.Drawing.Image.FromFile(fn).Dump();
    }

    static (NDArray<byte> alphamask, NDArray<byte> imageTensor) runInference(NDArray<byte> inputImageTensor, InferenceSession session)
    {
        const int resolution = 1024;
        var srcHeight = inputImageTensor.shape[0];
        var srcWidth = inputImageTensor.shape[1];
        var srcChannels = inputImageTensor.shape[2];
        var keepAspect = false;
        var resizedImageTensor = tensorResizeBilinear(inputImageTensor, resolution, resolution, keepAspect);
        var inputTensor = TensorHWCtoBCHW(resizedImageTensor);
        var feeds = new List<NamedOnnxValue>();
        var input = new DenseTensor<float>(new Memory<float>(inputTensor.Data<float>().ToArray(), 0, inputTensor.size), inputTensor.shape);

        feeds.Add(NamedOnnxValue.CreateFromTensor("input", input));
        var outputData = session.Run(feeds);
        var ret = new List<NDArray<float>>();
        foreach (var item in outputData)
        {
            if (item.Name == "output")
            {
                //ret.Add(new NDArray<float>(item.va
                if (item.Value is DenseTensor<float> data)
                {
                    var n = new NDArray<float>(data.ToArray(), new Shape(data.Dimensions.ToArray()));
                    ret.Add(n);
                }
            }
        }
        var alphamask = new NDArray<float>(ret[0].Data<float>(), new Shape(resolution, resolution, 1));
        var alphamaskU8 = convertFloat32ToUint8(alphamask);
        alphamaskU8 = tensorResizeBilinear(alphamaskU8, srcWidth, srcHeight, keepAspect);
        return (alphamaskU8, inputImageTensor);
    }
    static NDArray<byte> convertFloat32ToUint8(NDArray<float> float32Array)
    {
        int count = float32Array.size;
        var uint8Array = new byte[count];
        var data = float32Array.GetData<float>();
        for (int i = 0; i < count; i++)
        {
            uint8Array[i] = Convert.ToByte(data[i] * 255);
        }
        return new NDArray<byte>(uint8Array, float32Array.shape);
    }
    static NDArray<float> TensorHWCtoBCHW(
        NDArray<byte> imageTensor,
        float[] mean = null,
        float[] std = null)
    {
        // Default mean and std if not provided
        if (mean == null)
            mean = new float[] { 128, 128, 128 };
        if (std == null)
            std = new float[] { 256, 256, 256 };

        var imageBufferData = imageTensor.Data<byte>();
        var srcHeight = imageTensor.shape[0];
        var srcWidth = imageTensor.shape[1];
        var srcChannels = imageTensor.shape[2];

        int stride = srcHeight * srcWidth;
        float[] float32Data = new float[3 * stride];

        var i = 0;
        var j = 0;
        for (; i < imageBufferData.Count; i += 4, j += 1)
        {
            float32Data[j] = 1f * (imageBufferData[i] - mean[0]) / std[0];
            float32Data[j + stride] = 1f * (imageBufferData[i + 1] - mean[1]) / std[1];
            float32Data[j + stride + stride] = 1f * (imageBufferData[i + 2] - mean[2]) / std[2];
        }

        return new NDArray<float>(float32Data, new Shape(1, 3, srcHeight, srcWidth));
    }
    static NDArray<byte> tensorResizeBilinear(NDArray<byte> imageTensor, int newWidth, int newHeight, bool proportional)
    {

        var srcHeight = imageTensor.shape[0];
        var srcWidth = imageTensor.shape[1];
        var srcChannels = imageTensor.shape[2];
        var scaleX = 1f * srcWidth / newWidth;
        var scaleY = 1f * srcHeight / newHeight;
        if (proportional)
        {
            var downscaling = Math.Max(scaleX, scaleY) > 1.0;
            scaleX = scaleY = downscaling
              ? Math.Max(scaleX, scaleY)
              : Math.Min(scaleX, scaleY);
        }
        var rgbaBytes = new byte[srcChannels * newWidth * newHeight];
        var resizedImageData = new NDArray<byte>(rgbaBytes, new Shape(newHeight, newWidth, srcChannels));
        // Perform interpolation to fill the resized NdArray
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                var srcX = x * scaleX;
                var srcY = y * scaleY;
                int x1 = (int)Math.Max(Math.Floor(srcX), 0);
                int x2 = (int)Math.Min(Math.Ceiling(srcX), srcWidth - 1);
                int y1 = (int)Math.Max(Math.Floor(srcY), 0);
                int y2 = (int)Math.Min(Math.Ceiling(srcY), srcHeight - 1);

                double dx = srcX - x1;
                double dy = srcY - y1;

                for (int c = 0; c < srcChannels; c++)
                {
                    byte p1 = imageTensor[y1, x1, c];
                    byte p2 = imageTensor[y1, x2, c];
                    byte p3 = imageTensor[y2, x1, c];
                    byte p4 = imageTensor[y2, x2, c];

                    // Perform bilinear interpolation
                    var interpolatedValue =
                        (1 - dx) * (1 - dy) * p1 +
                        dx * (1 - dy) * p2 +
                        (1 - dx) * dy * p3 +
                        dx * dy * p4;

                    // Assuming resizedImageData is a suitable structure to hold the interpolated value
                    //resizedImageData[y, x, c] =(byte) Math.Round(interpolatedValue);
                    resizedImageData.SetData((byte)Math.Round(interpolatedValue), y, x, c);
                }
            }
        }
        return resizedImageData;
    }

    static InferenceSession initBase(string modelFilePath)
    {
        var options = new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            EnableCpuMemArena = true,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        //var model = @"D:\src\github.com\background-removal-js\bundle\models\isnet_fp16";
        var session = new InferenceSession(modelFilePath, options);
        return session;
    }

    static NDArray<byte> imageSourceToImageData(string path)
    {
        using var image = Image.Load<Rgba32>(path);
        int width = image.Width;
        int height = image.Height;
        byte[] rgbaBytes = new byte[width * height * 4];

        int index = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Rgba32 pixel = image[x, y];
                rgbaBytes[index++] = pixel.R;
                rgbaBytes[index++] = pixel.G;
                rgbaBytes[index++] = pixel.B;
                rgbaBytes[index++] = pixel.A;
            }
        }
        var ret = new NDArray<byte>(rgbaBytes, new Shape(height, width, 4));
        return ret;
    }
}