#define NSavingThreads 10
#include "../CudaInc/EHCuda.h"

CreateThread(Reading_Thread);
CreateMutex(ReadingThread_Mutex);
CreateCond(ReadingThread_Cond);

CreateThread(Saving_Thread[NSavingThreads]);
CreateMutex(SavingThread_Mutex[NSavingThreads]);
CreateCond(SavingThread_Cond[NSavingThreads]);

CreateStream(Reading_Stream);
CreateStream(Processing_Stream);

struct
slab_background
{
	uint2 slabDims;
	three_d_volume *ballVol;
	u32 *background;
};

struct
slab_background_LL_node
{
	slab_background *slabBackground;
	slab_background_LL_node *next;
};

struct
interpolation_background
{
	slab_background_LL_node *slabBackground;
	u32 doInterpolation;
	u32 slabCounter;
};

void
InitialiseSlabBackground(slab_background *slabBackground, uint2 slabDims, three_d_volume *ballVol)
{
	slabBackground->ballVol = ballVol;	
	slabBackground->slabDims = slabDims;
	CudaMallocManaged(slabBackground->background, slabDims.x * slabDims.y * sizeof(f32));
}

void
FreeSlabBackground(slab_background *slabBackground)
{
	cudaFree(slabBackground->background);
}

void
SlabBackgroundLLAddAfter(slab_background_LL_node *head, slab_background_LL_node *tail)
{
	head->next = tail;
}

void
InitialiseSlabBackgroundLLNode(slab_background_LL_node *node, uint2 slabDims, three_d_volume *ballVol)
{
	InitialiseSlabBackground(node->slabBackground, slabDims, ballVol);
}

void
FreeSlabBackgroundLLNode(slab_background_LL_node *node)
{
	FreeSlabBackground(node->slabBackground);
}

void
InitialiseInterpolationBackground(interpolation_background *interpolationBackground, uint2 slabDims, three_d_volume *ballVol)
{

	InitialiseSlabBackgroundLLNode(interpolationBackground->slabBackground, slabDims, ballVol);
	interpolationBackground->doInterpolation = 0;
	interpolationBackground->slabCounter = 0;
}

struct
image_filter
{
	stream fftStream;
	cufftHandle fftPlanFwd;
	cufftHandle fftPlanInv;
	dim3 fftDims;
	dim3 kernelDims;
	dim3 kernelCentre;
	fComplex *imageSpectrum;
	fComplex *kernelSpectrum;
	f32 *paddedImage;
	f32 *imageOut;
};

struct rolling_image_buffer
{
	u32 capacity;
	image_data *buffers[2];
	slab_background_LL_node *slabBackground;
	three_d_volume fullImageVolume;
	image_loader *loader;
	u32 currentZStart;
	image_filter *filter;
};

void
CalculateFFTDims(dim3 *result, dim3 imageDims, dim3 kernelDims)
{
	u32 *result_array = (u32 *)result;
	u32 *imageDims_array = (u32 *)(&imageDims);
	u32 *kernelDims_array = (u32 *)(&kernelDims);

	__m128i imageDims_intrinsic = _mm_setr_epi32(imageDims_array[0], imageDims_array[1], imageDims_array[2], 0);
	__m128i kernelDims_intrinsic = _mm_setr_epi32(kernelDims_array[0], kernelDims_array[1], kernelDims_array[2], 0);
	
	__m128i one = _mm_setr_epi32(1, 1, 1, 1);
	__m128i fifteen = _mm_setr_epi32(15, 15, 15, 15);
	__m128i fiveeleven = _mm_setr_epi32(511, 511, 511, 511);

	__m128i round16 = _mm_sub_epi32(_mm_add_epi32(imageDims_intrinsic, kernelDims_intrinsic), one);
	round16 = _mm_andnot_si128(fifteen, _mm_add_epi32(round16, fifteen));
	round16 = _mm_sub_epi32(round16, one);

	u32 highBit[3];
	asm(
	"bsrl %1, %0 \n\t"
	: "=r" (highBit[0])
	: "r" (_mm_extract_epi32(round16, 0)));
	asm(
	"bsrl %1, %0 \n\t"
	: "=r" (highBit[1])
	: "r" (_mm_extract_epi32(round16, 1)));
	asm(
	"bsrl %1, %0 \n\t"
	: "=r" (highBit[2])
	: "r" (_mm_extract_epi32(round16, 2)));

	__m128i mask = _mm_setr_epi32((highBit[0] < 10) * 0xffffffff, (highBit[1] < 10) * 0xffffffff, (highBit[2] < 10) * 0xffffffff, 0xffffffff);

	__m128i result_simd = _mm_blendv_epi8(_mm_andnot_si128(fiveeleven, _mm_add_epi32(round16, fiveeleven)), _mm_setr_epi32((1U << (highBit[0] + 1)), (1U << (highBit[1] + 1)), (1U << (highBit[2] + 1)), 0), mask);

	result_array[0] = _mm_extract_epi32(result_simd, 0);
	result_array[1] = _mm_extract_epi32(result_simd, 1);
	result_array[2] = _mm_extract_epi32(result_simd, 2);
}

CudaKernel
PadKernel_Cuda(f32 *paddedKernel, f32 *kernel, dim3 fftDims, dim3 kernelDims, dim3 kernelCentre)
{
	u32 kernelN = Dim3N(kernelDims);

	OneDCudaLoop(index, kernelN)
	{
		int3 index_int3 = OneDToThreeD(index, kernelDims);
		dim3 index_dim3 = Int3ToDim3(index_int3);

		int3 k = Dim3Subtract(index_dim3, kernelCentre);

		if (k.x < 0) k.x += fftDims.x;
		if (k.y < 0) k.y += fftDims.y;
		if (k.z < 0) k.z += fftDims.z;

		paddedKernel[ThreeDToOneD(Int3ToDim3(k), fftDims)] = kernel[index];
	}
}

u32
FFTDataSize(dim3 fftDims)
{
	return(fftDims.x * fftDims.y * ((fftDims.z / 2) + 1));
}

void
CreateImageFilter(image_filter *filter, dim3 imageDims, f32 *kernel, dim3 kernelDims, dim3 kernelCentre)
{
	cudaStreamCreate(&filter->fftStream);
	
	CalculateFFTDims(&filter->fftDims, imageDims, kernelDims);
	filter->kernelDims = kernelDims;
	filter->kernelCentre = kernelCentre;
	
	printf("FFT dims: %u,%u,%u\n", filter->fftDims.x, filter->fftDims.y, filter->fftDims.z);

	CudaMallocManaged(filter->imageOut, Dim3N(imageDims) * sizeof(f32));
	CudaMallocManaged(filter->paddedImage, Dim3N(filter->fftDims) * sizeof(f32));	
	CudaMallocManaged(filter->imageSpectrum, FFTDataSize(filter->fftDims) * sizeof(fComplex));	
	CudaMallocManaged(filter->kernelSpectrum, FFTDataSize(filter->fftDims) * sizeof(fComplex));	
	cudaMemset(filter->paddedImage, 0, Dim3N(filter->fftDims) * sizeof(f32));

	cufftPlan3d(&filter->fftPlanFwd, filter->fftDims.x, filter->fftDims.y, filter->fftDims.z, CUFFT_R2C);
	cufftPlan3d(&filter->fftPlanInv, filter->fftDims.x, filter->fftDims.y, filter->fftDims.z, CUFFT_C2R);

	cufftSetStream(filter->fftPlanFwd, filter->fftStream);
	cufftSetStream(filter->fftPlanInv, filter->fftStream);
	
	f32 *d_kernel, *paddedKernel;
	CudaMallocManaged(d_kernel, Dim3N(kernelDims) * sizeof(f32));
	cudaMemcpy(d_kernel, kernel, Dim3N(kernelDims) * sizeof(f32), cudaMemcpyHostToDevice);
	CudaMallocManaged(paddedKernel, Dim3N(filter->fftDims) * sizeof(f32));
	cudaMemset(paddedKernel, 0, Dim3N(filter->fftDims) * sizeof(f32));

	LaunchCudaKernel_Simple_Stream(PadKernel_Cuda, filter->fftStream, paddedKernel, d_kernel, filter->fftDims, filter->kernelDims, filter->kernelCentre);
	cufftExecR2C(filter->fftPlanFwd, (cufftReal *)paddedKernel, (cufftComplex *)filter->kernelSpectrum);

	cudaFree(paddedKernel);
	cudaFree(d_kernel);
}

void
FreeImageFilter(image_filter *filter)
{
	cudaFree(filter->paddedImage);
	cudaFree(filter->imageSpectrum);
	cudaFree(filter->kernelSpectrum);
	cudaFree(filter->imageOut);
}

void
CreateRollingImageBuffer(rolling_image_buffer *buffer, image_loader *loader, image_data *im1, image_data *im2, u32 localZ, u32 bytesPerPixel, f32 *kernel, dim3 kernelDims, dim3 kernelCentre)
{
	dim3 imageDims;
	imageDims.x = ImageLoaderGetWidth(loader);
	imageDims.y = ImageLoaderGetHeight(loader);
	imageDims.z = localZ;
	
	CreateImageData_FromDim3(im1, imageDims, bytesPerPixel);
	CreateImageData_FromDim3(im2, imageDims, bytesPerPixel);
	
	CreateImageFilter(buffer->filter, imageDims, kernel, kernelDims, kernelCentre);
	
	buffer->buffers[0] = im1;
	buffer->buffers[1] = im2;
	buffer->loader = loader;
	buffer->currentZStart = 0;
	buffer->capacity = 2 * localZ;

	three_d_volume vol;
	u32 trackSize;
	
	u32 *startingIndex = GetImageLoaderCurrentTrackIndexAndSize(loader, &trackSize);
	CreateThreeDVol_FromInt(&vol, ImageLoaderGetWidth(loader), ImageLoaderGetHeight(loader), trackSize - *startingIndex);
	
	buffer->fullImageVolume = vol;
}

void
FreeRollingImageBuffer(rolling_image_buffer *buffer)
{
	FreeImageData(buffer->buffers[0]);
	FreeImageData(buffer->buffers[1]);
	FreeImageLoader(buffer->loader);
	FreeImageFilter(buffer->filter);
}

three_d_volume*
GetImageVolume_FromImageBuffer(rolling_image_buffer *buffer)
{
	return(&buffer->fullImageVolume);
}

u08*
GetContentsOfCurrentBuffer(rolling_image_buffer *buffer)
{
	return(buffer->buffers[0]->image);	
}

u08*
GetContentsOfNextBuffer(rolling_image_buffer *buffer)
{
	return(buffer->buffers[1]->image);	
}

CudaKernel
ResetBackground(u32 *background, dim3 ballGridDims)
{
	u32 ballGridN = Dim3N(ballGridDims);

	OneDCudaLoop(index, ballGridN)
	{
		background[index] = u32_max;
	}
}

CudaKernel
CalculateBackground_Cuda_u08(u08 *image, u32 *background, f32 ballCentreHeight, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	dim3 ballGlobalDims = Dim3Hadamard(ballGridDims, ballRadius);
	u32 ballGlobalN = Dim3N(ballGlobalDims);

	OneDCudaLoop(index, ballGlobalN)
	{
		int3 ballGlobalPosition_asInt = OneDToThreeD(index, ballGlobalDims);
		dim3 ballGlobalPosition_asDim = Int3ToDim3(ballGlobalPosition_asInt);
		dim3 ballGridPosition = Dim3Divide(ballGlobalPosition_asDim, ballRadius);
		int3 ballLocalPosition_asInt = Dim3Subtract(ballGlobalPosition_asDim, Dim3Hadamard(ballGridPosition, ballRadius));
		dim3 ballLocalPosition_asDim = Int3ToDim3(ballLocalPosition_asInt);
		u32 ballIndex = ThreeDToOneD(ballGridPosition, ballGridDims);
		dim3 imageCoords = CoordinateMap_Reflection(ballGlobalPosition_asInt, imageDims);
		u32 imageIndex = ThreeDToOneD(imageCoords, imageDims);
		
		int3 ballLocalVectorFromCentre = Dim3Subtract(ballLocalPosition_asDim, Dim3Half_Ceil(ballRadius));
		u32 ballLocalDistanceFromCentreSq = Int3VectorLengthSq(ballLocalVectorFromCentre);

		f32 ballHeight = ballCentreHeight - sqrtf((f32)ballLocalDistanceFromCentreSq);
		f32 heightAboveBall = (f32)image[imageIndex] - ballHeight;

		u32 candidateBg = *((u32 *)(&heightAboveBall));
		candidateBg = FloatFlip(candidateBg);
		atomicMin(background + ballIndex, candidateBg);
	}
}

CudaKernel
CalculateBackground_Cuda_u16(u16 *image, u32 *background, f32 ballCentreHeight, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	dim3 ballGlobalDims = Dim3Hadamard(ballGridDims, ballRadius);
	u32 ballGlobalN = Dim3N(ballGlobalDims);

	OneDCudaLoop(index, ballGlobalN)
	{
		int3 ballGlobalPosition_asInt = OneDToThreeD(index, ballGlobalDims);
		dim3 ballGlobalPosition_asDim = Int3ToDim3(ballGlobalPosition_asInt);
		dim3 ballGridPosition = Dim3Divide(ballGlobalPosition_asDim, ballRadius);
		int3 ballLocalPosition_asInt = Dim3Subtract(ballGlobalPosition_asDim, Dim3Hadamard(ballGridPosition, ballRadius));
		dim3 ballLocalPosition_asDim = Int3ToDim3(ballLocalPosition_asInt);
		u32 ballIndex = ThreeDToOneD(ballGridPosition, ballGridDims);
		dim3 imageCoords = CoordinateMap_Reflection(ballGlobalPosition_asInt, imageDims);
		u32 imageIndex = ThreeDToOneD(imageCoords, imageDims);
		
		int3 ballLocalVectorFromCentre = Dim3Subtract(ballLocalPosition_asDim, Dim3Half_Ceil(ballRadius));
		u32 ballLocalDistanceFromCentreSq = Int3VectorLengthSq(ballLocalVectorFromCentre);

		f32 ballHeight = ballCentreHeight - sqrtf((f32)ballLocalDistanceFromCentreSq);
		f32 heightAboveBall = (f32)image[imageIndex] - ballHeight;

		u32 candidateBg = *((u32 *)(&heightAboveBall));
		candidateBg = FloatFlip(candidateBg);
		
		atomicMin(background + ballIndex, candidateBg);
	}
}

CudaKernel
CalculateBackground_Cuda_u32(u32 *image, u32 *background, f32 ballCentreHeight, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	dim3 ballGlobalDims = Dim3Hadamard(ballGridDims, ballRadius);
	u32 ballGlobalN = Dim3N(ballGlobalDims);

	OneDCudaLoop(index, ballGlobalN)
	{
		int3 ballGlobalPosition_asInt = OneDToThreeD(index, ballGlobalDims);
		dim3 ballGlobalPosition_asDim = Int3ToDim3(ballGlobalPosition_asInt);
		dim3 ballGridPosition = Dim3Divide(ballGlobalPosition_asDim, ballRadius);
		int3 ballLocalPosition_asInt = Dim3Subtract(ballGlobalPosition_asDim, Dim3Hadamard(ballGridPosition, ballRadius));
		dim3 ballLocalPosition_asDim = Int3ToDim3(ballLocalPosition_asInt);
		u32 ballIndex = ThreeDToOneD(ballGridPosition, ballGridDims);
		dim3 imageCoords = CoordinateMap_Reflection(ballGlobalPosition_asInt, imageDims);
		u32 imageIndex = ThreeDToOneD(imageCoords, imageDims);
		
		int3 ballLocalVectorFromCentre = Dim3Subtract(ballLocalPosition_asDim, Dim3Half_Ceil(ballRadius));
		u32 ballLocalDistanceFromCentreSq = Int3VectorLengthSq(ballLocalVectorFromCentre);

		f32 ballHeight = ballCentreHeight - sqrtf((f32)ballLocalDistanceFromCentreSq);
		f32 heightAboveBall = (f32)image[imageIndex] - ballHeight;

		u32 candidateBg = *((u32 *)(&heightAboveBall));
		candidateBg = FloatFlip(candidateBg);
		atomicMin(background + ballIndex, candidateBg);
	}
}

CudaKernel
CalculateBackground_Cuda_f32(f32 *image, u32 *background, f32 ballCentreHeight, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	dim3 ballGlobalDims = Dim3Hadamard(ballGridDims, ballRadius);
	u32 ballGlobalN = Dim3N(ballGlobalDims);

	OneDCudaLoop(index, ballGlobalN)
	{
		int3 ballGlobalPosition_asInt = OneDToThreeD(index, ballGlobalDims);
		dim3 ballGlobalPosition_asDim = Int3ToDim3(ballGlobalPosition_asInt);
		dim3 ballGridPosition = Dim3Divide(ballGlobalPosition_asDim, ballRadius);
		int3 ballLocalPosition_asInt = Dim3Subtract(ballGlobalPosition_asDim, Dim3Hadamard(ballGridPosition, ballRadius));
		dim3 ballLocalPosition_asDim = Int3ToDim3(ballLocalPosition_asInt);
		u32 ballIndex = ThreeDToOneD(ballGridPosition, ballGridDims);
		dim3 imageCoords = CoordinateMap_Reflection(ballGlobalPosition_asInt, imageDims);
		u32 imageIndex = ThreeDToOneD(imageCoords, imageDims);
		
		int3 ballLocalVectorFromCentre = Dim3Subtract(ballLocalPosition_asDim, Dim3Half_Ceil(ballRadius));
		u32 ballLocalDistanceFromCentreSq = Int3VectorLengthSq(ballLocalVectorFromCentre);

		f32 ballHeight = ballCentreHeight - sqrtf((f32)ballLocalDistanceFromCentreSq);
		f32 heightAboveBall = image[imageIndex] - ballHeight;

		u32 candidateBg = *((u32 *)(&heightAboveBall));
		candidateBg = FloatFlip(candidateBg);
		atomicMin(background + ballIndex, candidateBg);
	}
}

CudaFunction
dim3
GetCurrentAndNextBallAndDist(u32 coord, u32 ballDim, u32 nBalls)
{
	dim3 result;

	u32 halfBallDim = ballDim / 2;
	if (coord < halfBallDim)
	{
		result.x = 0;
		result.y = 0;
		result.z = 0;
	}
	else
	{
		result.x = (coord - halfBallDim) / ballDim;
		result.y = result.x + 1;
		result.z = coord - ((result.x * ballDim) + halfBallDim);
	}

	if (result.y >= nBalls)
	{
		result.y = result.x;
		result.z = 0;
	}

	return(result);
}

CudaFunction
f32
BiLinear(u32 *ballIndex, u32 *background, f32 disX, f32 disY, f32 ballCentreHeight)
{
	f32 omDisX = 1.0 - disX;
	f32 omDisY = 1.0 - disY;

	f32 values[4];
	for ( 	u32 index = 0;
		index < 4;
		++index )
	{
		u32 bg = IFloatFlip(*(background + ballIndex[index]));
		values[index] = ballCentreHeight + *((f32 *)(&bg));
	}

	f32 result = 	(omDisX	* omDisY	*	values[0]) +
			(disX 	* omDisY	*	values[1]) +
			(omDisX	* disY		*	values[2]) +
			(disX	* disY		*	values[3]);

	return(result);
}

//TODO use bilinear texture fetches for background?
CudaKernel
RemoveBackground_Cuda_u08(u08 *image, u08 *out, u32 *background_1, u32 *background_2, f32 ballCentreHeight, f32 interpolation, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	u32 imageN = Dim3N(imageDims);

	OneDCudaLoop(index, imageN)
	{
		dim3 imageCoords = Int3ToDim3(OneDToThreeD(index, imageDims));

		dim3 interpX = GetCurrentAndNextBallAndDist(imageCoords.x, ballRadius.x, ballGridDims.x);
		dim3 interpY = GetCurrentAndNextBallAndDist(imageCoords.y, ballRadius.y, ballGridDims.y);
		f32 disX = (f32)interpX.z / (f32)ballRadius.x;
		f32 disY = (f32)interpY.z / (f32)ballRadius.y;

		dim3 ballGridPosition[4];
		ballGridPosition[0].x = interpX.x;
		ballGridPosition[0].y = interpY.x;
		ballGridPosition[0].z = 0;
		ballGridPosition[1].x = interpX.y;
		ballGridPosition[1].y = interpY.x;
		ballGridPosition[1].z = 0;
		ballGridPosition[2].x = interpX.x;
		ballGridPosition[2].y = interpY.y;
		ballGridPosition[2].z = 0;
		ballGridPosition[3].x = interpX.y;
		ballGridPosition[3].y = interpY.y;
		ballGridPosition[3].z = 0;

		u32 ballIndex[4];
		for (	u32 bIndex = 0;
			bIndex < 4;
			++bIndex )
		{
			ballIndex[bIndex] = ThreeDToOneD(ballGridPosition[bIndex], ballGridDims);
		}

		f32 planeInterpVals[2];
		planeInterpVals[0] = BiLinear(ballIndex, background_1, disX, disY, ballCentreHeight);
		planeInterpVals[1] = BiLinear(ballIndex, background_2, disX, disY, ballCentreHeight);
		
		f32 bg = ((1.0 - interpolation) * planeInterpVals[0]) + (interpolation * planeInterpVals[1]);

		out[index] = (u08)(Max(((f32)image[index] - bg), 0) + 0.5);
	}
}

CudaKernel
RemoveBackground_Cuda_u16(u16 *image, u16 *out, u32 *background_1, u32 *background_2, f32 ballCentreHeight, f32 interpolation, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	u32 imageN = Dim3N(imageDims);

	OneDCudaLoop(index, imageN)
	{
		dim3 imageCoords = Int3ToDim3(OneDToThreeD(index, imageDims));

		dim3 interpX = GetCurrentAndNextBallAndDist(imageCoords.x, ballRadius.x, ballGridDims.x);
		dim3 interpY = GetCurrentAndNextBallAndDist(imageCoords.y, ballRadius.y, ballGridDims.y);
		f32 disX = (f32)interpX.z / (f32)ballRadius.x;
		f32 disY = (f32)interpY.z / (f32)ballRadius.y;

		dim3 ballGridPosition[4];
		ballGridPosition[0].x = interpX.x;
		ballGridPosition[0].y = interpY.x;
		ballGridPosition[0].z = 0;
		ballGridPosition[1].x = interpX.y;
		ballGridPosition[1].y = interpY.x;
		ballGridPosition[1].z = 0;
		ballGridPosition[2].x = interpX.x;
		ballGridPosition[2].y = interpY.y;
		ballGridPosition[2].z = 0;
		ballGridPosition[3].x = interpX.y;
		ballGridPosition[3].y = interpY.y;
		ballGridPosition[3].z = 0;

		u32 ballIndex[4];
		for (	u32 bIndex = 0;
			bIndex < 4;
			++bIndex )
		{
			ballIndex[bIndex] = ThreeDToOneD(ballGridPosition[bIndex], ballGridDims);
		}

		f32 planeInterpVals[2];
		planeInterpVals[0] = BiLinear(ballIndex, background_1, disX, disY, ballCentreHeight);
		planeInterpVals[1] = BiLinear(ballIndex, background_2, disX, disY, ballCentreHeight);

		f32 bg = ((1.0 - interpolation) * planeInterpVals[0]) + (interpolation * planeInterpVals[1]);

		out[index] = (u16)(Max(((f32)image[index] - bg), 0) + 0.5);
	}
}

CudaKernel
RemoveBackground_Cuda_u32(u32 *image, u32 *out, u32 *background_1, u32 *background_2, f32 ballCentreHeight, f32 interpolation, dim3 imageDims, dim3 ballGridDims, dim3 ballRadius)
{
	u32 imageN = Dim3N(imageDims);

	OneDCudaLoop(index, imageN)
	{
		dim3 imageCoords = Int3ToDim3(OneDToThreeD(index, imageDims));

		dim3 interpX = GetCurrentAndNextBallAndDist(imageCoords.x, ballRadius.x, ballGridDims.x);
		dim3 interpY = GetCurrentAndNextBallAndDist(imageCoords.y, ballRadius.y, ballGridDims.y);
		f32 disX = (f32)interpX.z / (f32)ballRadius.x;
		f32 disY = (f32)interpY.z / (f32)ballRadius.y;

		dim3 ballGridPosition[4];
		ballGridPosition[0].x = interpX.x;
		ballGridPosition[0].y = interpY.x;
		ballGridPosition[0].z = 0;
		ballGridPosition[1].x = interpX.y;
		ballGridPosition[1].y = interpY.x;
		ballGridPosition[1].z = 0;
		ballGridPosition[2].x = interpX.x;
		ballGridPosition[2].y = interpY.y;
		ballGridPosition[2].z = 0;
		ballGridPosition[3].x = interpX.y;
		ballGridPosition[3].y = interpY.y;
		ballGridPosition[3].z = 0;

		u32 ballIndex[4];
		for (	u32 bIndex = 0;
			bIndex < 4;
			++bIndex )
		{
			ballIndex[bIndex] = ThreeDToOneD(ballGridPosition[bIndex], ballGridDims);
		}

		f32 planeInterpVals[2];
		planeInterpVals[0] = BiLinear(ballIndex, background_1, disX, disY, ballCentreHeight);
		planeInterpVals[1] = BiLinear(ballIndex, background_2, disX, disY, ballCentreHeight);
		
		f32 bg = ((1.0 - interpolation) * planeInterpVals[0]) + (interpolation * planeInterpVals[1]);

		out[index] = (u32)(Max(((f32)image[index] - bg), 0) + 0.5);
	}
}

CudaKernel
PadImage_Cuda_u32(f32 *paddedImage, u32 *imageIn, dim3 fftDims, dim3 imageDims, dim3 kernelCentre)
{
	u32 fftN = Dim3N(fftDims);
	dim3 border = Dim3Add(imageDims, kernelCentre);

	OneDCudaLoop(index, fftN)
	{
		int3 index_int3 = OneDToThreeD(index, fftDims);
		dim3 index_dim3 = Int3ToDim3(index_int3);

		dim3 d;

		if (index_dim3.x < imageDims.x)
		{
			d.x = index_dim3.x;
		}
		else if (index_dim3.x < border.x)
		{
			d.x = imageDims.x - 1;
		}
		else
		{
			d.x = 0;
		}
		
		if (index_dim3.y < imageDims.y)
		{
			d.y = index_dim3.y;
		}
		else if (index_dim3.y < border.y)
		{
			d.y = imageDims.y - 1;
		}
		else
		{
			d.y = 0;
		}
		
		if (index_dim3.z < imageDims.z)
		{
			d.z = index_dim3.z;
		}
		else if (index_dim3.z < border.z)
		{
			d.z = imageDims.z - 1;
		}
		else
		{
			d.z = 0;
		}

		paddedImage[index] = (f32)imageIn[ThreeDToOneD(d, imageDims)];
	}
}

CudaKernel
PadImage_Cuda_u16(f32 *paddedImage, u16 *imageIn, dim3 fftDims, dim3 imageDims, dim3 kernelCentre)
{
	u32 fftN = Dim3N(fftDims);
	dim3 border = Dim3Add(imageDims, kernelCentre);

	OneDCudaLoop(index, fftN)
	{
		int3 index_int3 = OneDToThreeD(index, fftDims);
		dim3 index_dim3 = Int3ToDim3(index_int3);

		dim3 d;

		if (index_dim3.x < imageDims.x)
		{
			d.x = index_dim3.x;
		}
		else if (index_dim3.x < border.x)
		{
			d.x = imageDims.x - 1;
		}
		else
		{
			d.x = 0;
		}
		
		if (index_dim3.y < imageDims.y)
		{
			d.y = index_dim3.y;
		}
		else if (index_dim3.y < border.y)
		{
			d.y = imageDims.y - 1;
		}
		else
		{
			d.y = 0;
		}
		
		if (index_dim3.z < imageDims.z)
		{
			d.z = index_dim3.z;
		}
		else if (index_dim3.z < border.z)
		{
			d.z = imageDims.z - 1;
		}
		else
		{
			d.z = 0;
		}

		paddedImage[index] = (f32)imageIn[ThreeDToOneD(d, imageDims)];
	}
}

CudaKernel
PadImage_Cuda_u08(f32 *paddedImage, u08 *imageIn, dim3 fftDims, dim3 imageDims, dim3 kernelCentre)
{
	u32 fftN = Dim3N(fftDims);
	dim3 border = Dim3Add(imageDims, kernelCentre);

	OneDCudaLoop(index, fftN)
	{
		int3 index_int3 = OneDToThreeD(index, fftDims);
		dim3 index_dim3 = Int3ToDim3(index_int3);

		dim3 d;

		if (index_dim3.x < imageDims.x)
		{
			d.x = index_dim3.x;
		}
		else if (index_dim3.x < border.x)
		{
			d.x = imageDims.x - 1;
		}
		else
		{
			d.x = 0;
		}
		
		if (index_dim3.y < imageDims.y)
		{
			d.y = index_dim3.y;
		}
		else if (index_dim3.y < border.y)
		{
			d.y = imageDims.y - 1;
		}
		else
		{
			d.y = 0;
		}
		
		if (index_dim3.z < imageDims.z)
		{
			d.z = index_dim3.z;
		}
		else if (index_dim3.z < border.z)
		{
			d.z = imageDims.z - 1;
		}
		else
		{
			d.z = 0;
		}

		paddedImage[index] = (f32)imageIn[ThreeDToOneD(d, imageDims)];
	}
}

CudaKernel
UnpadImage_Cuda(f32 *paddedImage, f32 *imageOut, dim3 fftDims, dim3 imageDims)
{
	u32 imageN = Dim3N(imageDims);

	OneDCudaLoop(index, imageN)
	{
		imageOut[index] = paddedImage[ThreeDToOneD(Int3ToDim3(OneDToThreeD(index, imageDims)), fftDims)];	
	}
}

CudaFunction
void
MultiplyAndScale(fComplex *a, fComplex b, f32 c)
{
	*a = {c * ((a->x * b.x) - (a->y * b.y)), c * ((a->y * b.x) + (a->x * b.y))};
}

CudaKernel
ModulateAndNormalise_Cuda(fComplex *imageSpectrum, fComplex *kernelSpectrum, u32 dataSize, f32 normalisationConstant)
{
	OneDCudaLoop(index, dataSize)
	{
		MultiplyAndScale(imageSpectrum + index, kernelSpectrum[index], normalisationConstant);
	}
}

f32
FFTNormalisationConstant(dim3 fftDims)
{
	return(1.0 / (f32)(Dim3N(fftDims)));
}

void
PadImage(image_filter *imageFilter, u08 *imageIn, dim3 imageDims, u32 bytesPerPixel)
{
	switch (bytesPerPixel)
	{
		case 4:
			{
				LaunchCudaKernel_Simple_Stream(PadImage_Cuda_u32, imageFilter->fftStream, imageFilter->paddedImage, (u32 *)imageIn, imageFilter->fftDims, imageDims, imageFilter->kernelCentre);
			} break;

		case 2:
			{
				LaunchCudaKernel_Simple_Stream(PadImage_Cuda_u16, imageFilter->fftStream, imageFilter->paddedImage, (u16 *)imageIn, imageFilter->fftDims, imageDims, imageFilter->kernelCentre);
			} break;

		case 1:
			{
				LaunchCudaKernel_Simple_Stream(PadImage_Cuda_u08, imageFilter->fftStream, imageFilter->paddedImage, imageIn, imageFilter->fftDims, imageDims, imageFilter->kernelCentre);
			} break;
	}
}

void
UnpadImage(image_filter *imageFilter, dim3 imageDims)
{
	LaunchCudaKernel_Simple_Stream(UnpadImage_Cuda, imageFilter->fftStream, imageFilter->paddedImage, imageFilter->imageOut, imageFilter->fftDims, imageDims);	
}

void
ModulateAndNormalise(image_filter *imageFilter)
{
	LaunchCudaKernel_Simple_Stream(ModulateAndNormalise_Cuda, imageFilter->fftStream, imageFilter->imageSpectrum, imageFilter->kernelSpectrum, FFTDataSize(imageFilter->fftDims), FFTNormalisationConstant(imageFilter->fftDims));
}

void
FilterImage(image_filter *imageFilter, u08 *imageIn, dim3 imageDims, u32 bytesPerPixel)
{
	PadImage(imageFilter, imageIn, imageDims, bytesPerPixel);
	
	cufftExecR2C(imageFilter->fftPlanFwd, (cufftReal *)imageFilter->paddedImage, (cufftComplex *)imageFilter->imageSpectrum);
	ModulateAndNormalise(imageFilter);
	cufftExecC2R(imageFilter->fftPlanInv, (cufftComplex *)imageFilter->imageSpectrum, (cufftReal *)imageFilter->paddedImage);

	UnpadImage(imageFilter, imageDims);
	
	cudaStreamSynchronize(imageFilter->fftStream);
}

void
CalculateNewBackground(image_data *im, slab_background *slabBackground, u32 actualZSize, f32 ballCentreHeight, u32 bytesPerPixel, image_filter *imageFilter)
{
	u32 *bg = slabBackground->background;
	
	dim3 imageDims = im->vol.dims;
	imageDims.z = actualZSize;

	dim3 ballGridDims;
	ballGridDims.x = slabBackground->slabDims.x;
	ballGridDims.y = slabBackground->slabDims.y;
	ballGridDims.z = 1;

	LaunchCudaKernel_Simple_Stream(ResetBackground, *Reading_Stream, bg, ballGridDims);
	
	FilterImage(imageFilter, im->image, imageDims, bytesPerPixel);

	LaunchCudaKernel_Simple_Stream(CalculateBackground_Cuda_f32, *Reading_Stream, imageFilter->imageOut, bg, ballCentreHeight, imageDims, ballGridDims, slabBackground->ballVol->dims);
	
	cudaStreamSynchronize(*Reading_Stream);
}

struct
reading_thread_data_in
{
	image_data *im;
	image_loader *loader;
	slab_background *bg;
	f32 ballCentreHeight;
	u32 bytesPerPixel;
	image_filter *filter;
};

reading_thread_data_in *ReadingThread_DataIn;
threadSig *ReadingThreadContinueSignal;
threadSig *ReadingThreadRunSignal;

void
FillSingleBuffer_threaded(image_data *im, image_loader *loader, slab_background *bg, f32 ballCentreHeight, u32 bytesPerPixel, image_filter *filter)
{
	while (*ReadingThreadContinueSignal) {}
	FenceIn(*ReadingThreadContinueSignal = 1);
	LockMutex(ReadingThread_Mutex);
	SignalCondition(ReadingThread_Cond);
		
	ReadingThread_DataIn->im = im;
	ReadingThread_DataIn->loader = loader;
	ReadingThread_DataIn->bg = bg;
	ReadingThread_DataIn->ballCentreHeight = ballCentreHeight;
	ReadingThread_DataIn->bytesPerPixel = bytesPerPixel;
	ReadingThread_DataIn->filter = filter;

	UnlockMutex(ReadingThread_Mutex);
}

void
FillSingleBuffer(image_data *im, image_loader *loader, slab_background *slabBackground, f32 ballCentreHeight, u32 bytesPerPixel, image_filter *filter)
{
	u32 nBytesPlane = im->vol.dims.x * im->vol.dims.y * bytesPerPixel;
	u32 localactualCapacity = 0;

	for (	;
		localactualCapacity < im->vol.dims.z;
		++localactualCapacity)
	{
		if(!LoadCurrentImageAndAdvanceIndex(loader, im->image + (localactualCapacity * nBytesPlane))) break;
	}

	if (localactualCapacity)
	{
		CalculateNewBackground(im, slabBackground, localactualCapacity, ballCentreHeight, bytesPerPixel, filter);
	}
}

void
FillRollingBuffer(rolling_image_buffer *buffer, f32 ballCentreHeight, u32 bytesPerPixel)
{
	FillSingleBuffer_threaded(buffer->buffers[0], buffer->loader, buffer->slabBackground->slabBackground, ballCentreHeight, bytesPerPixel, buffer->filter);
	FillSingleBuffer_threaded(buffer->buffers[1], buffer->loader, buffer->slabBackground->next->slabBackground, ballCentreHeight, bytesPerPixel, buffer->filter);
}

u08*
GetImagePlane(rolling_image_buffer *buffer, u32 zIndex, f32 ballCentreHeight, u32 bytesPerPixel)
{
	u32 bufferOffset = zIndex - buffer->currentZStart;
	u32 bin = (2 * bufferOffset) / buffer->capacity;
	bufferOffset -= (bin * (buffer->capacity / 2));
	image_filter *filter = buffer->filter;

	u08 *result = buffer->buffers[bin]->image + (bufferOffset * bytesPerPixel * buffer->fullImageVolume.dims.x * buffer->fullImageVolume.dims.y);

	if (bin)
	{
		buffer->currentZStart += (buffer->capacity / 2);
		
		image_data *tmp;
		tmp = buffer->buffers[0];
		buffer->buffers[0] = buffer->buffers[1];
		buffer->buffers[1] = tmp;

		buffer->slabBackground = buffer->slabBackground->next;

		FillSingleBuffer_threaded(buffer->buffers[1], buffer->loader, buffer->slabBackground->next->slabBackground, ballCentreHeight, bytesPerPixel, filter);
	}

	return(result);
}

void
FillBackground(interpolation_background *interpolationBackground, rolling_image_buffer *imageBuffer)
{
	three_d_volume *imVol = GetImageVolume_FromImageBuffer(imageBuffer);

	u32 atStart = interpolationBackground->slabCounter == 0;
	u32 atOne = interpolationBackground->slabCounter == 1;
	u32 atEnd = interpolationBackground->slabCounter == IntDivideCeil(imVol->dims.z, interpolationBackground->slabBackground->slabBackground->ballVol->dims.z);

	++interpolationBackground->slabCounter;

	if (atStart)
	{
	
	}
	else if (atOne)
	{
		interpolationBackground->doInterpolation = 1;
	}
	else
	{
		if (atEnd)
		{
			interpolationBackground->doInterpolation = 0;
		}

		interpolationBackground->slabBackground = interpolationBackground->slabBackground->next;
	}
}

void
CreateInterpolationBackground(interpolation_background *interpolationBackground, u32 x, u32 y, three_d_volume *ballVol, dim3 ballRadius)
{
	CreateThreeDVol_FromDim(ballVol, ballRadius);
	
	uint2 slabDims;
	slabDims.x = IntDivideCeil(x, ballVol->dims.x);
	slabDims.y = IntDivideCeil(y, ballVol->dims.y);

	InitialiseInterpolationBackground(interpolationBackground, slabDims, ballVol);	
}

void
CalcResultImage(rolling_image_buffer *imInBuffer, image_data *imDataOut, interpolation_background *interpolationBackground, u32 currentZ, f32 interpDis, u32 bytesPerPixel, f32 ballCentreHeight)
{
	u08 *imIn = GetImagePlane(imInBuffer, currentZ, ballCentreHeight, bytesPerPixel);
	u08 *imOut = imDataOut->image;
	
	u32 *background_1 = interpolationBackground->slabBackground->slabBackground->background;
	u32 *background_2;
	f32 interpDis_local;
	if (interpolationBackground->doInterpolation)
	{
		background_2 = interpolationBackground->slabBackground->next->slabBackground->background;
		interpDis_local = interpDis;
	}
	else
	{
		background_2 = background_1;
		interpDis_local = 0;
	}

	three_d_volume *imVol = GetImageVolume_FromImageBuffer(imInBuffer);
	dim3 imageDims = imVol->dims;
	imageDims.z = 1;
	dim3 localBallGridDims;
	localBallGridDims.x = interpolationBackground->slabBackground->slabBackground->slabDims.x;
	localBallGridDims.y = interpolationBackground->slabBackground->slabBackground->slabDims.y;
	localBallGridDims.z = 1;

	switch (bytesPerPixel)
	{
		case 4:
			{
				LaunchCudaKernel_Simple_Stream(RemoveBackground_Cuda_u32, *Processing_Stream, (u32 *)imIn, (u32 *)imOut, background_1, background_2, ballCentreHeight, interpDis_local, imageDims, localBallGridDims, interpolationBackground->slabBackground->slabBackground->ballVol->dims);
			} break;

		case 2:
			{
				LaunchCudaKernel_Simple_Stream(RemoveBackground_Cuda_u16, *Processing_Stream, (u16 *)imIn, (u16 *)imOut, background_1, background_2, ballCentreHeight, interpDis_local, imageDims, localBallGridDims, interpolationBackground->slabBackground->slabBackground->ballVol->dims);
			} break;

		case 1:
			{
				LaunchCudaKernel_Simple_Stream(RemoveBackground_Cuda_u08, *Processing_Stream, (u08 *)imIn, (u08 *)imOut, background_1, background_2, ballCentreHeight, interpDis_local, imageDims, localBallGridDims, interpolationBackground->slabBackground->slabBackground->ballVol->dims);
			} break;
	}
	cudaStreamSynchronize(*Processing_Stream);
}

void *
ReadingThreadFunc(void *dataIn)
{	
	reading_thread_data_in *readingThreadDataIn;
	readingThreadDataIn = (reading_thread_data_in *)dataIn;

	while (*ReadingThreadRunSignal)
	{
		LockMutex(ReadingThread_Mutex);
		FenceIn(*ReadingThreadContinueSignal = 0);
		WaitOnCond(ReadingThread_Cond, ReadingThread_Mutex);
		UnlockMutex(ReadingThread_Mutex);
		
		if (*ReadingThreadRunSignal)
		{
			FillSingleBuffer(readingThreadDataIn->im, readingThreadDataIn->loader, readingThreadDataIn->bg, readingThreadDataIn->ballCentreHeight, readingThreadDataIn->bytesPerPixel, readingThreadDataIn->filter);
		}
	}

	return(NULL);
}

struct
write_buffer_data
{
	image_data *im;
	u32 threadIndex;
	memory_arena *zlibArena_comp;
	SSIF_file_for_writing *SSIFfile_writing;
	u08 *compBuffer;
	image_file_coords coordsToWrite;
};

void
InitialiseWriteBufferData(write_buffer_data *data, u32 threadIndex, memory_arena *zlibArena_comp, SSIF_file_for_writing *SSIFfile_writing, u08 *compBuffer)
{
	data->threadIndex = threadIndex;
	data->zlibArena_comp = zlibArena_comp;
	data->SSIFfile_writing = SSIFfile_writing;
	data->compBuffer = compBuffer;
}

struct
write_buffer_node
{
	write_buffer_data *data;
	write_buffer_node *next;
	write_buffer_node *prev;
};

struct
write_buffer
{
	threadSig nFreeNodes;
	write_buffer_node *bufferHead;
};

write_buffer *WriteBuffer;
CreateMutex(WriteBufferMutex);

write_buffer_node *SavingThread_DataIn[NSavingThreads];
threadSig *SavingThreadContinueSignal[NSavingThreads];
threadSig *SavingThreadRunSignal[NSavingThreads];

void
WriteBuffer_InsertAfter(write_buffer_node *head, write_buffer_node *tail)
{
	tail->next = head->next;
	tail->next->prev = tail;
	head->next = tail;
	tail->prev = head;
}

void
WriteBuffer_InsertBefore(write_buffer_node *head, write_buffer_node *tail)
{
	tail->prev->next = head;
	head->prev = tail->prev;
	head->next = tail;
	tail->prev = head;
}

void
WriteBuffer_Remove(write_buffer_node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
}

void
WriteBuffer_InsertAtStart(write_buffer *buff, write_buffer_node *node)
{
	WriteBuffer_InsertAfter(buff->bufferHead, node);
}

write_buffer_node*
WriteBuffer_RemoveFromEnd(write_buffer *buff)
{
	write_buffer_node *node = buff->bufferHead->prev;
	WriteBuffer_Remove(node);

	return(node);
}

void
SaveResultImage_threaded(image_data *im, image_file_coords coords, u32 threadIndex)
{
	while (*SavingThreadContinueSignal[threadIndex]) {}
	FenceIn(*SavingThreadContinueSignal[threadIndex] = 1);
	LockMutex(SavingThread_Mutex[threadIndex]);
	SignalCondition(SavingThread_Cond[threadIndex]);
		
	SavingThread_DataIn[threadIndex]->data->im = im;
	SavingThread_DataIn[threadIndex]->data->coordsToWrite = coords;
	UnlockMutex(SavingThread_Mutex[threadIndex]);
}

image_data *
AddDataToWriteBuffer(write_buffer *buff, image_data *im, image_file_coords coords)
{
	while(buff->nFreeNodes == 0) {}

	LockMutex(WriteBufferMutex);
	write_buffer_node *node = WriteBuffer_RemoveFromEnd(buff);
	image_data *freeBuffer = node->data->im;
	--buff->nFreeNodes;
	UnlockMutex(WriteBufferMutex);

	SaveResultImage_threaded(im, coords, node->data->threadIndex);

	return(freeBuffer);
}

void
SaveResultImage(write_buffer_data *data)
{
	while(!WriteImageToSSIFFile(data->zlibArena_comp, data->SSIFfile_writing, data->im->image, data->im->nBytes, data->compBuffer, &data->coordsToWrite)) {}
}

void *
SavingThreadFunc(void *dataIn)
{
	write_buffer_node *savingThreadDataIn;
	savingThreadDataIn = (write_buffer_node *)dataIn;

	while (*SavingThreadRunSignal[savingThreadDataIn->data->threadIndex])
	{
		LockMutex(SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		FenceIn(*SavingThreadContinueSignal[savingThreadDataIn->data->threadIndex] = 0);
		WaitOnCond(SavingThread_Cond[savingThreadDataIn->data->threadIndex], SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		UnlockMutex(SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		
		if (*SavingThreadRunSignal[savingThreadDataIn->data->threadIndex])
		{
			SaveResultImage(savingThreadDataIn->data);
			LockMutex(WriteBufferMutex);
			WriteBuffer_InsertAtStart(WriteBuffer, savingThreadDataIn);
			++WriteBuffer->nFreeNodes;
			UnlockMutex(WriteBufferMutex);
		}
	}

	return(NULL);
}

void
ShutDownReadingThread()
{
	while (*ReadingThreadContinueSignal) {}
	FenceIn(*ReadingThreadContinueSignal = 1);
	LockMutex(ReadingThread_Mutex);
	SignalCondition(ReadingThread_Cond);
	
	FenceIn(*ReadingThreadRunSignal = 0);

	UnlockMutex(ReadingThread_Mutex);
}

void
ShutDownSavingThread(u32 index)
{
	while (*SavingThreadContinueSignal[index]) {}
	FenceIn(*SavingThreadContinueSignal[index] = 1);
	LockMutex(SavingThread_Mutex[index]);
	SignalCondition(SavingThread_Cond[index]);
	
	FenceIn(*SavingThreadRunSignal[index] = 0);

	UnlockMutex(SavingThread_Mutex[index]);
}

void
CreateWriteBuffer(write_buffer *buff)
{
	buff->bufferHead->next = buff->bufferHead->prev = buff->bufferHead;

	buff->nFreeNodes = NSavingThreads;
}

void
AddNodesToWriteBuffer(write_buffer *buff)
{
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		WriteBuffer_InsertAtStart(buff, SavingThread_DataIn[index]);
	}
}

SSIF_file_for_writing *
CreateOutputFile(memory_arena *arena, image_loader *loader, program_arguments *pArgs)
{
	char *inputFile = pArgs->inputFile;
	
	SSIF_file_header *newHeader = PushStructP(arena, SSIF_file_header);
	u32 nameLen = CopySSIFHeader(loader->SSIFfile->SSIFfile->header, newHeader, loader->SSIFfile->SSIFfile->header->name);
	CopySSIFHeaderName((char *)"_bgSub\0", newHeader->name + nameLen, nameLen);

	newHeader->depth = (u32)pArgs->zRange.end + 1 - (u32)pArgs->zRange.start;
	newHeader->timepoints = (u32)pArgs->tRange.end + 1 - (u32)pArgs->tRange.start;
	newHeader->channels = (u32)pArgs->cRange.end + 1 - (u32)pArgs->cRange.start;

	switch (pArgs->track)
	{
		case track_depth:
			{
				newHeader->packingOrder = ZTC;		
			} break;

		case track_timepoint:
			{
				newHeader->packingOrder = TZC;
			} break;

		case track_channel:
			{
				newHeader->packingOrder = CZT;
			} break;
	}

	char buff[256];
	u32 fileNameLength = CopyNullTerminatedString(inputFile, buff);
	CopyNullTerminatedString((char *)"_bgSub.ssif\0", buff + fileNameLength - 5);

	SSIF_file_for_writing *SSIFfile_writing = OpenSSIFFileForWriting(arena, newHeader, buff);

	return(SSIFfile_writing);
}

u32
ParseInputParams(program_arguments *programArguments, u32 nArgs, const char **args)
{
	u32 sucsess = 0;
	
	u32 ballRadiusSet = 0;
	u32 zRangeSet = 0;
	u32 tRangeSet = 0;
	u32 cRangeSet = 0;
	u32 trackSet = 0; 

	CopyNullTerminatedString((char *)*args, programArguments->inputFile);
	
	for (	u32 index = 1;
		index < nArgs;
		++index )
	{
		if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--ballRadius\0"))
		{
			ballRadiusSet = 1;
			string_to_int_result parse_x = StringToInt((char *)*(args + index + 1));		
			string_to_int_result parse_y = StringToInt((char *)*(args + index + 2));
			string_to_int_result parse_z = StringToInt((char *)*(args + index + 3));
			programArguments->ballRadius.x = parse_x.integerValue;
			programArguments->ballRadius.y = parse_y.integerValue;
			programArguments->ballRadius.z = parse_z.integerValue;

			index += 3;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--zRange\0"))
		{
			zRangeSet = 1;
			string_to_int_result parse_zRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_zRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->zRange.start = (s32)parse_zRange_start.integerValue;
			programArguments->zRange.end = (s32)parse_zRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--tRange\0"))
		{
			tRangeSet = 1;
			string_to_int_result parse_tRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_tRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->tRange.start = (s32)parse_tRange_start.integerValue;
			programArguments->tRange.end = (s32)parse_tRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--cRange\0"))
		{
			cRangeSet = 1;
			string_to_int_result parse_cRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_cRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->cRange.start = (s32)parse_cRange_start.integerValue;
			programArguments->cRange.end = (s32)parse_cRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--track\0"))
		{
			if (AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"z\0"))
			{
				trackSet = 1;
				programArguments->track = track_depth;
			}
			else if(AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"t\0"))
			{
				trackSet = 1;
				programArguments->track = track_timepoint;
			}
			else if(AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"c\0"))
			{
				trackSet = 1;
				programArguments->track = track_channel;
			}

			index += 1;
		}
	}

	sucsess = ballRadiusSet; // must set ballRadius

	if (!zRangeSet)
	{
		programArguments->zRange.E = {0, -1};
	}
	if (!tRangeSet)
	{
		programArguments->tRange.E = {0, -1};
	}
	if (!cRangeSet)
	{
		programArguments->cRange.E = {0, -1};
	}
	if (!trackSet)
	{
		programArguments->track = track_timepoint;
	}

	return(sucsess);
}

struct
rolling_ball
{
	f32 height;
	dim3 radius;
};

void
CreateRollingBall(rolling_ball *ball, dim3 fullRadius)
{
	ball->height = (f32)Max(Max(fullRadius.x, fullRadius.y), fullRadius.z);

	f32 arcTrim;
        if (ball->height <= 10)
        {
        	arcTrim = 0.24; // trim 24%
        }
        else if (ball->height <= 100)
        {
                arcTrim = 0.32; // trim 32%
        }
        else
        {
                arcTrim = 0.40; // trim 40%
        }

	ball->radius.x = (u32)((arcTrim * (f32)fullRadius.x) + 0.5);
	ball->radius.y = (u32)((arcTrim * (f32)fullRadius.y) + 0.5);
	ball->radius.z = (u32)((arcTrim * (f32)fullRadius.z) + 0.5);
}

MainArgs
{
	memory_arena arena;
	CreateMemoryArena(arena, MegaByte(8 * (NSavingThreads + 1)));

	program_arguments *pArgs = PushStruct(arena, program_arguments);
	pArgs->inputFile = PushArray(arena, char, 256);
	if (!ParseInputParams(pArgs, ArgCount - 1, ArgBuffer + 1))
	{
		printf("--ballRadius must be supplied\n");
		return(1);
	}

	Reading_Stream = PushStruct(arena, stream);
	Processing_Stream = PushStruct(arena, stream);
	cudaStreamCreate(Reading_Stream);
	cudaStreamCreate(Processing_Stream);
	
	ReadingThreadContinueSignal = PushStruct(arena, threadSig);
	ReadingThreadRunSignal = PushStruct(arena, threadSig);	
	FenceIn(*ReadingThreadContinueSignal = 1);
	FenceIn(*ReadingThreadRunSignal = 1);
	ReadingThread_DataIn = PushStruct(arena, reading_thread_data_in);
	ReadingThread_Mutex = PushStruct(arena, mutex);
	ReadingThread_Cond = PushStruct(arena, cond);
	InitialiseMutex(ReadingThread_Mutex);
	InitialiseCond(ReadingThread_Cond);
	Reading_Thread = PushStruct(arena, thread);
	LaunchThread(Reading_Thread, ReadingThreadFunc, (void *)ReadingThread_DataIn);
	
	image_loader *imageLoader = PushStruct(arena, image_loader);
	
	memory_arena *zlibDecompArena = PushSubArena(arena, KiloByte(64));
	CreateImageLoader(&arena, zlibDecompArena, imageLoader, pArgs);

	SSIF_file_for_writing *SSIFfile_writing = CreateOutputFile(&arena, imageLoader, pArgs);

	WriteBuffer = PushStruct(arena, write_buffer);
	write_buffer_node *head = PushStruct(arena, write_buffer_node);
	WriteBuffer->bufferHead = head;
	CreateWriteBuffer(WriteBuffer);
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		SavingThread_DataIn[index] = PushStruct(arena, write_buffer_node);
		
		write_buffer_data *writeBufferData = PushStruct(arena, write_buffer_data);
		
		memory_arena *zlibCompArena = PushSubArena(arena, KiloByte(512));
		u08 *compBuffer = PushArray(arena, u08, 2 * GetSSIFBytesPerImage(imageLoader->SSIFfile->SSIFfile));

		InitialiseWriteBufferData(writeBufferData, index, zlibCompArena, SSIFfile_writing, compBuffer);
		SavingThread_DataIn[index]->data = writeBufferData;

		SavingThreadContinueSignal[index] = PushStruct(arena, threadSig);
		SavingThreadRunSignal[index] = PushStruct(arena, threadSig);
		FenceIn(*SavingThreadContinueSignal[index] = 1);
		FenceIn(*SavingThreadRunSignal[index] = 1);
		SavingThread_Mutex[index] = PushStruct(arena, mutex);
		SavingThread_Cond[index] = PushStruct(arena, cond);
		InitialiseMutex(SavingThread_Mutex[index]);
		InitialiseCond(SavingThread_Cond[index]);
		Saving_Thread[index] = PushStruct(arena, thread);
		LaunchThread(Saving_Thread[index], SavingThreadFunc, (void *)SavingThread_DataIn[index]);
	}
	AddNodesToWriteBuffer(WriteBuffer);

	WriteBufferMutex = PushStruct(arena, mutex);
	InitialiseMutex(WriteBufferMutex);

	u32 bytesPerPixel = GetSSIFBytesPerPixel(imageLoader->SSIFfile->SSIFfile);

	u32 NSavingBuffers = NSavingThreads + 1;
	image_data *result = PushArray(arena, image_data, NSavingBuffers);
	for (	u32 index = 0;
		index < NSavingBuffers;
		++index )
	{
		CreateImageData_FromInt(result + index, ImageLoaderGetWidth(imageLoader), ImageLoaderGetHeight(imageLoader), 1, bytesPerPixel);
	}
	u32 currentResultIndex = 0;
	image_data *currentResultBuffer, *freeBuffer;

	interpolation_background *interpolationBackground = PushStruct(arena, interpolation_background);
	slab_background *slabBackground = PushArray(arena, slab_background, 3);
	slab_background_LL_node *nodes = PushArray(arena, slab_background_LL_node, 3);

	nodes->slabBackground = slabBackground;
	(nodes + 1)->slabBackground = slabBackground + 1;
	(nodes + 2)->slabBackground = slabBackground + 2;

	interpolationBackground->slabBackground = nodes;
	three_d_volume *ballVol = PushStruct(arena, three_d_volume);
	
	rolling_ball *ball = PushStruct(arena, rolling_ball);
	CreateRollingBall(ball, pArgs->ballRadius);
	
	CreateInterpolationBackground(interpolationBackground, ImageLoaderGetWidth(imageLoader), ImageLoaderGetHeight(imageLoader), ballVol, ball->radius);

	InitialiseSlabBackgroundLLNode(nodes + 1, interpolationBackground->slabBackground->slabBackground->slabDims, ballVol);
	InitialiseSlabBackgroundLLNode(nodes + 2, interpolationBackground->slabBackground->slabBackground->slabDims, ballVol);

	SlabBackgroundLLAddAfter(nodes, nodes + 1);
	SlabBackgroundLLAddAfter(nodes + 1, nodes + 2);
	SlabBackgroundLLAddAfter(nodes + 2, nodes);

	u32 localZ = ballVol->dims.z;
	u32 halfLocalZ = localZ / 2;

	image_data *inputIm = PushArray(arena, image_data, 2);
	image_filter *filter = PushStruct(arena, image_filter);
	rolling_image_buffer *imageBuffer = PushStruct(arena, rolling_image_buffer);
	imageBuffer->slabBackground = nodes;
	imageBuffer->filter = filter;
	{
		f32 kernel[125] = {0.00708348919034, 0.00752150769936, 0.00767345223391, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00767345223391, 0.00814795202014, 0.00831255156942, 0.00814795202014, 0.00767345223391, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00767345223391, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00798661176031, 0.00848047625016, 0.00865179323448, 0.00848047625016, 0.00798661176031, 0.00814795202014, 0.00865179323448, 0.00882657105145, 0.00865179323448, 0.00814795202014, 0.00798661176031, 0.00848047625016, 0.00865179323448, 0.00848047625016, 0.00798661176031, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00767345223391, 0.00814795202014, 0.00831255156942, 0.00814795202014, 0.00767345223391, 0.00814795202014, 0.00865179323448, 0.00882657105145, 0.00865179323448, 0.00814795202014, 0.00831255156942, 0.00882657105145, 0.00900487961453, 0.00882657105145, 0.00831255156942, 0.00814795202014, 0.00865179323448, 0.00882657105145, 0.00865179323448, 0.00814795202014, 0.00767345223391, 0.00814795202014, 0.00831255156942, 0.00814795202014, 0.00767345223391, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00798661176031, 0.00848047625016, 0.00865179323448, 0.00848047625016, 0.00798661176031, 0.00814795202014, 0.00865179323448, 0.00882657105145, 0.00865179323448, 0.00814795202014, 0.00798661176031, 0.00848047625016, 0.00865179323448, 0.00848047625016, 0.00798661176031, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00767345223391, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00767345223391, 0.00814795202014, 0.00831255156942, 0.00814795202014, 0.00767345223391, 0.00752150769936, 0.00798661176031, 0.00814795202014, 0.00798661176031, 0.00752150769936, 0.00708348919034, 0.00752150769936, 0.00767345223391, 0.00752150769936, 0.00708348919034};

		dim3 kernelDims = dim3(5, 5, 5);
		dim3 kernelCentre = dim3(2, 2, 2);

		CreateRollingImageBuffer(imageBuffer, imageLoader, inputIm, inputIm + 1, localZ, bytesPerPixel, kernel, kernelDims, kernelCentre);
	}
	u32 nImagesTrack;
	u32 *startingIndex = GetImageLoaderCurrentTrackIndexAndSize(imageLoader, &nImagesTrack);
	nImagesTrack -= *startingIndex;

	u32 nImagesToLoad = (ImageLoaderGetChannels(imageLoader) - imageLoader->channel) * (ImageLoaderGetTimePoints(imageLoader) - imageLoader->timepoint) * (ImageLoaderGetDepth(imageLoader) - imageLoader->depth);	

	u32 currentInterpStart;

	for (	u32 index = 0;
		index < nImagesToLoad;
		++index )
	{
		u32 localIndex = index % nImagesTrack;
		
		if (localIndex == 0)
		{
			if (index)
			{
				switch (imageLoader->track)
				{
					case track_depth:
						{
							imageLoader->depth = (u32)pArgs->zRange.start;
						} break;

					case track_timepoint:
						{
							imageLoader->timepoint = (u32)pArgs->tRange.start;
						} break;

					case track_channel:
						{
							imageLoader->channel = (u32)pArgs->cRange.start;
						} break;
				}
				
				switch (SSIFfile_writing->SSIFfile->header->packingOrder)
				{
					case ZTC:
						{
							if (++imageLoader->timepoint == ImageLoaderGetTimePoints(imageLoader))
							{
								imageLoader->timepoint = (u32)pArgs->tRange.start;
								++imageLoader->channel;
							}
						} break;
					
					case CTZ:
						{
							if (++imageLoader->timepoint == ImageLoaderGetTimePoints(imageLoader))
							{
								imageLoader->timepoint = (u32)pArgs->tRange.start;
								++imageLoader->depth;
							}
						} break;
					
					case TZC:
						{
							if (++imageLoader->depth == ImageLoaderGetDepth(imageLoader))
							{
								imageLoader->depth = (u32)pArgs->zRange.start;
								++imageLoader->channel;
							}
						} break;
					
					case CZT:
						{
							if (++imageLoader->depth == ImageLoaderGetDepth(imageLoader))
							{
								imageLoader->depth = (u32)pArgs->zRange.start;
								++imageLoader->timepoint;
							}
						} break;

					case ZCT:
						{
							if (++imageLoader->channel == ImageLoaderGetChannels(imageLoader))
							{
								imageLoader->channel = (u32)pArgs->cRange.start;
								++imageLoader->timepoint;
							}
						} break;
					
					case TCZ:
						{
							if (++imageLoader->channel == ImageLoaderGetChannels(imageLoader))
							{
								imageLoader->channel = (u32)pArgs->cRange.start;
								++imageLoader->depth;
							}
						} break;
				}
			}
			
			currentInterpStart = 0;
			
			interpolationBackground->slabBackground = nodes;
			interpolationBackground->slabCounter = 0;
			interpolationBackground->doInterpolation = 0;
			
			imageBuffer->slabBackground = nodes;
			imageBuffer->currentZStart = 0;

			FillRollingBuffer(imageBuffer, ball->height, bytesPerPixel);
			FillBackground(interpolationBackground, imageBuffer);
		}
		
		if (currentResultIndex < NSavingBuffers)
		{
			currentResultBuffer = result + currentResultIndex++;
		}
		else
		{
			currentResultBuffer = freeBuffer;
		}
	
		if ((localIndex >= halfLocalZ) && (((localIndex - halfLocalZ) % localZ) == 0))
		{
			FillBackground(interpolationBackground, imageBuffer);
			currentInterpStart = localIndex;
		}

		f32 interpDis = (f32)(localIndex - currentInterpStart) / (f32)localZ;
		CalcResultImage(imageBuffer, currentResultBuffer, interpolationBackground, localIndex, interpDis, bytesPerPixel, ball->height);

		image_file_coords coords = LinearIndexToImageCoords(index, SSIFfile_writing->SSIFfile->header);

		freeBuffer = AddDataToWriteBuffer(WriteBuffer, currentResultBuffer, coords);
		
		printf("\r%1.2f%% complete...", 100.0*(f32)(index + 1)/(f32)nImagesToLoad);
		fflush(stdout);
	}
	
	ShutDownReadingThread();
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		ShutDownSavingThread(index);
	}
	WaitForThread(Reading_Thread);
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		WaitForThread(Saving_Thread[index]);
	}

	FreeRollingImageBuffer(imageBuffer);
	for (	u32 index = 0;
		index < NSavingBuffers;
		++index )
	{
		FreeImageData(result + index);
	}
	CloseSSIFFile(SSIFfile_writing->SSIFfile);
	FreeSlabBackgroundLLNode(nodes);
	FreeSlabBackgroundLLNode(nodes + 1);
	FreeSlabBackgroundLLNode(nodes + 2);
	FreeMemoryArena(arena);

	EndMain;
}
