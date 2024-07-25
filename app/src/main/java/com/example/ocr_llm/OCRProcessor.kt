package com.example.ocr_llm

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.*
import org.opencv.dnn.Dnn.NMSBoxesRotated
import org.opencv.imgproc.Imgproc.boxPoints
import org.opencv.imgproc.Imgproc.getPerspectiveTransform
import org.opencv.imgproc.Imgproc.warpPerspective
import org.opencv.utils.Converters.vector_RotatedRect_to_Mat
import org.opencv.utils.Converters.vector_float_to_Mat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.HashMap

class OCRProcessor(private val trOcrDetector: Interpreter, private val trOcrRecognizer: Interpreter) {
    private lateinit var indicesMat: MatOfInt
    private val alphabets = "0123456789abcdefghijklmnopqrstuvwxyz"
    private val displayImageSize = 257
    private val detectionImageHeight = 320
    private val detectionImageWidth = 320
    private val detectionImageMeans = floatArrayOf(103.94.toFloat(), 116.78.toFloat(), 123.68.toFloat())
    private val detectionImageStds = floatArrayOf(1.toFloat(), 1.toFloat(), 1.toFloat())
    private val detectionOutputNumRows = 80
    private val detectionOutputNumCols = 80
    private val detectionConfidenceThreshold = 0.5.toFloat()
    private val detectionNMSThreshold = 0.4.toFloat()
    private val recognitionImageHeight = 31
    private val recognitionImageWidth = 200
    private val recognitionImageMean = 0.toFloat()
    private val recognitionImageStd = 255.toFloat()
    private val recognitionModelOutputSize = 48
    private var recognitionResult = ByteBuffer.allocateDirect(recognitionModelOutputSize * 8)
    private lateinit var ocrResults: java.util.HashMap<String, Int>

    fun detectTexts(data: Bitmap): Pair<MatOfRotatedRect, MatOfInt> {
        val detectorTensorImage = preprocessImage(data, detectionImageWidth, detectionImageHeight, detectionImageMeans, detectionImageStds)
        val detectionInputs = arrayOf(detectorTensorImage.buffer.rewind())
        val detectionOutputs: HashMap<Int, Any> = HashMap()

        val ratioHeight = data.height.toFloat() / detectionImageHeight
        val ratioWidth = data.width.toFloat() / detectionImageWidth

        indicesMat = MatOfInt()

        val detectionScores = Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(1) } } }
        val detectionGeometries = Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(5) } } }
        detectionOutputs[0] = detectionScores
        detectionOutputs[1] = detectionGeometries

        trOcrDetector.runForMultipleInputsOutputs(detectionInputs, detectionOutputs)

        val transposedDetectionScores = Array(1) { Array(1) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }
        val transposedDetectionGeometries = Array(1) { Array(5) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }

        for (i in 0 until transposedDetectionScores[0][0].size) {
            for (j in 0 until transposedDetectionScores[0][0][0].size) {
                for (k in 0 until 1) {
                    transposedDetectionScores[0][k][i][j] = detectionScores[0][i][j][k]
                }
                for (k in 0 until 5) {
                    transposedDetectionGeometries[0][k][i][j] = detectionGeometries[0][i][j][k]
                }
            }
        }

        val detRotatedRects = ArrayList<RotatedRect>()
        val detConfidences = ArrayList<Float>()

        for (y in 0 until transposedDetectionScores[0][0].size) {
            val detectionScoreData = transposedDetectionScores[0][0][y]
            val detectionGeometryX0Data = transposedDetectionGeometries[0][0][y]
            val detectionGeometryX1Data = transposedDetectionGeometries[0][1][y]
            val detectionGeometryX2Data = transposedDetectionGeometries[0][2][y]
            val detectionGeometryX3Data = transposedDetectionGeometries[0][3][y]
            val detectionRotationAngleData = transposedDetectionGeometries[0][4][y]

            for (x in 0 until transposedDetectionScores[0][0][0].size) {
                if (detectionScoreData[x] < 0.5) {
                    continue
                }

                val offsetX = x * 4.0
                val offsetY = y * 4.0

                val h = detectionGeometryX0Data[x] + detectionGeometryX2Data[x]
                val w = detectionGeometryX1Data[x] + detectionGeometryX3Data[x]

                val angle = detectionRotationAngleData[x]
                val cos = Math.cos(angle.toDouble())
                val sin = Math.sin(angle.toDouble())

                val offset = Point(
                    offsetX + cos * detectionGeometryX1Data[x] + sin * detectionGeometryX2Data[x],
                    offsetY - sin * detectionGeometryX1Data[x] + cos * detectionGeometryX2Data[x]
                )
                val p1 = Point(-sin * h + offset.x, -cos * h + offset.y)
                val p3 = Point(-cos * w + offset.x, sin * w + offset.y)
                val center = Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y))

                val textDetection = RotatedRect(
                    center,
                    Size(w.toDouble(), h.toDouble()),
                    (-1 * angle * 180.0 / Math.PI)
                )
                detRotatedRects.add(textDetection)
                detConfidences.add(detectionScoreData[x])
            }
        }

        val detConfidencesMat = MatOfFloat(vector_float_to_Mat(detConfidences))
        val boundingBoxesMat = MatOfRotatedRect(vector_RotatedRect_to_Mat(detRotatedRects))

        NMSBoxesRotated(
            boundingBoxesMat,
            detConfidencesMat,
            detectionConfidenceThreshold,
            detectionNMSThreshold,
            indicesMat
        )

        return Pair(boundingBoxesMat, indicesMat)
    }

    fun recognizeTexts(
        data: Bitmap,
        boundingBoxesMat: MatOfRotatedRect,
        indicesMat: MatOfInt
    ): Bitmap {

        recognitionResult.order(ByteOrder.nativeOrder())
        ocrResults = HashMap<String, Int>()

        val bitmapWithBoundingBoxes = data.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmapWithBoundingBoxes)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.toFloat()
        paint.color = Color.GREEN

        val ratioHeight = data.height.toFloat() / detectionImageHeight
        val ratioWidth = data.width.toFloat() / detectionImageWidth

        for (i in indicesMat.toArray()) {
            val boundingBox = boundingBoxesMat.toArray()[i]
            val targetVertices = ArrayList<Point>()
            targetVertices.add(Point(0.toDouble(), (recognitionImageHeight - 1).toDouble()))
            targetVertices.add(Point(0.toDouble(), 0.toDouble()))
            targetVertices.add(Point((recognitionImageWidth - 1).toDouble(), 0.toDouble()))
            targetVertices.add(Point((recognitionImageWidth - 1).toDouble(), (recognitionImageHeight - 1).toDouble()))

            val srcVertices = ArrayList<Point>()
            val boundingBoxPointsMat = Mat()
            boxPoints(boundingBox, boundingBoxPointsMat)
            for (j in 0 until 4) {
                srcVertices.add(
                    Point(
                        boundingBoxPointsMat.get(j, 0)[0] * ratioWidth,
                        boundingBoxPointsMat.get(j, 1)[0] * ratioHeight
                    )
                )
                if (j != 0) {
                    canvas.drawLine(
                        (boundingBoxPointsMat.get(j, 0)[0] * ratioWidth).toFloat(),
                        (boundingBoxPointsMat.get(j, 1)[0] * ratioHeight).toFloat(),
                        (boundingBoxPointsMat.get(j - 1, 0)[0] * ratioWidth).toFloat(),
                        (boundingBoxPointsMat.get(j - 1, 1)[0] * ratioHeight).toFloat(),
                        paint
                    )
                }
            }
            canvas.drawLine(
                (boundingBoxPointsMat.get(0, 0)[0] * ratioWidth).toFloat(),
                (boundingBoxPointsMat.get(0, 1)[0] * ratioHeight).toFloat(),
                (boundingBoxPointsMat.get(3, 0)[0] * ratioWidth).toFloat(),
                (boundingBoxPointsMat.get(3, 1)[0] * ratioHeight).toFloat(),
                paint
            )

            val srcVerticesMat = MatOfPoint2f(srcVertices[0], srcVertices[1], srcVertices[2], srcVertices[3])
            val targetVerticesMat = MatOfPoint2f(targetVertices[0], targetVertices[1], targetVertices[2], targetVertices[3])
            val rotationMatrix = getPerspectiveTransform(srcVerticesMat, targetVerticesMat)
            val recognitionBitmapMat = Mat()
            val srcBitmapMat = Mat()
            bitmapToMat(data, srcBitmapMat)
            warpPerspective(
                srcBitmapMat,
                recognitionBitmapMat,
                rotationMatrix,
                Size(recognitionImageWidth.toDouble(), recognitionImageHeight.toDouble())
            )

            val recognitionBitmap = createEmptyBitmap(
                recognitionImageWidth,
                recognitionImageHeight,
                0,
                Bitmap.Config.ARGB_8888
            )
            matToBitmap(recognitionBitmapMat, recognitionBitmap)

            val recognitionTensorImage = bitmapToTensorImageForRecognition(
                recognitionBitmap,
                recognitionImageWidth,
                recognitionImageHeight,
                recognitionImageMean,
                recognitionImageStd
            )

            recognitionResult.rewind()
            trOcrRecognizer.run(recognitionTensorImage.buffer, recognitionResult)

            var recognizedText = ""
            for (k in 0 until recognitionModelOutputSize) {
                val alphabetIndex = recognitionResult.getInt(k * 8)
                if (alphabetIndex in 0..alphabets.length - 1) recognizedText += alphabets[alphabetIndex]
            }
//            Log.d("Recognition result:", recognizedText)
            if (recognizedText != "") {
                ocrResults[recognizedText] = getRandomColor()
            }
        }
        return bitmapWithBoundingBoxes
    }

    private fun preprocessImage(bitmap: Bitmap, width: Int, height: Int, mean: FloatArray, std: FloatArray): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(mean, std))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)
        return tensorImage
    }

    private fun bitmapToTensorImageForRecognition(
        bitmapIn: Bitmap,
        width: Int,
        height: Int,
        mean: Float,
        std: Float
    ): TensorImage {
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
            .add(TransformToGrayscaleOp())
            .add(NormalizeOp(mean, std))
            .build()
        var tensorImage = TensorImage(DataType.FLOAT32)

        tensorImage.load(bitmapIn)
        tensorImage = imageProcessor.process(tensorImage)

        return tensorImage
    }

    private fun createEmptyBitmap(
        imageWidth: Int,
        imageHeight: Int,
        color: Int = 0,
        imageConfig: Bitmap.Config = Bitmap.Config.RGB_565
    ): Bitmap {
        val ret = Bitmap.createBitmap(imageWidth, imageHeight, imageConfig)
        if (color != 0) {
            ret.eraseColor(color)
        }
        return ret
    }

    fun getRandomColor(): Int {
        val random = Random()
        return Color.argb(
            128,
            (255 * random.nextFloat()).toInt(),
            (255 * random.nextFloat()).toInt(),
            (255 * random.nextFloat()).toInt()
        )
    }

    fun getOcrResults(): HashMap<String, Int> {
        return ocrResults
    }
}
