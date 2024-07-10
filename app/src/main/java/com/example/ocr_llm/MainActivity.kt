package com.example.ocr_llm
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var trOcrInterpreter: Interpreter
    private lateinit var inputImageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var processButton: Button

    private val imageSize = 320
    private val numChannels = 3
    private val maxOutputLength = 128

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView = findViewById(R.id.inputImageView)
        resultTextView = findViewById(R.id.resultTextView)
        processButton = findViewById(R.id.processButton)

        try {
            trOcrInterpreter = Interpreter(loadModelFile("crnn_dr.tflite"))
            println(trOcrInterpreter.inputTensorCount)
        } catch (e: Exception) {
            e.printStackTrace()
            resultTextView.text = "Error loading model: ${e.message}"
        }

        processButton.setOnClickListener { performOCR() }
    }

    private fun logModelInfo(interpreter: Interpreter) {
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        Log.d("OCR", "Input shape: ${inputTensor.shape().contentToString()}")
        Log.d("OCR", "Input dataType: ${inputTensor.dataType()}")
        Log.d("OCR", "Output shape: ${outputTensor.shape().contentToString()}")
        Log.d("OCR", "Output dataType: ${outputTensor.dataType()}")
    }



    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun performOCR() {
        val bitmap = getBitmapFromImageView()
        if (bitmap != null) {
            val inputBuffer = preprocessImage(bitmap)
            Log.d("OCR", "Input buffer size: ${inputBuffer.capacity()} bytes")

            val inputTensor = trOcrInterpreter.getInputTensor(0)
            val outputTensor = trOcrInterpreter.getOutputTensor(0)
            Log.d("OCR", "Input tensor shape: ${inputTensor.shape().contentToString()}")
            Log.d("OCR", "Output tensor shape: ${outputTensor.shape().contentToString()}")

            val outputSize = outputTensor.numBytes()
            val outputBuffer = ByteBuffer.allocateDirect(outputSize).order(ByteOrder.nativeOrder())

            try {
                trOcrInterpreter.run(inputBuffer, outputBuffer)
                Log.d("OCR", "Model execution successful")

                val result = postprocessOutput(outputBuffer)
                Log.d("OCR", "OCR Result: $result")
                resultTextView.text = result
            } catch (e: Exception) {
                Log.e("OCR", "Error during OCR: ${e.message}", e)
                resultTextView.text = "Error: ${e.message}"
            }
        } else {
            resultTextView.text = "Error: Could not get bitmap from image view"
        }
    }
    private fun getBitmapFromImageView(): Bitmap? {
        return try {
            val inputStream = assets.open("sample_image.jpg")
            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val modelInputWidth = 160
        val modelInputHeight = 80
        val channels = 1

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, true)
        val byteBuffer = ByteBuffer.allocateDirect(modelInputWidth * modelInputHeight * channels)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(modelInputWidth * modelInputHeight)
        resizedBitmap.getPixels(intValues, 0, modelInputWidth, 0, 0, modelInputWidth, modelInputHeight)

        for (pixelValue in intValues) {
            val grayscale = (pixelValue shr 16 and 0xFF) * 0.299f +
                    (pixelValue shr 8 and 0xFF) * 0.587f +
                    (pixelValue and 0xFF) * 0.114f
            byteBuffer.put((grayscale).toInt().toByte())
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    private fun postprocessOutput(outputBuffer: ByteBuffer): String {
        outputBuffer.rewind()
        val byteArray = ByteArray(outputBuffer.remaining())
        outputBuffer.get(byteArray)


        val decodedText = try {
            String(byteArray, Charsets.UTF_8).trim()
        } catch (e: Exception) {
            "Error decoding text: ${e.message}"
        }


        if (decodedText.isBlank()) {
            val intBuffer = outputBuffer.asIntBuffer()
            val intArray = IntArray(intBuffer.remaining())
            intBuffer.get(intArray)


            val sb = StringBuilder()
            for (index in intArray) {
                if (index >= 0 && index < charMap.size) {
                    sb.append(charMap[index])
                }
            }
            return "Extracted Text: ${sb.toString().trim()}"
        }

        return "Extracted Text: $decodedText"
    }

    private val charMap = listOf(" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", ":", "-", "(", ")", "/")
}

