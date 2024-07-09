package com.example.ocr_llm
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
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

    // Adjust these values based on your TrOCR model's requirements
    private val imageSize = 224
    private val numChannels = 3
    private val maxOutputLength = 128 // Adjust based on your model's output

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView = findViewById(R.id.inputImageView)
        resultTextView = findViewById(R.id.resultTextView)
        processButton = findViewById(R.id.processButton)

        try {
            trOcrInterpreter = Interpreter(loadModelFile("trocr_model.tflite"))
        } catch (e: Exception) {
            e.printStackTrace()
            resultTextView.text = "Error loading model: ${e.message}"
        }

        processButton.setOnClickListener { performOCR() }
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
            val outputBuffer = ByteBuffer.allocateDirect(maxOutputLength * 4).order(ByteOrder.nativeOrder())

            trOcrInterpreter.run(inputBuffer, outputBuffer)

            val result = postprocessOutput(outputBuffer)
            resultTextView.text = result
        } else {
            resultTextView.text = "Error: Could not get bitmap from image view"
        }
    }

    private fun getBitmapFromImageView(): Bitmap? {
        // In a real app, you'd implement logic to get the actual bitmap
        // This is just a placeholder using a sample image from assets
        return try {
            val inputStream = assets.open("image.jpg")



















            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(imageSize * imageSize * numChannels * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(imageSize * imageSize)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)

        inputBuffer.rewind()
        for (pixelValue in intValues) {
            inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) - 128) / 128f)
            inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) - 128) / 128f)
            inputBuffer.putFloat(((pixelValue and 0xFF) - 128) / 128f)
        }

        return inputBuffer
    }

    private fun postprocessOutput(outputBuffer: ByteBuffer): String {
        outputBuffer.rewind()
        val outputArray = FloatArray(maxOutputLength)
        outputBuffer.asFloatBuffer().get(outputArray)

        // This is a placeholder. You'll need to implement the actual decoding
        // based on your model's output format (e.g., token IDs to text)
        return outputArray.joinToString(", ")
    }
}