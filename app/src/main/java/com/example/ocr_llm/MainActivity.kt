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
import org.opencv.android.OpenCVLoader
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.time.TimeSource
import com.google.mediapipe.tasks.genai.llminference.LlmInference

class MainActivity : AppCompatActivity() {
    private lateinit var trOcrDetector: Interpreter
    private lateinit var trOcrRecognizer: Interpreter
    private lateinit var ocrProcessor: OCRProcessor
    private lateinit var inputImageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var llmTextView: TextView
    private lateinit var processButton: Button
    private lateinit var llmInference: LlmInference


    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        OpenCVLoader.initDebug()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView = findViewById(R.id.inputImageView)
        resultTextView = findViewById(R.id.resultTextView)
        processButton = findViewById(R.id.processButton)

//        llmInference = LlmInference.createFromOptions(this, options)

        try {
            trOcrDetector = Interpreter(loadModelFile("1.tflite"))
            trOcrRecognizer = Interpreter(loadModelFile("2.tflite"))
            ocrProcessor = OCRProcessor(trOcrDetector, trOcrRecognizer)
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
        val bitmapImage = getBitmapFromImageView()
        if (bitmapImage != null) {
            val timeSource = TimeSource.Monotonic

            val markD1 = timeSource.markNow()

            val detectionResults = ocrProcessor.detectTexts(bitmapImage)

            val markD2 = timeSource.markNow()
            val markD = markD2 - markD1

            Log.d("Time", "Detection Time: $markD")

            val markR1 = timeSource.markNow()
            val resultBitmap = ocrProcessor.recognizeTexts(
                bitmapImage,
                detectionResults.first,
                detectionResults.second
            )
            val markR2 = timeSource.markNow()
            val markR = markR2 - markR1

            Log.d("Time", "Recognition Time: $markR")

            inputImageView.setImageBitmap(resultBitmap)
            val ocrResults = ocrProcessor.getOcrResults()
            resultTextView.text = ocrResults.keys.joinToString("\n")


//            val mark1 = timeSource.markNow()
//            Log.d("Time", "Time 1 Marked")
//
//            val systemPrompt = "Return the following input in JSON format. Make the first word a key, the second word a value, and so on. Input:\n"
//            val inputPrompt = systemPrompt + ocrResults.keys.joinToString("\n")
//            val result = llmInference.generateResponse(inputPrompt)
//            println(result)
//            println(result.substringAfter("Output:").substringBefore("```"))
//
//            val mark2 = timeSource.markNow()
//            Log.d("Time", "Time 2 Marked")
//            val elapsed = mark2 - mark1
//            Log.d("Time", "Elapsed: $elapsed")
//
//            val promptTokenSize = llmInference.sizeInTokens(inputPrompt)
//            val outputTokenSize = llmInference.sizeInTokens(result)
//
//            Log.d("Tokens", "Input prompt token size: $promptTokenSize")
//
//            val basicTokenSpeed = promptTokenSize / elapsed.inWholeSeconds
//
//            Log.d("Time", "Crude Token/s : $basicTokenSpeed")


        } else {
            resultTextView.text = "Error loading image"
        }
    }


    private fun getBitmapFromImageView(): Bitmap? {
        return try {
            val inputStream = assets.open("16.jpg")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            Log.d("OCR", "Image Width: ${bitmap.width}, Image Height: ${bitmap.height}")
            bitmap
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}