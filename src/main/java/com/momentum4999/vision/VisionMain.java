package com.momentum4999.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class VisionMain {
    private static final String VIDEO_WINDOW = "nano-vision-video";
    private static final String FIELD_WINDOW = "nano-vision-field";

    public static void main(String[] args) throws IOException {
        String settingsPath = "settings.txt";
        if (args.length >= 1) {
            settingsPath = args[0];
        }

        Properties settings = new Properties();
        try (InputStream settingsFile = new FileInputStream(settingsPath)) {
            settings.load(settingsFile);
        }
        boolean gui = Boolean.parseBoolean(settings.getProperty("gui", "false"));
        boolean fieldGui = Boolean.parseBoolean(settings.getProperty("gui.field", "true"));

        VisionNetworkTable table = new VisionNetworkTable(settings.getProperty("nt.server", "localhost"));
        AprilTagVision aprilTag = new AprilTagVision(table, settings);

        if (gui) {
            HighGui.namedWindow(VIDEO_WINDOW);
            if (fieldGui) {
                HighGui.namedWindow(FIELD_WINDOW);
            }
        }
        VideoCapture video = new VideoCapture(0);
        Mat fieldImage;
        try (InputStream fieldImageFile = VisionMain.class.getResourceAsStream("/field.png")) {
            if (fieldImageFile != null) {
                fieldImage = Imgcodecs.imdecode(new MatOfByte(fieldImageFile.readAllBytes()), Imgcodecs.IMREAD_COLOR);
            } else {
                throw new RuntimeException("Field image resource unavailable");
            }
        }

        Mat videoBuffer = new Mat();
        Mat fieldBuffer = new Mat();
        while (true) {
            video.read(videoBuffer);
            fieldImage.copyTo(fieldBuffer);

            aprilTag.updateVideo(videoBuffer);
            aprilTag.updateField(fieldBuffer);

            if (gui) {
                HighGui.imshow(VIDEO_WINDOW, videoBuffer);
                if (fieldGui) {
                    HighGui.imshow(FIELD_WINDOW, fieldBuffer);
                }
                if (HighGui.waitKey(10) == java.awt.event.KeyEvent.VK_Q) {
                    break;
                }
            }
        }

        HighGui.destroyAllWindows();
        aprilTag.close();
        System.exit(0);
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
}
